"""Orchestrates one input: probe → subtitles → TTS → timing → background → mix → mux."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from . import ffmpeg, hardsubs, levels, mix, mux, separate, subtitles, timing
from .config import cache_dir
from .download import audio_content_key, download, name_from_url, probe_url
from .errors import ExternalToolError, NoSubtitlesError, UsageError
from .probe import MediaInfo, choose_audio_stream, choose_subtitle_stream, probe
from .tts import TTSCache, make_backend, synthesize_all
from .voices import VoiceMap

log = logging.getLogger("senpaidub")
JP_PREROLL_S = 0.3  # remove the Japanese voice this long before each subtitle starts

OUTPUT_SUFFIX = " [EN dub]"


@dataclass
class Options:
    sub_track: int | None = None
    sub_file: Path | None = None
    tts: str = "edge"
    voice: str | None = None
    voice_map: Path | None = None
    separate: bool = True
    duck_db: float = -12.0
    max_speedup: float = 1.4
    include_styles: str | None = None
    exclude_styles: str | None = None
    set_default: bool = False
    output: Path | None = None
    name: str | None = None
    dry_run: bool = False
    keep_temp: bool = False
    force: bool = False
    codec: str = "aac"
    device: str = "auto"
    concurrency: int = 4
    speech_only: bool = False
    speech_gain_db: float = 0.0
    skip_songs: bool = True
    series_name: str | None = None  # names episodes '<series> - 01' and files them as a series
    collect_originals: bool = False  # move the source video + subtitles into <series>/original/
    clean_cache: bool = True  # delete this episode's caches once its dub is written


@dataclass
class Result:
    input: str
    output: Path | None = None
    skipped_reason: str | None = None
    voiced: int = 0
    sped_up: int = 0
    overflowing: int = 0
    tts_cached: int = 0
    tts_new: int = 0
    separation_cached: bool | None = None
    overflow_times: list[int] = field(default_factory=list)
    voice_mismatches: int = 0
    voice_fixed: int = 0
    cache_files: list[Path] = field(default_factory=list)
    subs_file: Path | None = None  # subtitles read off the picture (OCR), filed with the originals


def default_output_name(stem: str, speech_only: bool = False) -> str:
    return f"{stem} [EN speech].wav" if speech_only else f"{stem}{OUTPUT_SUFFIX}.mkv"


def resolve_output(input_stem: str, input_dir: Path, opts: Options) -> Path:
    name = default_output_name(input_stem, opts.speech_only)
    if opts.output is None:
        return input_dir / name
    out = opts.output
    if out.is_dir() or str(out).endswith(("/", os.sep)):
        return out / name
    if opts.speech_only:
        return out.with_suffix(".wav")
    if out.suffix.lower() != ".mkv":
        log.warning("Output is always Matroska; writing %s", out.with_suffix(".mkv").name)
        return out.with_suffix(".mkv")
    return out


def _progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


def _seconds_progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


@contextmanager
def _workdir(keep: bool, console: Console) -> Iterator[Path]:
    d = Path(tempfile.mkdtemp(prefix="senpaidub-"))
    try:
        yield d
    finally:
        if keep:
            console.print(f"[dim]Kept temp files in {d}[/dim]")
        else:
            shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- entry points


def _series_stem(source_name: str, opts: Options) -> str | None:
    if not opts.series_name:
        return None
    from .stremio import output_name

    return output_name(source_name, opts.series_name)


def process_url(url: str, opts: Options, console: Console) -> Result:
    info = probe_url(url)
    stem = opts.name or _series_stem(name_from_url(url, info, None), opts) or name_from_url(url, info, None)
    out = resolve_output(stem, Path.cwd(), opts)
    if out.exists() and not opts.force and not opts.dry_run:
        return Result(url, out, skipped_reason="output exists (use --force to redo)")
    if not info.subtitles and opts.sub_file is None and not hardsubs.available():
        raise NoSubtitlesError(
            "This stream has no embedded subtitles. Download English subtitles for the episode "
            "(e.g. from the Stremio subtitles menu, OpenSubtitles or Jimaku) and pass them with "
            "--sub-file PATH."
        )
    with _workdir(opts.keep_temp, console) as work:
        local = work / "download.mkv"
        with _seconds_progress(console) as p:
            task = p.add_task("Downloading stream", total=info.duration or None)
            download(url, info, local, lambda s: p.update(task, completed=s))
        console.print(f"Downloaded stream → {local.stat().st_size / 1e6:.0f} MB")
        content_key = audio_content_key(local)
        res = _process(local, url, out, opts, console, work, content_key)
        _finish(res, local, opts, console, source_is_temp=True)
        return res


def process_file(path: Path, opts: Options, console: Console) -> Result:
    if not path.is_file():
        raise UsageError(f"Input not found: {path}")
    out = resolve_output(_series_stem(path.name, opts) or path.stem, path.parent, opts)
    if out.resolve() == path.resolve():
        raise UsageError("Output would overwrite the input; pass a different --output.")
    if out.exists() and not opts.force and not opts.dry_run:
        return Result(str(path), out, skipped_reason="output exists (use --force to redo)")
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    with _workdir(opts.keep_temp, console) as work:
        res = _process(path, str(path), out, opts, console, work, key)
        _finish(res, path, opts, console, source_is_temp=False)
        return res


def _finish(res: Result, source: Path, opts: Options, console: Console, *, source_is_temp: bool) -> None:
    """After a successful dub: file the originals with the series and drop this episode's caches."""
    if res.output is None or res.skipped_reason or opts.speech_only:
        return
    if opts.collect_originals:
        original = res.output.parent / "original"
        (original / "subs").mkdir(parents=True, exist_ok=True)
        base = res.output.name.removesuffix(".mkv").removesuffix(OUTPUT_SUFFIX)
        subs_out = original / "subs" / f"{base}.ass"
        if opts.sub_file is not None:
            shutil.copy2(opts.sub_file, subs_out.with_suffix(opts.sub_file.suffix))
        elif res.subs_file is not None:
            shutil.copy2(res.subs_file, subs_out.with_suffix(res.subs_file.suffix))
        else:
            try:
                ffmpeg.run(["ffmpeg", "-y", "-v", "error", "-i", res.output, "-map", "0:s:0", "-c", "copy",
                            subs_out], what="Saving subtitles")  # fmt: skip
            except ExternalToolError as e:
                log.warning("Couldn't save subtitles: %s", str(e).splitlines()[0])
        if source.resolve().parent != original.resolve():
            dest = original / (f"{base}{source.suffix}" if source_is_temp else source.name)
            shutil.move(str(source), dest)
            console.print(f"[dim]Original filed in {original}[/dim]")
    if opts.clean_cache:
        removed = 0
        for f in res.cache_files:
            try:
                f.unlink()
                removed += 1
            except FileNotFoundError:
                pass
        log.debug("Removed %d cache files for this episode", removed)


# --------------------------------------------------------------------------- core


def load_dialogue(
    video: Path, info: MediaInfo, opts: Options, work: Path, cache_key: str = "", console: Console | None = None
) -> tuple[subtitles.FilterResult, str, Path | None]:
    ocr_file = None
    if opts.sub_file is not None:
        subs = subtitles.load(opts.sub_file)
        source = f"file {opts.sub_file.name}"
    elif not info.subtitles and opts.sub_track is None and hardsubs.available():
        ocr_file = _burned_in_subtitles(video, info, cache_key, console or Console(stderr=True))
        if not ocr_file.stat().st_size:
            raise NoSubtitlesError(
                "This file has no subtitle tracks and no readable burned-in English subtitles. "
                "Supply English subtitles with --sub-file PATH (.srt or .ass)."
            )
        subs = subtitles.load(ocr_file)
        source = "burned-in subtitles (OCR)"
    else:
        stream = choose_subtitle_stream(info, opts.sub_track)
        subs = subtitles.load(subtitles.extract(video, stream, work))
        lang = f" [{stream.language}]" if stream.language else ""
        title = f" “{stream.title}”" if stream.title else ""
        source = f"stream #{stream.index} ({stream.codec}{lang}{title})"
    result = subtitles.filter_dialogue(
        subs, opts.include_styles, opts.exclude_styles,
        skip_songs=opts.skip_songs,
        extra_song_spans=subtitles.chapter_song_spans(info.chapters),
    )  # fmt: skip
    if not result.lines and ocr_file is not None:
        raise NoSubtitlesError(
            "This file has no subtitle tracks and no readable burned-in English subtitles. "
            "Supply English subtitles with --sub-file PATH (.srt or .ass)."
        )
    if not result.lines:
        raise NoSubtitlesError(
            f"No dialogue lines left after filtering subtitles from {source} "
            f"({len(result.skipped)} lines skipped). Check with --dry-run --verbose, and adjust "
            "--include-styles / --exclude-styles or pick another --sub-track."
        )
    return result, source, ocr_file


def _burned_in_subtitles(video: Path, info: MediaInfo, cache_key: str, console: Console) -> Path:
    """OCR the subtitles burned into the picture; cached per episode like the other caches."""
    key = hashlib.sha256(f"ocr-v2|{cache_key or video.resolve()}".encode()).hexdigest()[:24]
    out = cache_dir() / "ocr" / f"{key}.srt"
    if out.is_file():
        return out
    console.print("No subtitle track — reading the burned-in subtitles off the picture (OCR)")
    with _seconds_progress(console) as p:
        task = p.add_task("Reading subtitles", total=info.duration or None)
        hardsubs.extract(video, out, info.duration, lambda s: p.update(task, completed=s))
    return out


def _process(
    video: Path, label: str, out: Path, opts: Options, console: Console, work: Path, cache_key: str
) -> Result:
    info = probe(video)
    if info.encrypted:
        raise UsageError("This file is encrypted (DRM); that isn't supported.")
    filtered, source, ocr_file = load_dialogue(video, info, opts, work, cache_key, console)
    lines = filtered.lines

    backend = make_backend(opts.tts)
    voices = VoiceMap.load(opts.voice_map, opts.voice or backend.default_voice)
    if opts.voice:  # an explicit --voice beats the map file's `default`
        voices.default = opts.voice
    line_voices = [voices.voice_for(l.speaker) for l in lines]
    from .voicecheck import REGISTER_VOICES, voices_by_register

    # Without cloning, and with no voices chosen by the user, pick a male or female voice per line.
    by_register = (
        backend.name in REGISTER_VOICES and opts.voice is None and opts.voice_map is None
        and not voices.speakers
    )  # fmt: skip
    if by_register:
        line_voices = ["male/female by pitch"] * len(lines)

    console.print(
        f"Subtitles: {source} — [bold]{len(lines)}[/bold] spoken lines, "
        f"{len(filtered.skipped)} skipped"
    )
    if opts.dry_run:
        print_dry_run(console, filtered, line_voices, verbose=log.isEnabledFor(logging.DEBUG))
        return Result(label, None, skipped_reason="dry run", voiced=len(lines))

    res = Result(label)
    if ocr_file is not None:
        res.subs_file = ocr_file
        res.cache_files.append(ocr_file)

    # ---- background first: its vocals stem sets each line's loudness
    audio = choose_audio_stream(info)
    if opts.separate:
        bg, vocals, res.separation_cached = _separated_background(
            video, audio.index, cache_key, opts, console, work
        )
        res.cache_files += [bg, vocals]
    else:
        with console.status("Extracting original audio"):
            bg = separate.extract_audio(video, audio.index, work / "original.wav", timing.RATE)
        vocals = bg  # without separation the full mix is the best level reference we have
    targets = levels.reference_levels(vocals, [(l.start, l.end) for l in lines])
    if by_register:
        male, female = REGISTER_VOICES[backend.name]
        with console.status("Matching voices to male/female lines"):
            line_voices, n_male, n_female = voices_by_register(lines, vocals, male, female)
        console.print(f"Voices: {n_male} male ({male}), {n_female} female ({female}), "
                      f"{len(lines) - n_male - n_female} follow the previous line")  # fmt: skip

    # ---- TTS
    refs: list[Path | None] = [None] * len(lines)
    if getattr(backend, "needs_reference", False):
        if not opts.separate:
            log.warning("Voice cloning without vocal separation: references include music/SFX.")
        from .tts.clone import MIN_REF_S

        refs = list(levels.write_references(vocals, [(l.start, l.end) for l in lines],
                                            work / "refs", MIN_REF_S))  # fmt: skip
    cache = TTSCache(cache_dir() / "tts")
    try:
        with _progress(console) as p:
            task = p.add_task(f"Synthesizing speech ({backend.name})", total=len(lines))
            clip_paths, stats = asyncio.run(
                synthesize_all(
                    backend,
                    [(l.text, v, r) for l, v, r in zip(lines, line_voices, refs)],
                    cache,
                    concurrency=getattr(backend, "concurrency", opts.concurrency),
                    on_done=lambda: p.advance(task),
                )
            )
        if getattr(backend, "needs_reference", False):
            from .voicecheck import fix_mismatches

            with _progress(console) as p:
                task = p.add_task("Checking voices (male/female)", total=len(lines))
                found, fixed = fix_mismatches(backend, lines, line_voices, refs, vocals, cache,
                                              on_done=lambda: p.advance(task))  # fmt: skip
            res.voice_mismatches, res.voice_fixed = found, fixed
    finally:
        if hasattr(backend, "close"):
            backend.close()
    res.tts_cached, res.tts_new = stats.cached, stats.synthesized
    marks = cache.dir.parent / "voicecheck"
    res.cache_files += clip_paths + [marks / c.stem[:2] / c.stem for c in clip_paths]

    # ---- timing + level matching
    ends = timing.slot_ends(lines)
    gain = 10 ** (opts.speech_gain_db / 20)
    with _progress(console) as p, ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        task = p.add_task("Fitting clips to timing", total=len(lines))

        def fit(i: int) -> timing.PlacedClip:
            c = timing.fit_clip(lines[i], ends[i], str(clip_paths[i]), opts.max_speedup,
                                target_rms=targets[i] * gain)  # fmt: skip
            p.advance(task)
            return c

        clips = list(pool.map(fit, range(len(lines))))
    res.voiced = len(clips)
    res.sped_up = sum(c.tempo > 1.0 for c in clips)
    for c in clips:
        if c.overflow_s:
            res.overflow_times.append(c.line.start)
            log.warning(
                "%s overflows its slot by %.2fs at %.2fx: %s",
                subtitles.fmt_time(c.line.start), c.overflow_s, c.tempo, c.line.text[:70],
            )  # fmt: skip
    res.overflowing = len(res.overflow_times)

    n = timing.total_samples(info.duration, clips)

    if opts.speech_only:
        mix.mix(None, clips, n, work / "speech.wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(work / "speech.wav"), out)
        res.output = out
        return res

    duck = restore = None
    if opts.separate:
        # Remove the Japanese voice only where an English line (or its subtitle span) is.
        # Speech usually starts a beat before its subtitle appears, so start removing it early.
        pre = int(JP_PREROLL_S * timing.RATE)
        spans = [(max(0, int(c.line.start * timing.RATE / 1000) - pre), int(c.line.end * timing.RATE / 1000))
                 for c in clips] + [(c.start_sample, c.end_sample) for c in clips]  # fmt: skip
        restore = (vocals, separate.speech_intervals(spans, timing.RATE, merge_gap_s=0.15))
    else:
        spans = [(c.start_sample, c.end_sample) for c in clips]
        duck = (separate.speech_intervals(spans, timing.RATE), opts.duck_db)

    # ---- mix, encode, mux
    with _seconds_progress(console) as p:
        task = p.add_task("Mixing", total=n / timing.RATE)
        mixed = mix.mix(bg, clips, n, work / "mix.wav", duck=duck, restore=restore,
                        on_progress=lambda s: p.update(task, completed=s))  # fmt: skip
    with console.status(f"Normalizing loudness and encoding ({opts.codec})"):
        original_lufs = mix.measure_lufs(video, f"0:{audio.index}")
        dub = mix.encode(mixed, work / "dub.mka", opts.codec, original_lufs)
    with console.status("Muxing"):
        mux.mux(video, info, dub, out, set_default=opts.set_default)
    res.output = out
    return res


def _separated_background(
    video: Path, stream_index: int, cache_key: str, opts: Options, console: Console, work: Path
) -> tuple[Path, Path, bool]:
    """(no_vocals, vocals) stems at 48 kHz, cached per input file."""
    key = hashlib.sha256(f"{cache_key}|a{stream_index}|{separate.MODEL_NAME}".encode()).hexdigest()
    folder = cache_dir() / "separation"
    cached_bg, cached_vo = folder / f"{key}.flac", folder / f"{key}.vocals.flac"
    if cached_bg.is_file() and cached_vo.is_file():
        console.print("Background: using cached vocal separation")
        return cached_bg, cached_vo, True

    with console.status("Extracting original audio"):
        src = separate.extract_audio(video, stream_index, work / "original_44k.wav", 44100)
    with _seconds_progress(console) as p:
        task = p.add_task("Separating vocals (Demucs)", total=None)

        def prog(done: float, total: float) -> None:
            p.update(task, completed=done, total=total)

        stems = separate.separate_stems(src, work / "no_vocals_44k.wav", work / "vocals_44k.wav",
                                        device=opts.device, on_progress=prog)  # fmt: skip
    folder.mkdir(parents=True, exist_ok=True)
    with console.status("Caching separated stems"):
        for stem, dest in zip(stems, (cached_bg, cached_vo)):
            partial = dest.with_name(f"{dest.stem}.partial{os.getpid()}.flac")
            ffmpeg.run(
                ["ffmpeg", "-y", "-v", "error", "-i", stem, "-ar", str(timing.RATE),
                 "-c:a", "flac", "-sample_fmt", "s32", partial],
                what="Resampling separated audio",
            )  # fmt: skip
            os.replace(partial, dest)
    return cached_bg, cached_vo, False


# --------------------------------------------------------------------------- reporting


def print_dry_run(
    console: Console,
    filtered: subtitles.FilterResult,
    voices: list[str],
    *,
    verbose: bool = False,
) -> None:
    t = Table(title="Lines to be spoken", show_lines=False)
    t.add_column("#", justify="right", style="dim")
    t.add_column("Start")
    t.add_column("End")
    t.add_column("Style", style="cyan")
    t.add_column("Speaker", style="magenta")
    t.add_column("Voice", style="green")
    t.add_column("Text")
    for i, (l, v) in enumerate(zip(filtered.lines, voices), 1):
        t.add_row(str(i), subtitles.fmt_time(l.start), subtitles.fmt_time(l.end),
                  l.style, l.speaker, v, l.text)  # fmt: skip
    console.print(t)

    counts = filtered.reason_counts()
    if counts:
        styles: dict[str, set[str]] = {}
        for s in filtered.skipped:
            styles.setdefault(s.reason, set()).add(s.style)
        st = Table(title="Skipped lines")
        st.add_column("Reason")
        st.add_column("Count", justify="right")
        st.add_column("Styles")
        for reason, n in counts.most_common():
            st.add_row(reason, str(n), ", ".join(sorted(styles[reason]))[:80])
        console.print(st)
    if verbose and filtered.skipped:
        sk = Table(title="Skipped lines (detail)")
        for col in ("Start", "Style", "Reason", "Text"):
            sk.add_column(col)
        for s in filtered.skipped:
            sk.add_row(subtitles.fmt_time(s.start), s.style, s.reason, s.text[:80])
        console.print(sk)
    elif filtered.skipped:
        console.print("[dim]Add --verbose to list every skipped line.[/dim]")
