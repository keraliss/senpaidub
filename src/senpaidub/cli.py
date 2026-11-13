"""Command-line interface."""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from . import __version__
from .batch import process_folder, report
from .config import config_path, load_config, resolve
from .download import is_url, probe_url, sanitize_filename
from .errors import SenpaiDubError, UsageError
from .ffmpeg import check_tools
from .mix import CODECS
from .pipeline import Options, process_file, process_url
from .probe import MediaInfo, probe

app = typer.Typer(add_completion=False, context_settings={"help_option_names": ["-h", "--help"]})
console = Console(stderr=True)
log = logging.getLogger("senpaidub")


def _version(value: bool) -> None:
    if value:
        print(f"senpaidub {__version__}")
        raise typer.Exit()


@app.command()
def dub(
    input: Optional[str] = typer.Argument(None, help="Video file, folder (recursive .mkv/.mp4), or http(s) stream URL. Not needed with --stremio."),
    stremio: Optional[str] = typer.Option(None, "--stremio", metavar="SERIES", help="Dub every episode of SERIES that Stremio has open (matched against file names), then keep watching for more."),
    series_name: Optional[str] = typer.Option(None, "--series-name", help="File episodes as a series: output goes to <library>/<series-name>/'<series-name> - 01 [EN dub].mkv', with the originals and subtitles moved into its 'original' folder."),
    keep_cache: bool = typer.Option(False, "--keep-cache", help="Keep vocal-separation and voice caches after each episode (faster re-runs, uses disk)."),
    once: bool = typer.Option(False, "--once", help="With --stremio: dub what's loaded now and exit instead of watching."),
    list_tracks: bool = typer.Option(False, "--list-tracks", help="Print audio and subtitle streams and exit."),
    sub_track: Optional[int] = typer.Option(None, "--sub-track", metavar="N", help="Subtitle stream index to voice (see --list-tracks)."),
    sub_file: Optional[Path] = typer.Option(None, "--sub-file", help="External .srt/.ass instead of an embedded track."),
    tts: Optional[str] = typer.Option(None, "--tts", metavar="[turbo|edge|google]", help="TTS backend: turbo = voice cloning from the original audio (default), edge = fast generic voices."),
    voice: Optional[str] = typer.Option(None, "--voice", help="Voice for edge/google (default: en-US-AriaNeural for edge). Ignored by turbo."),
    voice_map: Optional[Path] = typer.Option(None, "--voice-map", help="YAML mapping speaker names to voices."),
    no_separate: bool = typer.Option(False, "--no-separate", help="Skip vocal removal; duck the original audio under speech."),
    duck_db: Optional[float] = typer.Option(None, "--duck-db", help="Ducking amount in dB with --no-separate (default: -12)."),
    max_speedup: Optional[float] = typer.Option(None, "--max-speedup", help="Max speed-up to fit a line in its slot (default: 1.4)."),
    include_styles: Optional[str] = typer.Option(None, "--include-styles", metavar="RE", help="Only voice ASS styles matching this regex (replaces default filtering)."),
    exclude_styles: Optional[str] = typer.Option(None, "--exclude-styles", metavar="RE", help="Also skip ASS styles matching this regex."),
    set_default: bool = typer.Option(False, "--set-default", help="Make the new track the default audio track."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file or folder."),
    name: Optional[str] = typer.Option(None, "--name", help="Output file name for URL inputs."),
    codec: Optional[str] = typer.Option(None, "--codec", metavar="[aac|opus]", help="Dub track codec (default: aac 192k)."),
    device: Optional[str] = typer.Option(None, "--device", metavar="[auto|cpu|cuda|mps]", help="Device for Demucs (default: auto)."),
    concurrency: Optional[int] = typer.Option(None, "--concurrency", help="Parallel TTS requests (default: 4)."),
    voice_songs: bool = typer.Option(False, "--voice-songs", help="Also voice opening/ending lyrics (skipped by default)."),
    speech_gain: Optional[float] = typer.Option(None, "--speech-gain", help="dB to add to speech relative to the original dialogue level (default: 0)."),
    speech_only: bool = typer.Option(False, "--speech-only", help="Write only the aligned speech as a WAV (no mixing/muxing)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the lines that would be spoken and exit."),
    keep_temp: bool = typer.Option(False, "--keep-temp", help="Keep intermediate files."),
    force: bool = typer.Option(False, "--force", help="Reprocess even if the output exists."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log ffmpeg commands and extra detail."),
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Show version."),
) -> None:
    """Add an English TTS audio track to an anime episode, voiced from its English subtitles."""
    handler = RichHandler(console=console, show_path=False, show_time=False)
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.propagate = False
    logging.basicConfig(level=logging.WARNING, handlers=[handler])  # third-party: warnings only

    try:
        check_tools()
        cfg = resolve(
            {
                "tts": tts, "voice": voice, "voice_map": voice_map,
                "separate": False if no_separate else None, "duck_db": duck_db,
                "max_speedup": max_speedup, "include_styles": include_styles,
                "exclude_styles": exclude_styles, "set_default": True if set_default else None,
                "codec": codec, "device": device, "concurrency": concurrency,
                "speech_gain": speech_gain, "keep_cache": True if keep_cache else None,
            },
            load_config(),
        )  # fmt: skip
        log.debug("Config file: %s", config_path())
        _validate(cfg)
        if sub_file is not None and not sub_file.is_file():
            raise UsageError(f"--sub-file not found: {sub_file}")
        opts = Options(
            sub_track=sub_track,
            sub_file=sub_file,
            tts=cfg["tts"],
            voice=cfg["voice"],
            voice_map=Path(cfg["voice_map"]).expanduser() if cfg["voice_map"] else None,
            separate=cfg["separate"],
            duck_db=float(cfg["duck_db"]),
            max_speedup=float(cfg["max_speedup"]),
            include_styles=cfg["include_styles"] or None,
            exclude_styles=cfg["exclude_styles"] or None,
            set_default=cfg["set_default"],
            output=output,
            name=name,
            dry_run=dry_run,
            keep_temp=keep_temp,
            force=force,
            codec=cfg["codec"],
            device=cfg["device"],
            concurrency=int(cfg["concurrency"]),
            speech_only=speech_only,
            speech_gain_db=float(cfg["speech_gain"]),
            skip_songs=not voice_songs,
            series_name=series_name,
            collect_originals=bool(series_name or stremio),
            clean_cache=not cfg["keep_cache"],
        )
        library = Path(cfg["library"]).expanduser()
        if (series_name or stremio) and opts.output is None:
            opts = replace(opts, output=library / sanitize_filename(series_name or stremio))
            opts.output.mkdir(parents=True, exist_ok=True)

        if stremio is not None:
            from .stremio import run_series

            if input is not None:
                raise UsageError("Pass either INPUT or --stremio, not both.")
            if opts.output.suffix:
                raise UsageError("With --stremio, --output must be a folder.")
            try:
                results, failures = run_series(stremio, opts, console, series=series_name, watch=not once)
            except KeyboardInterrupt:
                console.print("[yellow]Stopped watching.[/yellow]")
                return
            _batch_summary(results, failures)
            if failures:
                sys.exit(max(code for _, _, code in failures))
            return
        if input is None:
            raise UsageError("Missing INPUT (a file, folder or stream URL), or use --stremio SERIES.")

        if is_url(input):
            if list_tracks:
                _print_tracks(probe_url(input))
                return
            res = process_url(input, opts, console)
            report(console, res)
            return

        path = Path(input).expanduser()
        if path.is_dir():
            if list_tracks:
                raise UsageError("--list-tracks needs a single file, not a folder.")
            if opts.output is not None and opts.output.suffix:
                raise UsageError("With a folder input, --output must be a folder.")
            results, failures = process_folder(path, opts, console)
            _batch_summary(results, failures)
            if failures:
                sys.exit(max(code for _, _, code in failures))
            return
        if not path.exists():
            raise UsageError(f"Input not found: {input}")
        if list_tracks:
            _print_tracks(probe(path))
            return
        report(console, process_file(path, opts, console))
    except SenpaiDubError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(e.exit_code)
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow]")
        sys.exit(130)


def _validate(cfg: dict) -> None:
    if cfg["tts"] not in ("turbo", "edge", "google", "mock"):
        raise UsageError(f"--tts must be turbo, edge or google (got {cfg['tts']!r}).")
    if cfg["codec"] not in CODECS:
        raise UsageError(f"--codec must be one of {', '.join(CODECS)} (got {cfg['codec']!r}).")
    if cfg["device"] not in ("auto", "cpu", "cuda", "mps"):
        raise UsageError(f"--device must be auto, cpu, cuda or mps (got {cfg['device']!r}).")
    if float(cfg["max_speedup"]) < 1.0:
        raise UsageError("--max-speedup must be at least 1.0.")
    if int(cfg["concurrency"]) < 1:
        raise UsageError("--concurrency must be at least 1.")


def _print_tracks(info: MediaInfo) -> None:
    t = Table(title=f"Streams in {info.path}")
    for col, kw in (("#", {"justify": "right"}), ("Type", {}), ("Codec", {}), ("Kind", {}),
                    ("Language", {}), ("Title", {}), ("Default", {"justify": "center"})):  # fmt: skip
        t.add_column(col, **kw)
    for s in info.streams:
        if s.type not in ("audio", "subtitle"):
            continue
        kind = ("text" if s.is_text_sub else "image") if s.type == "subtitle" else f"{s.channels or '?'}ch"
        t.add_row(str(s.index), s.type, s.codec, kind, s.language or "—", s.title or "",
                  "✓" if s.default else "")  # fmt: skip
    Console().print(t)
    if not info.subtitles:
        console.print("[yellow]No subtitle streams — burned-in subtitles will be read off the picture "
                      "(macOS), or pass --sub-file.[/yellow]")


def _batch_summary(results: list, failures: list) -> None:
    done = [r for r in results if r.output and not r.skipped_reason]
    skipped = [r for r in results if r.skipped_reason and r.skipped_reason != "dry run"]
    console.rule("Batch summary")
    console.print(f"{len(done)} dubbed, {len(skipped)} skipped, {len(failures)} failed")
    for path, msg, _code in failures:
        console.print(f"  [red]✗[/red] {path}: {msg}")


def _is(exc: BaseException, name: str) -> bool:
    # Typer ≥0.20 vendors its own click, so match click exception types by name.
    return any(c.__name__ == name for c in type(exc).__mro__)


def main() -> None:
    try:
        rv = app(standalone_mode=False)
    except Exception as e:
        if _is(e, "UsageError") and hasattr(e, "show"):
            e.show()
            sys.exit(1)
        if _is(e, "Abort"):
            console.print("[yellow]Interrupted.[/yellow]")
            sys.exit(130)
        raise
    sys.exit(rv if isinstance(rv, int) else 0)


if __name__ == "__main__":
    main()
