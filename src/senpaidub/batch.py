"""Folder input: process every episode in natural order, never stopping on one failure."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from rich.console import Console

from .errors import SenpaiDubError
from .pipeline import OUTPUT_SUFFIX, Options, Result, process_file

VIDEO_EXTS = {".mkv", ".mp4"}


def natural_key(p: Path) -> list[object]:
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", str(p))]


def find_videos(folder: Path) -> list[Path]:
    found = [
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTS
        and not p.stem.endswith(OUTPUT_SUFFIX)  # our own outputs
        and not p.name.startswith(".")
    ]
    return sorted(found, key=natural_key)


def process_folder(folder: Path, opts: Options, console: Console) -> tuple[list[Result], list[tuple[Path, str, int]]]:
    videos = find_videos(folder)
    if not videos:
        console.print(f"[yellow]No .mkv/.mp4 files found in {folder}[/yellow]")
    results: list[Result] = []
    failures: list[tuple[Path, str, int]] = []
    for i, video in enumerate(videos, 1):
        console.rule(f"[{i}/{len(videos)}] {video.relative_to(folder)}")
        file_opts = opts
        if opts.output is not None:
            # Mirror the input folder structure inside the output folder.
            out_dir = opts.output / video.parent.relative_to(folder)
            out_dir.mkdir(parents=True, exist_ok=True)
            file_opts = replace(opts, output=out_dir)
        try:
            res = process_file(video, file_opts, console)
            results.append(res)
            report(console, res)
        except SenpaiDubError as e:
            failures.append((video, str(e).splitlines()[0], e.exit_code))
            console.print(f"[red]Failed:[/red] {e}")
        except Exception as e:  # noqa: BLE001 — keep the batch going
            failures.append((video, f"{type(e).__name__}: {e}", 1))
            console.print_exception()
    return results, failures


def report(console: Console, res: Result) -> None:
    if res.skipped_reason == "dry run":
        return
    if res.skipped_reason:
        console.print(f"[yellow]Skipped[/yellow] {res.input}: {res.skipped_reason}")
        return
    sep = ""
    if res.separation_cached is not None:
        sep = ", separation cached" if res.separation_cached else ", vocals separated"
    console.print(
        f"[green]✓[/green] {res.output}\n"
        f"  {res.voiced} lines voiced, {res.sped_up} sped up, {res.overflowing} overflowing; "
        f"TTS: {res.tts_new} new, {res.tts_cached} cached{sep}"
        + (f"\n  voice check: {res.voice_fixed} of {res.voice_mismatches} male/female mix-ups fixed"
           if res.voice_mismatches else "")
    )
