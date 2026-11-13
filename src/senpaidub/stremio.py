"""Series mode for Stremio: dub every episode of a series that Stremio has opened.

Stremio's local streaming server lists the torrents it currently has loaded. We match their
video files against the series name, dub them in episode order, and (in watch mode) keep
picking up episodes as you open them in Stremio. Finding episodes is left to Stremio.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

from rich.console import Console

from .batch import natural_key, report
from .download import download, probe_url, sanitize_filename
from .errors import SenpaiDubError, ExternalToolError, NoSubtitlesError
from .pipeline import Options, Result, default_output_name, process_file

log = logging.getLogger("senpaidub")

SERVER = "http://127.0.0.1:11470"
VIDEO_EXTS = (".mkv", ".mp4", ".m4v", ".webm", ".avi")
POLL_S = 15


@dataclass(frozen=True)
class StremioFile:
    info_hash: str
    index: int
    name: str

    @property
    def url(self) -> str:
        return f"{SERVER}/{self.info_hash}/{self.index}"


def list_files(server: str = SERVER) -> list[StremioFile]:
    try:
        with urllib.request.urlopen(f"{server}/stats.json", timeout=10) as r:
            data = json.load(r)
    except OSError as e:
        raise ExternalToolError(
            f"Can't reach Stremio's streaming server at {server}. Is Stremio running?\n{e}"
        ) from e
    items = data.items() if isinstance(data, dict) else enumerate(data)
    out = []
    for key, t in items:
        if not isinstance(t, dict):
            continue
        h = t.get("infoHash") or str(key)
        for i, f in enumerate(t.get("files") or []):
            name = Path(str(f.get("path") or f.get("name") or "")).name
            if name.lower().endswith(VIDEO_EXTS):
                out.append(StremioFile(h, i, name))
    return out


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.casefold())


def matches(name: str, pattern: str) -> bool:
    """Every word of the pattern appears in the file name, in any order."""
    words = set(_words(name))
    return all(w in words for w in _words(pattern))


def episode_number(name: str) -> tuple[int | None, int | None]:
    """(season, episode) from common release names; either may be None."""
    stem = re.sub(r"\.[a-z0-9]{2,4}$", "", name, flags=re.IGNORECASE)
    stem = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", stem)  # [Group], (1080p), [CRC]
    if m := re.search(r"\bS(\d{1,2})\s*E(\d{1,4})\b", stem, re.IGNORECASE):
        return int(m[1]), int(m[2])
    if m := re.search(r"\s-\s(\d{1,4})(?:v\d)?\b", stem):
        return None, int(m[1])
    if m := re.search(r"\b(?:E|EP|Episode)\s*(\d{1,4})\b", stem, re.IGNORECASE):
        return None, int(m[1])
    return None, None


def output_name(file_name: str, series: str | None) -> str:
    season, ep = episode_number(file_name)
    if series and ep is not None:
        tag = f"S{season:02d}E{ep:02d}" if season else f"{ep:02d}"
        return sanitize_filename(f"{series} - {tag}")
    stem = re.sub(r"\.[a-z0-9]{2,4}$", "", file_name, flags=re.IGNORECASE)
    stem = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", stem)
    return sanitize_filename(stem)


def sort_key(f: StremioFile) -> tuple:
    season, ep = episode_number(f.name)
    return (season or 0, ep if ep is not None else 10**6, natural_key(Path(f.name)))


def run_series(
    pattern: str,
    opts: Options,
    console: Console,
    *,
    series: str | None = None,
    watch: bool = True,
) -> tuple[list[Result], list[tuple[str, str, int]]]:
    """Dub every matching episode Stremio has loaded; with watch=True keep waiting for more.

    Downloading and dubbing run separately: each episode is downloaded as soon as it appears
    (Stremio unloads torrents a while after you stop playing them), and downloaded episodes
    are dubbed one by one in episode order. Downloads land in <output>/.incoming, so an
    interrupted session resumes where it left off.
    """
    assert opts.output is not None
    out_dir = opts.output
    incoming = out_dir / ".incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []
    failures: list[tuple[str, str, int]] = []
    stop = threading.Event()
    seen: set[tuple[str, int]] = set()
    scan_done = threading.Event()  # set after the first full scan + downloads (for --once)

    def target(name: str) -> Path:
        return out_dir / default_output_name(name)

    def downloader() -> None:
        while not stop.is_set():
            try:
                files = sorted((f for f in list_files() if matches(f.name, pattern)), key=sort_key)
            except SenpaiDubError as e:
                console.print(f"[red]{e}[/red]")
                files = []
            for f in files:
                if stop.is_set() or (f.info_hash, f.index) in seen:
                    continue
                seen.add((f.info_hash, f.index))
                name = output_name(f.name, series)
                local = incoming / f"{name}.mkv"
                if (target(name).exists() and not opts.force) or local.exists():
                    continue
                try:
                    info = probe_url(f.url)
                    if not info.subtitles and opts.sub_file is None:
                        raise NoSubtitlesError("no embedded subtitles (series mode needs them)")
                    console.print(f"[cyan]↓ Downloading[/cyan] {name} from Stremio")
                    part = local.with_suffix(".part")
                    download(f.url, info, part)
                    os.replace(part, local)
                    console.print(f"[cyan]✓ Downloaded[/cyan] {name} ({local.stat().st_size / 1e6:.0f} MB)")
                except SenpaiDubError as e:
                    failures.append((f.name, str(e).splitlines()[0], e.exit_code))
                    console.print(f"[red]Couldn't download {f.name}:[/red] {str(e).splitlines()[0]}")
            scan_done.set()
            if not watch:
                return
            stop.wait(POLL_S)

    dl = threading.Thread(target=downloader, daemon=True, name="stremio-downloader")
    dl.start()
    waiting_shown = False
    try:
        while True:
            pending = sorted(
                (p for p in incoming.glob("*.mkv") if not p.name.startswith(".")),
                key=lambda p: sort_key(StremioFile("", 0, p.name)),
            )
            if not pending:
                if not watch and scan_done.is_set() and not dl.is_alive():
                    break
                if watch and not waiting_shown and scan_done.is_set():
                    console.print(
                        f"[bold]Watching Stremio for “{pattern}” episodes[/bold] — open them in "
                        f"Stremio and they'll be dubbed into {out_dir}. Press Ctrl+C to stop."
                    )
                    waiting_shown = True
                time.sleep(2)
                continue
            local = pending[0]
            console.rule(local.stem)
            try:
                res = process_file(local, replace(opts, output=out_dir), console)
                results.append(res)
                report(console, res)
                if not opts.keep_temp:
                    local.unlink(missing_ok=True)
            except SenpaiDubError as e:
                failures.append((local.name, str(e).splitlines()[0], e.exit_code))
                console.print(f"[red]Failed:[/red] {e}")
                local.rename(local.with_suffix(".failed"))
            except Exception as e:  # noqa: BLE001 — keep the series going
                failures.append((local.name, f"{type(e).__name__}: {e}", 1))
                console.print_exception()
                local.rename(local.with_suffix(".failed"))
            waiting_shown = False
    finally:
        stop.set()
        try:
            incoming.rmdir()  # only if empty
        except OSError:
            pass
    if not seen:
        console.print(
            f"[yellow]Stremio has no episodes matching “{pattern}” loaded.[/yellow] "
            "Open them in Stremio (start playing, then you can go back) and run again."
        )
    return results, failures
