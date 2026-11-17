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
from .config import config_path, load_config, resolve, validate_config
from .download import is_url, probe_url, sanitize_filename
from .errors import SenpaiDubError, UsageError
from .ffmpeg import check_tools
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
    tts: Optional[str] = typer.Option(None, "--tts", metavar="[qwen|edge|google]", help="TTS backend: qwen = local voice cloning from the original audio, on the GPU (default); edge = fast generic online voices."),
    voice: Optional[str] = typer.Option(None, "--voice", help="Voice for edge/google (default: en-US-AriaNeural for edge). Ignored by qwen."),
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
    fast: bool = typer.Option(False, "--fast", help="qwen at full speed (~2x faster) but the computer lags meanwhile. Default: gentle, which keeps it usable."),
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
                "codec": codec, "device": device, "concurrency": concurrency, "qwen_mode": "fast" if fast else None,
                "speech_gain": speech_gain, "keep_cache": True if keep_cache else None,
            },
            load_config(),
        )  # fmt: skip
        log.debug("Config file: %s", config_path())
        validate_config(cfg)
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
            qwen_mode=cfg["qwen_mode"],
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


# `serve` and `setup` live in their own Typer apps: `dub` takes an optional INPUT argument, so
# subcommands on `app` would break `senpaidub ep.mkv`. main() dispatches on the first word.
serve_app = typer.Typer(add_completion=False, context_settings={"help_option_names": ["-h", "--help"]})
setup_app = typer.Typer(add_completion=False, context_settings={"help_option_names": ["-h", "--help"]})
SUBCOMMANDS = {"serve": serve_app, "setup": setup_app}


def _free_port(start: int, tries: int = 50):
    import socket

    for port in range(start, start + tries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            sock.close()
            continue
        return sock, port
    raise UsageError(f"No free port between {start} and {start + tries - 1}; pass --port.")


@serve_app.command()
def serve(
    port: int = typer.Option(7860, "--port", help="Port to listen on (the next free one is used if it's taken)."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open the browser."),
    library: Optional[Path] = typer.Option(None, "--library", help="Library folder for series (default: from config.toml)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log every HTTP request."),
) -> None:
    """Start the local web UI (http://127.0.0.1:7860). Only this computer can reach it."""
    import secrets
    import signal
    import threading
    import webbrowser

    try:
        import uvicorn

        from .server.app import create_app
        from .server.jobs import JobManager
    except ImportError as e:  # pragma: no cover
        console.print(f"[bold red]Error:[/bold red] the web UI needs fastapi and uvicorn ({e}). "
                      "Reinstall senpaidub: pip install -e .")  # fmt: skip
        sys.exit(1)
    try:
        check_tools()
        cfg = resolve({}, load_config())
        lib = (library or Path(cfg["library"])).expanduser().resolve()
        sock, port = _free_port(port)
    except SenpaiDubError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(e.exit_code)
    token = secrets.token_urlsafe(24)
    manager = JobManager(library=lib if library else None)
    manager.start()
    url = f"http://127.0.0.1:{port}/?token={token}"
    console.print("[bold]Senpai Dub[/bold] is running. Open this link:")
    console.print(url, markup=False, highlight=False, soft_wrap=True)
    console.print("It includes your private key, so don't share it. Press Ctrl+C to stop.")
    if not no_browser:
        timer = threading.Timer(0.8, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    config = uvicorn.Config(create_app(manager, token, lib, port=port), host="127.0.0.1", port=port,
                            log_level="info" if verbose else "warning", access_log=verbose)  # fmt: skip
    server = uvicorn.Server(config)

    def _terminate(_sig, _frame):  # uvicorn re-raises SIGTERM after a graceful exit; stop the workers too
        raise KeyboardInterrupt

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _terminate)
    try:
        server.run(sockets=[sock])
    except KeyboardInterrupt:
        pass
    finally:
        manager.close()
        console.print("[yellow]Stopped.[/yellow] Unfinished dubs will continue next time you run senpaidub serve.")


@setup_app.command()
def setup(
    what: str = typer.Argument(..., metavar="qwen", help="What to set up. Only 'qwen' (local voice cloning) so far."),
    no_model: bool = typer.Option(False, "--no-model", help="Skip downloading the ~4.5 GB voice model now."),
) -> None:
    """Install optional parts. `senpaidub setup qwen` installs local voice cloning (a few GB)."""
    from .setup_tts import SetupError, install

    if what != "qwen":
        console.print(f"[bold red]Error:[/bold red] unknown setup target {what!r}; try `senpaidub setup qwen`.")
        sys.exit(1)
    try:
        python = install(on_line=lambda line: console.print(line, markup=False, highlight=False),
                         download_model=not no_model)  # fmt: skip
    except SetupError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(e.exit_code)
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    console.print(f"[green]✓[/green] Qwen voices installed ({python}).")


def _pick_app(argv: list[str]):
    """The Typer app and args for argv (without the program name)."""
    if argv and argv[0] in SUBCOMMANDS and not Path(argv[0]).exists():
        return SUBCOMMANDS[argv[0]], argv[1:], f"senpaidub {argv[0]}"
    return app, argv, "senpaidub"


def main() -> None:
    target, args, prog = _pick_app(sys.argv[1:])
    try:
        rv = target(args=args, prog_name=prog, standalone_mode=False)
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
