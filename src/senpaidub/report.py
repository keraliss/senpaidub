"""Progress reporting: one interface for the terminal (rich) and the web UI (events).

The pipeline never talks to a console directly; it opens *stages* and sends messages through a
`Reporter`. `RichReporter` reproduces the CLI's bars, spinners and prints; `NullReporter` is silent;
the web worker uses its own subclass that turns events into queue messages.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Protocol

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

GROUPS = ["subtitles", "separate", "voice", "mix", "mux"]

STAGE_GROUPS: dict[str, str] = {
    "download": "subtitles",
    "subtitles": "subtitles",
    "ocr": "subtitles",
    "extract": "separate",
    "separate": "separate",
    "cache_stems": "separate",
    "voices": "voice",
    "tts": "voice",
    "voicecheck": "voice",
    "fit": "mix",
    "mix": "mix",
    "encode": "mix",
    "mux": "mux",
}

UNITS = ("items", "seconds", "none")


class Stage(Protocol):
    def advance(self, n: float = 1) -> None: ...
    def update(self, completed: float | None = None, total: float | None = None) -> None: ...


class _NullStage:
    def advance(self, n: float = 1) -> None:
        pass

    def update(self, completed: float | None = None, total: float | None = None) -> None:
        pass


class Reporter:
    """Base reporter: does nothing. Subclasses override what they need; all must be thread-safe."""

    @contextmanager
    def _null_stage(self) -> Iterator[Stage]:
        yield _NullStage()

    def stage(
        self, key: str, label: str, total: float | None = None, unit: str = "items"
    ) -> AbstractContextManager[Stage]:
        return self._null_stage()

    def info(self, msg: str, style: str | None = None) -> None:
        pass

    def warn(self, msg: str) -> None:
        pass


class NullReporter(Reporter):
    """Explicitly silent reporter (same as the base class)."""


def items_progress(console: Console) -> Progress:
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


def seconds_progress(console: Console) -> Progress:
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


class _RichStage:
    def __init__(self, progress: Progress, task: TaskID) -> None:
        self._p, self._task = progress, task

    def advance(self, n: float = 1) -> None:
        self._p.advance(self._task, n)

    def update(self, completed: float | None = None, total: float | None = None) -> None:
        kw: dict[str, float] = {}
        if completed is not None:
            kw["completed"] = completed
        if total is not None:
            kw["total"] = total
        if kw:
            self._p.update(self._task, **kw)  # type: ignore[arg-type]


class RichReporter(Reporter):
    """Today's terminal output: transient progress bars, status spinners and plain prints."""

    # Stages that had no visible indicator in the CLI before (and wrap other stages' output).
    SILENT = frozenset({"subtitles"})

    def __init__(self, console: Console) -> None:
        self.console = console

    @contextmanager
    def _stage(self, key: str, label: str, total: float | None, unit: str) -> Iterator[Stage]:
        if key in self.SILENT:
            yield _NullStage()
        elif unit == "none":
            with self.console.status(label):
                yield _NullStage()
        else:
            make = seconds_progress if unit == "seconds" else items_progress
            with make(self.console) as p:
                yield _RichStage(p, p.add_task(label, total=total))

    def stage(
        self, key: str, label: str, total: float | None = None, unit: str = "items"
    ) -> AbstractContextManager[Stage]:
        return self._stage(key, label, total, unit)

    def info(self, msg: str, style: str | None = None) -> None:
        self.console.print(msg, style=style, markup=False)

    def warn(self, msg: str) -> None:
        # Warnings are also logged, and the logging handler already prints them.
        pass
