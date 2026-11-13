"""Error types. Each carries the process exit code the CLI should use."""

from __future__ import annotations


class SenpaiDubError(Exception):
    exit_code = 1


class UsageError(SenpaiDubError):
    """Bad arguments or configuration."""

    exit_code = 1


class NoSubtitlesError(SenpaiDubError):
    """No usable (text) subtitles were found."""

    exit_code = 2


class ExternalToolError(SenpaiDubError):
    """ffmpeg/ffprobe/TTS/Demucs failed."""

    exit_code = 3
