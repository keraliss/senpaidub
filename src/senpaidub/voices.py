"""Speaker → voice mapping (from the ASS Name/Actor field and an optional YAML file)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .errors import UsageError


@dataclass
class VoiceMap:
    default: str
    speakers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._lookup = {k.strip().casefold(): v for k, v in self.speakers.items()}

    def voice_for(self, speaker: str) -> str:
        return self._lookup.get(speaker.strip().casefold(), self.default) if speaker else self.default

    @classmethod
    def load(cls, path: Path | None, default: str) -> VoiceMap:
        """Load a YAML voice map. Its `default` applies unless --voice was given explicitly."""
        if path is None:
            return cls(default)
        try:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except FileNotFoundError as e:
            raise UsageError(f"Voice map not found: {path}") from e
        except yaml.YAMLError as e:
            raise UsageError(f"Voice map {path} is not valid YAML: {e}") from e
        if not isinstance(data, dict):
            raise UsageError(f"Voice map {path} must be a mapping with 'default' and 'speakers'.")
        speakers = data.get("speakers") or {}
        if not isinstance(speakers, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in speakers.items()
        ):
            raise UsageError(f"'speakers' in {path} must map names to voice names.")
        return cls(str(data.get("default") or default), {str(k): v for k, v in speakers.items()})
