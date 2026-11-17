"""Subtitle extraction, parsing, dialogue filtering and text cleanup."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pysubs2

from . import ffmpeg
from .errors import NoSubtitlesError, UsageError
from .probe import Stream

# Style-name words that mark non-dialogue lines. Matched against whole *words* of the
# style name (split on punctuation, digits and camelCase), so "OP_Romaji", "Signs-2",
# "SongJP" and "TL Note" are skipped while "Top", "Edit", "Default", "Main" are not.
DEFAULT_SKIP_STYLES = (
    r"signs?|songs?|op|ed|opening|ending|karaoke|kara|titles?|notes?|tl ?notes?|tln|insert|"
    r"lyrics?|romaji|kanji|eyecatch|preview"
)
_SKIP_WORD_RE = re.compile(rf"^(?:{DEFAULT_SKIP_STYLES})$", re.IGNORECASE)

_OVERRIDE_RE = re.compile(r"\{[^}]*\}")
_POSITION_TAG_RE = re.compile(r"\\(?:pos|move|org)\s*\(|\\p[1-9]")
_MUSIC_CHARS = "♪♫♬♩"
_LEADING_DASH_RE = re.compile(r"^\s*[-–—‐]+\s*")


@dataclass(frozen=True)
class DialogueLine:
    start: int  # ms
    end: int  # ms
    style: str
    speaker: str
    text: str


@dataclass(frozen=True)
class SkippedLine:
    start: int
    end: int
    style: str
    text: str
    reason: str


@dataclass
class FilterResult:
    lines: list[DialogueLine]
    skipped: list[SkippedLine]

    def reason_counts(self) -> Counter[str]:
        return Counter(s.reason for s in self.skipped)


# --------------------------------------------------------------------------- extract/load


def extract(video: Path, stream: Stream, out_dir: Path) -> Path:
    """Extract one subtitle stream to .ass (keeps styles and speaker names) or .srt."""
    if stream.codec in ("ass", "ssa"):
        out, codec = out_dir / "subs.ass", "copy"
    elif stream.codec in ("subrip", "srt"):
        out, codec = out_dir / "subs.srt", "copy"
    else:  # webvtt, mov_text, ...
        out, codec = out_dir / "subs.srt", "srt"
    ffmpeg.run(
        ["ffmpeg", "-y", "-v", "error", "-i", video, "-map", f"0:{stream.index}", "-c:s", codec, out],
        what=f"Extracting subtitle stream #{stream.index}",
    )
    return out


def load(path: Path) -> pysubs2.SSAFile:
    if not path.is_file():
        raise UsageError(f"Subtitle file not found: {path}")
    try:
        return pysubs2.load(str(path), encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pysubs2.load(str(path), encoding="latin-1")
    except Exception as e:  # pysubs2 raises a few different types for bad files
        raise NoSubtitlesError(f"Could not parse subtitles in {path}: {e}") from e


# --------------------------------------------------------------------------- cleanup


def clean_text(raw: str) -> str:
    """Strip ASS override blocks, join line breaks, drop speaker dashes, collapse spaces."""
    text = _OVERRIDE_RE.sub("", raw)
    text = text.replace("\\h", " ")
    parts = re.split(r"\\[Nn]|\n", text)
    parts = [_LEADING_DASH_RE.sub("", p).strip() for p in parts]
    text = " ".join(p for p in parts if p)
    # Basic HTML-ish tags occasionally survive in SRT files.
    text = re.sub(r"</?[ibus](?:\s[^>]*)?>|</?font[^>]*>", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def style_words(style: str) -> list[str]:
    """Split a style name into words: 'OP_Romaji' -> [OP, Romaji], 'SongJP' -> [Song, JP]."""
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", style)  # camelCase boundary
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)  # 'TLNote' -> 'TL Note'
    return [w for w in re.split(r"[^A-Za-z]+", spaced) if w]


def is_default_skipped_style(style: str) -> bool:
    words = style_words(style)
    if any(_SKIP_WORD_RE.match(w) for w in words):
        return True
    # Two-word forms like "TL Note" / "Tl-Note".
    joined = " ".join(words)
    return bool(re.search(r"\btl ?notes?\b", joined, re.IGNORECASE))


def _is_music_only(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    # Lyrics wrapped in music markers ("♪ la la ♪") are songs, not dialogue.
    if stripped[0] in _MUSIC_CHARS or stripped[-1] in _MUSIC_CHARS:
        return True
    return False


def _has_speakable_content(text: str) -> bool:
    return any(c.isalnum() for c in text)


# --------------------------------------------------------------------------- OP/ED detection

SONG_MIN_S = 45  # TV-size OP/EDs run ~90 s; lyric blocks shorter than this aren't songs
SONG_MAX_S = 150
SONG_MIN_LINES = 8
SONG_MAX_GAP_S = 10  # a gap this long between lyric lines ends the block
SONG_EDGE_FRACTION = (0.4, 0.75)  # OP starts in the first 40%, ED ends after 75%
# Not "intro"/"outro": releases use those for the cold open and the after-credits scene, which
# are dialogue (Gushing over Magical Girls labels its 2-minute cold open "Intro").
CHAPTER_SONG_RE = re.compile(r"^(?:op|ed|opening|ending|credits|op\d|ed\d)\b", re.IGNORECASE)


def _covered_ms(events: list[pysubs2.SSAEvent]) -> int:
    """Total time on screen, counting overlapping events once."""
    total, end = 0, -1
    for e in sorted(events, key=lambda e: e.start):
        if e.end > end:
            total += e.end - max(e.start, end)
            end = e.end
    return total


def song_spans(subs: pysubs2.SSAFile) -> list[tuple[int, int]]:
    """Find opening/ending lyric blocks without relying on style names.

    Many releases (e.g. Crunchyroll rips) name styles generically ("Q0", "Q1"), so the
    style-name filter can't see the songs. A lyric block is: a run of lines in one
    non-dominant style, 45–150 s long, near the start or end of the episode, with almost no
    lines of the main dialogue style inside it.
    """
    events = [e for e in subs.events if not e.is_comment and e.end > e.start and clean_text(e.text)]
    if not events:
        return []
    counts = Counter(e.style for e in events)
    # The dialogue style is the one on screen the longest, not the one with the most events:
    # karaoke styles have one event per syllable (22,635 vs 210 dialogue lines in one release).
    main_style = max(counts, key=lambda st: _covered_ms([e for e in events if e.style == st]))
    total = max(e.end for e in events)
    spans: list[tuple[int, int]] = []
    for style in counts:
        if style == main_style:
            continue
        lines = sorted((e for e in events if e.style == style and not _POSITION_TAG_RE.search(e.text)),
                       key=lambda e: e.start)  # fmt: skip
        block: list[pysubs2.SSAEvent] = []
        for e in [*lines, None]:
            if e is not None and (not block or e.start - block[-1].end <= SONG_MAX_GAP_S * 1000):
                block.append(e)
                continue
            if block:
                a, b = block[0].start, max(x.end for x in block)
                dialogue_inside = sum(1 for x in events if x.style == main_style and a <= x.start < b)
                if (
                    len(block) >= SONG_MIN_LINES
                    and SONG_MIN_S * 1000 <= b - a <= SONG_MAX_S * 1000
                    and (a <= total * SONG_EDGE_FRACTION[0] or b >= total * SONG_EDGE_FRACTION[1])
                    and dialogue_inside <= len(block) * 0.2
                ):
                    spans.append((a, b))
            block = [e] if e is not None else []
    return sorted(spans)


def chapter_song_spans(chapters: list[dict]) -> list[tuple[int, int]]:
    """OP/ED spans from container chapters titled e.g. 'Opening' / 'ED'."""
    spans = []
    for c in chapters:
        title = ((c.get("tags") or {}).get("title") or "").strip()
        if CHAPTER_SONG_RE.match(title):
            spans.append((int(float(c["start_time"]) * 1000), int(float(c["end_time"]) * 1000)))
    return spans


# --------------------------------------------------------------------------- filter


def filter_dialogue(
    subs: pysubs2.SSAFile,
    include_styles: str | None = None,
    exclude_styles: str | None = None,
    *,
    skip_songs: bool = True,
    extra_song_spans: list[tuple[int, int]] | None = None,
) -> FilterResult:
    """Keep only spoken lines. See the project spec §5.3 for the rules.

    With skip_songs, lines inside the opening/ending (detected from the subtitles, plus any
    `extra_song_spans` such as chapter markers) aren't voiced; the original audio plays there.
    """
    try:
        include_re = re.compile(include_styles, re.IGNORECASE) if include_styles else None
        exclude_re = re.compile(exclude_styles, re.IGNORECASE) if exclude_styles else None
    except re.error as e:
        raise UsageError(f"Invalid style regex: {e}") from e

    kept: list[DialogueLine] = []
    skipped: list[SkippedLine] = []
    seen: set[tuple[int, int, str]] = set()
    songs = (song_spans(subs) + list(extra_song_spans or [])) if skip_songs else []

    for ev in subs.events:
        style = ev.style or ""
        raw = ev.text or ""

        def skip(reason: str, text: str = raw) -> None:
            skipped.append(SkippedLine(ev.start, ev.end, style, text, reason))  # noqa: B023 (called within this iteration)

        if ev.is_comment:
            skip("comment")
            continue
        if ev.end <= ev.start:
            skip("zero length")
            continue
        if any(a <= ev.start < b for a, b in songs):
            skip("opening/ending")
            continue
        if include_re is not None:
            if not include_re.search(style):
                skip("style not included")
                continue
        else:
            if is_default_skipped_style(style):
                skip("sign/song style")
                continue
            if _POSITION_TAG_RE.search(" ".join(_OVERRIDE_RE.findall(raw))):
                skip("positioned/drawing")
                continue
        if exclude_re is not None and exclude_re.search(style):
            skip("excluded style")
            continue

        text = clean_text(raw)
        if _is_music_only(text):
            skip("music/empty", text)
            continue
        text = text.strip(_MUSIC_CHARS + " ")
        if not _has_speakable_content(text):
            skip("no speakable text", text)
            continue
        key = (ev.start, ev.end, text)
        if key in seen:
            skip("duplicate layer", text)
            continue
        seen.add(key)
        kept.append(DialogueLine(ev.start, ev.end, style, (ev.name or "").strip(), text))

    kept.sort(key=lambda l: (l.start, l.end))
    return FilterResult(kept, skipped)


def fmt_time(ms: int) -> str:
    s, ms = divmod(max(ms, 0), 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}.{ms // 10:02d}"
