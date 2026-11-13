from pathlib import Path

import pysubs2
import pytest

from senpaidub.subtitles import (
    clean_text,
    filter_dialogue,
    is_default_skipped_style,
    load,
    style_words,
)

FIXTURES = Path(__file__).parent / "fixtures"


def make_subs(*events: tuple[str, str] | tuple[str, str, dict]) -> pysubs2.SSAFile:
    subs = pysubs2.SSAFile()
    t = 0
    for ev in events:
        style, text, *rest = ev
        kw = rest[0] if rest else {}
        subs.events.append(
            pysubs2.SSAEvent(start=kw.get("start", t), end=kw.get("end", t + 1000), style=style,
                             text=text, name=kw.get("name", ""), type=kw.get("type", "Dialogue"))
        )  # fmt: skip
        t += 1500
    return subs


# --------------------------------------------------------------------------- style names


@pytest.mark.parametrize(
    "style",
    ["Default", "Main", "Dialogue", "Italics", "Flashback", "Top", "Narration", "Overlap",
     "Thoughts", "Default - Top", "Main_Italic", "Edit", "Opal", "Speed",
     "Tone", "Notebook", "Kanade", "Signature", "Edward", "Opera"],
)  # fmt: skip
def test_dialogue_styles_are_not_skipped(style):
    assert not is_default_skipped_style(style), style


@pytest.mark.parametrize(
    "style",
    ["Sign", "Signs", "sign_top", "Song", "SongJP", "OP", "OP_Romaji", "op-kanji", "ED", "ED-Eng",
     "Opening", "Ending", "Karaoke", "Kara-OP", "Title", "Episode Title", "Note", "TL Note", "TLNote",
     "tl-note", "Insert Song", "Lyrics", "Romaji", "Kanji", "Eyecatch", "Preview", "OP1", "ED2 English"],
)  # fmt: skip
def test_non_dialogue_styles_are_skipped(style):
    assert is_default_skipped_style(style), style


def test_style_words_split():
    assert style_words("OP_Romaji") == ["OP", "Romaji"]
    assert style_words("SongJP") == ["Song", "JP"]
    assert style_words("TLNote") == ["TL", "Note"]
    assert style_words("ED2-Eng") == ["ED", "Eng"]


# --------------------------------------------------------------------------- cleanup


@pytest.mark.parametrize(
    "raw, clean",
    [
        ("{\\i1}Hello{\\i0} there", "Hello there"),
        ("Line one\\NLine two", "Line one Line two"),
        ("Line one\\nLine two", "Line one Line two"),
        ("- Who?\\N- Me.", "Who? Me."),
        ("— Wait!", "Wait!"),
        ("Too   many    spaces ", "Too many spaces"),
        ("Hard\\hspace", "Hard space"),
        ("<i>srt italics</i>", "srt italics"),
        ("Well-known - not a dash", "Well-known - not a dash"),
    ],
)
def test_clean_text(raw, clean):
    assert clean_text(raw) == clean


# --------------------------------------------------------------------------- filtering


def test_fixture_file_filtering():
    res = filter_dialogue(load(FIXTURES / "sample.ass"))
    texts = [l.text for l in res.lines]
    assert texts == [
        "Hello there.",
        "Lord Rimuru! Wait for me!",
        "Who are you? I'm Gobta.",
        "This one is very long and will certainly need to be sped up to fit its slot.",
        "The end.",
    ]
    assert [l.speaker for l in res.lines] == ["Rimuru", "Shion", "Gobta", "Benimaru", ""]
    reasons = res.reason_counts()
    assert reasons["sign/song style"] == 4  # Signs, OP_Romaji, ED-Eng, TLNote
    assert reasons["positioned/drawing"] == 2  # \pos in Default, \p1 drawing
    assert reasons["comment"] == 1
    assert reasons["duplicate layer"] == 1
    assert reasons["music/empty"] == 1
    assert reasons["no speakable text"] == 1


def test_positioning_tags():
    res = filter_dialogue(make_subs(
        ("Default", "{\\pos(1,2)}sign"),
        ("Default", "{\\move(1,2,3,4)}sign"),
        ("Default", "{\\org(1,2)\\frz10}sign"),
        ("Default", "{\\p2}m 0 0 l 1 1"),
        ("Default", "{\\an8}Top-aligned dialogue is fine"),
        ("Default", "{\\fad(100,100)}Fades are fine"),
        ("Default", "{\\p0}Drawing off is fine"),
        ("Default", "Literal \\pos outside a tag is just text"),
    ))  # fmt: skip
    assert [l.text for l in res.lines] == [
        "Top-aligned dialogue is fine",
        "Fades are fine",
        "Drawing off is fine",
        "Literal \\pos outside a tag is just text",
    ]


def test_comments_and_music():
    res = filter_dialogue(make_subs(
        ("Default", "a comment", {"type": "Comment"}),
        ("Default", "♪"),
        ("Default", "♪ ♫"),
        ("Default", "♪ Sing along ♪"),
        ("Default", "  "),
        ("Default", "Speak"),
    ))  # fmt: skip
    assert [l.text for l in res.lines] == ["Speak"]


def test_layered_duplicates_voiced_once_but_different_timing_kept():
    res = filter_dialogue(make_subs(
        ("Default", "Same", {"start": 0, "end": 1000}),
        ("Default", "{\\blur3}Same", {"start": 0, "end": 1000}),
        ("Default", "Same", {"start": 2000, "end": 3000}),
    ))  # fmt: skip
    assert len(res.lines) == 2


def test_include_styles_replaces_default_rules():
    subs = make_subs(("Signs", "{\\pos(1,1)}A sign"), ("Default", "Talk"), ("Song", "Lyrics"))
    res = filter_dialogue(subs, include_styles=r"^signs?$")
    assert [l.text for l in res.lines] == ["A sign"]


def test_exclude_styles_adds_to_defaults():
    subs = make_subs(("Default", "Talk"), ("Flashback", "Old talk"), ("Signs", "x"))
    res = filter_dialogue(subs, exclude_styles="flashback")
    assert [l.text for l in res.lines] == ["Talk"]


def test_lines_sorted_by_start():
    subs = make_subs(("Default", "second", {"start": 5000, "end": 6000}),
                     ("Default", "first", {"start": 1000, "end": 2000}))  # fmt: skip
    assert [l.text for l in filter_dialogue(subs).lines] == ["first", "second"]


def test_bad_regex_is_usage_error():
    from senpaidub.errors import UsageError

    with pytest.raises(UsageError):
        filter_dialogue(make_subs(("Default", "x")), include_styles="(")


def test_srt_loading():
    res = filter_dialogue(load(FIXTURES / "sample.srt"))
    assert [l.text for l in res.lines] == ["Good morning.", "Where are we going?"]


# --------------------------------------------------------------------------- OP/ED detection

from senpaidub.subtitles import chapter_song_spans, song_spans  # noqa: E402


def episode(op_style="Q0", op_lines=20, op_start=4000, dialogue_style="Q1", extra=()):
    """~24 min episode: lyric block at the start, dialogue every 6 s, lyric block at the end."""
    subs = pysubs2.SSAFile()
    add = lambda st, a, b, t: subs.events.append(pysubs2.SSAEvent(start=a, end=b, style=st, text=t))  # noqa: E731
    for i in range(op_lines):
        add(op_style, op_start + i * 4000, op_start + i * 4000 + 3500, f"lyric {i}")
    for t in range(100_000, 1_300_000, 6000):
        add(dialogue_style, t, t + 2500, f"dialogue at {t}")
    for i in range(op_lines):
        add(op_style, 1_330_000 + i * 4000, 1_333_500 + i * 4000, f"ending lyric {i}")
    for st, a, b, t in extra:
        add(st, a, b, t)
    return subs


def test_detects_generic_style_op_and_ed():
    spans = song_spans(episode())
    assert len(spans) == 2
    res = filter_dialogue(episode())
    assert res.reason_counts()["opening/ending"] == 40
    assert not any("lyric" in l.text for l in res.lines)
    assert res.lines[0].text.startswith("dialogue")


def test_main_dialogue_style_is_never_treated_as_song():
    # Only one style: the "lyrics" are the dominant style, so nothing is skipped.
    subs = episode(op_style="Q1")
    assert song_spans(subs) == []


def test_short_or_mid_episode_blocks_are_not_songs():
    short = episode(op_lines=6)  # ~24 s block: too short for an OP
    assert song_spans(short) == []
    # A long italics/flashback block in the middle of the episode isn't an OP/ED.
    mid = episode(op_lines=0, extra=[("Flashback", 600_000 + i * 4000, 603_000 + i * 4000, f"memory {i}")
                                     for i in range(20)])  # fmt: skip
    assert song_spans(mid) == []


def test_dialogue_inside_block_prevents_detection():
    # Thoughts in a secondary style interleaved with lots of main-style dialogue.
    subs = episode(op_lines=0, extra=[("Thoughts", 5000 + i * 4000, 7000 + i * 4000, f"thought {i}")
                                      for i in range(20)])  # fmt: skip
    for t in range(5000, 85_000, 3000):
        subs.events.append(pysubs2.SSAEvent(start=t, end=t + 1000, style="Q1", text="talk"))
    assert song_spans(subs) == []


def test_voice_songs_option_keeps_lyrics():
    res = filter_dialogue(episode(), skip_songs=False)
    assert res.reason_counts()["opening/ending"] == 0


def test_chapter_spans():
    chapters = [
        {"start_time": "0.0", "end_time": "90.0", "tags": {"title": "Opening"}},
        {"start_time": "90.0", "end_time": "1300.0", "tags": {"title": "Part A"}},
        {"start_time": "1300.0", "end_time": "1390.0", "tags": {"title": "ED"}},
        {"start_time": "1390.0", "end_time": "1420.0", "tags": {"title": "Preview"}},
        {"start_time": "0.0", "end_time": "1.0", "tags": {"title": "Edward's scene"}},
    ]
    assert chapter_song_spans(chapters) == [(0, 90_000), (1_300_000, 1_390_000)]
