import numpy as np
import pytest

from senpaidub.subtitles import DialogueLine
from senpaidub.timing import (
    RATE,
    PlacedClip,
    atempo_chain,
    render_speech,
    slot_ends,
    speed_factor,
    trim_silence,
)


def L(start, end):
    return DialogueLine(start, end, "Default", "", "x")


def test_slot_extends_into_gap_until_next_line():
    lines = [L(0, 1000), L(3000, 4000), L(3500, 5000), L(6000, 7000)]
    assert slot_ends(lines) == [3000, 4000, 6000, 7000]


def test_slot_does_not_shrink_when_next_line_overlaps():
    assert slot_ends([L(0, 2000), L(1000, 3000)]) == [2000, 3000]


def test_last_line_uses_own_end():
    assert slot_ends([L(0, 1000)]) == [1000]


@pytest.mark.parametrize(
    "clip, slot, cap, expected",
    [
        (1.0, 2.0, 1.4, 1.0),  # fits
        (2.0, 2.0, 1.4, 1.0),  # exactly fits
        (2.02, 2.0, 1.4, 1.0),  # <2% over: not worth stretching
        (2.4, 2.0, 1.4, 1.2),  # needs 1.2x
        (4.0, 2.0, 1.4, 1.4),  # capped
        (4.0, 2.0, 3.0, 2.0),
        (1.0, 0.0, 1.4, 1.0),  # degenerate slot
        (4.0, 2.0, 0.8, 1.0),  # cap below 1 never slows speech down
    ],
)
def test_speed_factor(clip, slot, cap, expected):
    assert speed_factor(clip, slot, cap) == pytest.approx(expected)


def test_atempo_chain_splits_large_factors():
    assert atempo_chain(1.25) == "atempo=1.250000"
    chain = atempo_chain(3.0)
    parts = [float(p.split("=")[1]) for p in chain.split(",")]
    assert parts == pytest.approx([2.0, 1.5])
    assert all(0.5 <= p <= 2.0 for p in parts)
    assert np.prod(parts) == pytest.approx(3.0)


def test_trim_silence_keeps_small_margin():
    x = np.zeros(RATE, np.float32)
    x[RATE // 4 : RATE // 2] = 0.5
    y = trim_silence(x)
    assert RATE // 4 <= y.size <= RATE // 4 + int(0.03 * RATE)


def test_render_speech_blocks_add_up():
    line = L(0, 1000)
    a = PlacedClip(line, 100, np.ones(300, np.float32), 1.0, 0.0)
    b = PlacedClip(line, 350, np.ones(100, np.float32), 1.0, 0.0)  # overlaps a
    whole = render_speech([a, b], 0, 600)
    pieces = np.concatenate([render_speech([a, b], s, 128) for s in range(0, 640, 128)])[:600]
    np.testing.assert_array_equal(whole, pieces)
    assert whole[99] == 0 and whole[100] == 1 and whole[360] == 2 and whole[400] == 1
