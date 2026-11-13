"""Burned-in subtitle reading: the parts that don't need Vision."""

from __future__ import annotations

import numpy as np

from senpaidub import hardsubs
from senpaidub.hardsubs import FPS, Run


def _strip() -> np.ndarray:
    return np.full((hardsubs.STRIP, hardsubs.WIDTH, 3), 120, np.uint8)  # mid-grey picture


def test_text_mask_finds_outlined_strokes_not_white_areas():
    img = _strip()
    # A "letter": 3 px white stroke inside a dark outline.
    img[100:140, 600:609] = 0
    img[100:140, 603:606] = 255
    # A large white area with a dark border (a shirt in line art) is not text.
    img[20:90, 200:400] = 0
    img[23:87, 203:397] = 255
    m = hardsubs.text_mask(img)
    assert m[120, 604]
    assert not m[50, 300]
    assert np.count_nonzero(m[:, 200:400]) < 0.05 * np.count_nonzero(img[:, 200:400].min(axis=2) == 255)


def test_keep_line_drops_credits_and_off_centre_text():
    line = {"text": "Every single one of these women", "conf": 1.0, "x": 0.3, "w": 0.4, "y": 0.2, "h": 0.2}
    assert hardsubs._keep_line(line)
    assert not hardsubs._keep_line({**line, "text": "集英社 NBC"})
    assert not hardsubs._keep_line({**line, "x": 0.05, "w": 0.2})  # off-centre
    assert not hardsubs._keep_line({**line, "h": 0.07})  # small sign text
    assert not hardsubs._keep_line({**line, "conf": 0.2})


def test_clean():
    assert hardsubs.clean("We prepared her for that ******!") == "We prepared her for that !"
    assert hardsubs.clean("• Look, it says") == "Look, it says"
    assert hardsubs.clean("-We shouldn't") == "We shouldn't"
    assert hardsubs.clean('"Year 2040, Tokyo"') == ""  # on-screen caption, not dialogue
    assert hardsubs.clean("She's gone missing. but I have a lead.") == "She's gone missing, but I have a lead."
    assert hardsubs.clean("1 suppose so...") == "I suppose so..."
    assert hardsubs.clean("Room 1 is here.") == "Room 1 is here."


def test_to_events_merges_jitter_and_drops_flicker():
    runs = [
        Run(0, 12, text="Every single one of these women"),
        Run(12, 20, text="Every single one of these wornen"),  # OCR jitter, same subtitle
        Run(20, 33, text="has gathered here to mate with you."),
        Run(40, 42, text="Flicker"),  # one frame: too short to be a subtitle
        Run(50, 56, text=""),
    ]
    subs = hardsubs.to_events(runs)
    assert [e.text for e in subs] == ["Every single one of these women", "has gathered here to mate with you."]
    assert subs[0].start == 0 and subs[0].end == int(20 / FPS * 1000)
    assert subs[1].start == int(20 / FPS * 1000)


def test_bmp_header(tmp_path):
    p = tmp_path / "x.bmp"
    hardsubs.write_bmp(p, np.zeros((3, 5, 3), np.uint8))
    data = p.read_bytes()
    assert data[:2] == b"BM" and int.from_bytes(data[2:6], "little") == len(data)
    assert int.from_bytes(data[18:22], "little") == 5 and int.from_bytes(data[22:26], "little") == 3


def test_voices_by_register(tmp_path):
    import soundfile as sf

    from senpaidub.subtitles import DialogueLine
    from senpaidub.voicecheck import voices_by_register

    rate = 48000
    t = np.arange(rate) / rate

    def voice(hz: float) -> np.ndarray:  # a buzzy voiced sound with harmonics
        return sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 6)).astype(np.float32) * 0.2

    audio = np.concatenate([voice(120), voice(260), np.zeros(rate, np.float32), voice(110)])
    path = tmp_path / "vocals.wav"
    sf.write(path, audio, rate)
    lines = [DialogueLine(start=i * 1000 + 100, end=i * 1000 + 900, text="x", style="Default", speaker="")
             for i in range(4)]  # fmt: skip
    voices, males, females = voices_by_register(lines, path, "M", "F")
    assert voices == ["M", "F", "F", "M"]  # the silent line follows the previous one
    assert (males, females) == (2, 1)
