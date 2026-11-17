"""Pausing for video playback (macOS power assertions)."""

from senpaidub import playback

PMSET = """\
Listed by owning process:
   pid 584(coreaudiod): [0x0001e363000184b2] 00:02:13 PreventUserIdleSystemSleep named: "com.apple.audio.output"
   pid 28685(Stremio): [0x0001e386000587c2] 00:01:39 PreventUserIdleDisplaySleep named: "Power Save Blocker"
   pid 423(WindowServer): [0x0001e32a0009878d] 00:01:30 UserIsActive named: "tickle"
   pid 365(powerd): [0x0001e34e00108578] 00:01:41 InternalPreventDisplaySleep named: "delayDisplayOff"
"""


def test_stremio_playing_is_detected():
    assert playback.parse_assertions(PMSET) == "Stremio"


def test_no_video_and_our_own_keep_awake_are_ignored():
    idle = "\n".join(line for line in PMSET.splitlines() if "Stremio" not in line)
    assert playback.parse_assertions(idle) is None
    caff = '   pid 99(caffeinate): [0x1] 00:00:10 PreventUserIdleDisplaySleep named: "caffeinate command-line tool"'
    assert playback.parse_assertions(caff) is None
    me = '   pid 4242(Python): [0x1] 00:00:10 PreventUserIdleDisplaySleep named: "x"'
    assert playback.parse_assertions(me, own_pid=4242) is None


def test_waits_until_the_video_stops():
    states = iter(["Stremio", "Stremio", "Stremio", None])
    slept: list[float] = []
    waited = playback.wait_while_video_plays(lambda: next(states), slept.append)
    assert len(slept) == 2 and waited == 2 * playback.CHECK_EVERY_S


def test_no_wait_without_video():
    slept: list[float] = []
    assert playback.wait_while_video_plays(lambda: None, slept.append) == 0.0 and not slept


def test_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("SENPAIDUB_PAUSE_FOR_VIDEO", "0")
    assert playback.video_playing() is None
