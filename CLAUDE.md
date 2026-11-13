# Agent guide: dubbing anime with Senpai Dub (`senpaidub`)

This file is for AI coding agents (Claude Code, Codex, Cursor and others). A user handed you this repo because they want anime episodes dubbed into English. Read this first; [README.md](README.md) has the human-facing details.

## What the tool does

`senpaidub` takes a video with Japanese audio and English subtitles. It writes `<name> [EN dub].mkv`, which has every original stream plus a new `English (TTS)` audio track. Demucs removes the Japanese voices, TTS voices each subtitle line, and the result is mixed over the original music and effects. Only the audio is added; the video is never re-encoded.

## Your job, in order

1. **Set up** (once). See "Setup" below. Check before installing; it may already be done.
2. **Find the episodes** the user means. Ask only if you can't locate them.
3. **Inspect one episode** before a batch:
   - `senpaidub EP --list-tracks` shows the audio and subtitle streams.
   - `senpaidub EP --dry-run` shows how many lines will be spoken. A normal episode has 250–450.
4. **Pick the voice engine.** See "Choosing --tts"; ask if the user hasn't said.
5. **Run it.** One episode, a folder, or a whole series. See "Commands".
6. **Verify.** See "Checking the output".
7. **Report.** Say where the files are, how many lines were voiced, and any warnings worth mentioning.

## Setup

Requirements: ffmpeg/ffprobe on PATH, and Python **3.10–3.13** (PyTorch has no wheels for newer versions yet).

```bash
# macOS
brew install ffmpeg python@3.13
python3.13 -m venv .venv && .venv/bin/pip install -e .
scripts/setup-tts.sh          # only for --tts turbo (default): makes .venv-tts, ~2 GB model on first run
```

- Linux: `apt install ffmpeg`. Windows: `winget install Gyan.FFmpeg`, then the same venv steps (`.venv\Scripts\...`).
- NVIDIA GPU: install a CUDA build of PyTorch into `.venv` *before* `pip install -e .`.
- Check: `.venv/bin/senpaidub --version`, plus `.venv-tts/bin/python -c "import chatterbox"` for turbo.
- `.venv-tts` must sit in the repo root, or set `SENPAIDUB_CLONE_PYTHON` to a Python that has `chatterbox-tts`.

## Choosing `--tts`

| Engine | Pick it when | Time per 24-min episode | Notes |
|---|---|---|---|
| `turbo` (default) | the user wants the best result | ~20–25 min on an Apple M-series CPU, ~4 GB RAM | Clones each original voice and emotion; local; no API key |
| `edge` | the user wants it fast, or turbo isn't installed | ~3–5 min | Free online Microsoft voices; male/female picked by pitch; needs internet |
| `google` | the user has a Google Cloud API key | ~3–5 min | Chirp 3 HD voices; key in `GOOGLE_API_KEY` or a `google-api-key` file next to config.toml |

Vocal separation (Demucs) runs for every engine and takes a few minutes per episode. It uses CUDA/MPS automatically.

## Commands

```bash
senpaidub "ep01.mkv"                                     # one episode -> "ep01 [EN dub].mkv" next to it
senpaidub "Season 1/" -o dubbed/                         # a folder, in episode order; skips done ones
senpaidub ep.mkv --series-name "Show Name"               # file into ~/Downloads/anime/Show Name/ (originals -> original/)
senpaidub ep.mkv --sub-track 3                           # a specific subtitle stream from --list-tracks
senpaidub ep.mkv --sub-file ep01.ass                     # external subtitles
senpaidub ep.mkv --tts edge                              # fast voices
senpaidub ep.mkv --force                                 # redo even if the output exists
```

**A whole season the user wants filed as a series:** loop over the files in episode order with `--series-name`. One `senpaidub` call per file keeps a single failure from stopping the rest:

```bash
for f in "/path/to/Show"/*.mkv; do senpaidub "$f" --series-name "Show" || echo "FAILED: $f"; done
```

**Long runs** (turbo takes about 4–5 hours per season): run in the background and log to a file. Lower the priority so the machine stays usable. On macOS, keep it awake and remind the user to stay on the charger with the lid open:

```bash
caffeinate -is nice -n 10 zsh -c 'for f in ...; do senpaidub "$f" --series-name "Show"; done' > dub.log 2>&1
```

Report progress from the log. Each episode ends with a `✓ <path>` line and a `N lines voiced` summary.

## Subtitles: the part that most often needs judgment

- The tool prefers an embedded English **text** track (.ass/.srt) and filters out signs, songs, OP/ED, karaoke and TL notes. Check the filtering with `--dry-run --verbose`.
- **Wrong or empty track:** use `--list-tracks`, then `--sub-track N`. If `--dry-run` shows very few lines, the wrong track was probably picked.
- **No subtitle track at all** (burned-in/hardsubbed video): on macOS the subtitles are read off the picture with Vision OCR automatically. That needs the Xcode command-line tools for `swiftc`. Elsewhere, ask the user for an `.srt`/`.ass` and pass `--sub-file`.
- Image subtitle tracks (PGS/VobSub) aren't supported, so a text `--sub-file` is required.
- Too much filtered: `--include-styles REGEX`. Too little filtered: `--exclude-styles REGEX`.

## Checking the output

```bash
ffprobe -v error -show_entries stream=index,codec_type:stream_tags=language,title -of compact "OUT.mkv"
```

There should be an extra audio stream titled `English (TTS)`. Warnings are normal and not failures:
- "overflows its slot": a long line ran into the next one. A few per episode is fine.
- "unauthenticated requests to the HF Hub": harmless.
- "voice check: N mix-ups fixed": the tool corrected male/female voices by itself.

**Exit codes:**
- `0`: ok.
- `1`: usage or config error. Read the message; it usually names the fix.
- `2`: no usable subtitles. Use `--sub-track`, `--sub-file`, or hardsub OCR on macOS.
- `3`: ffmpeg, TTS or Demucs failed. Rerun with `-v` for the failing command.

If the user says Japanese voices are heard before the English line, that dialogue line is missing from the subtitles. It isn't a TTS problem. Check `--dry-run --verbose` or find better subtitles.

## Disk and cleanup

- Outputs are about the same size as the input.
- Caches (separated vocals, TTS clips) live in the user cache dir, or in `SENPAIDUB_CACHE` if set. Series mode deletes them after each episode unless `--keep-cache` is given. They're safe to delete anytime.
- Don't delete the user's original videos. Series mode *moves* them into `<series>/original/`.

## Rules

- Dubs are for the user's personal viewing. Don't upload, share or redistribute output files, and don't download anime on the user's behalf unless they explicitly ask and the source is theirs to use.
- Never commit videos, subtitles, model weights or API keys. `.gitignore` covers them, so keep it that way.
- Ask before installing system packages (brew/apt/winget) or anything large.

## Changing the code

Layout, all under `src/senpaidub/`:
- `cli.py`: Typer CLI.
- `pipeline.py`: per-episode orchestration.
- `subtitles.py`: track choice and dialogue filtering.
- `hardsubs.py` + `ocr_vision.swift`: burned-in subtitle OCR.
- `separate.py`: Demucs.
- `tts/`: engines. `clone.py` and `clone_worker.py` run Chatterbox in `.venv-tts`.
- `timing.py`: fitting lines into their slots.
- `levels.py`, `mix.py`: loudness and mixing.
- `voicecheck.py`: male/female correction.
- `mux.py`: final mkv.
- `batch.py`, `stremio.py`: multi-episode modes.

Checks: run `.venv/bin/pytest -q` (offline; uses the `mock` TTS engine and synthetic ffmpeg videos) and `.venv/bin/ruff check src tests`. Both must pass. Add a test with each behaviour change.
