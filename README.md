# Senpai Dub

**Hate reading subtitles? Senpai Dub voices any subbed anime in English.** It adds an English voice track generated from the episode's subtitles, keeps the original music and sound effects, and by default clones each Japanese actor's voice for their English lines.

```bash
senpaidub "Episode 01.mkv"
# -> "Episode 01 [EN dub].mkv": every original stream, plus a new "English (TTS)" audio track
```

## What you get

- **A new audio track, nothing replaced.** The output `.mkv` keeps the original video, audio and subtitle streams (copied, not re-encoded) and adds an `English (TTS)` track. Switch to it in VLC, mpv, Stremio or any player that lets you pick audio tracks.
- **The original music and effects stay.** [Demucs](https://github.com/facebookresearch/demucs) separates the Japanese voices from the background, so the English speech sits on top of the real soundtrack.
- **Characters keep their voices.** The default engine clones each line from the original Japanese line, so speakers sound distinct and shouting stays shouting.
- **Timed to the subtitles.** Each line starts at its subtitle time. Wherever no English line plays (screams, songs, the opening and ending), the original vocals are kept.
- **Works with most subtitle sources:** embedded text tracks (`.ass`/`.srt`), an external subtitle file, or subtitles burned into the picture (OCR, macOS only).

## Quick start (macOS)

You need [Homebrew](https://brew.sh). Tested on Apple Silicon.

```bash
brew install ffmpeg python@3.13

git clone https://github.com/<you>/senpaidub
cd senpaidub
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The default voice engine (`qwen`, voice cloning with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)) runs in a second virtual environment, because it needs a different PyTorch version than Demucs:

```bash
senpaidub setup qwen      # installs it in the app data folder and downloads the ~4.5 GB model
# or, inside the project folder: scripts/setup-tts.sh  (makes .venv-qwen)
```

Dub an episode:

```bash
senpaidub "Episode 01.mkv"
```

The first run downloads the Demucs model. Qwen runs on the GPU (Apple Silicon or NVIDIA) and voices several lines at once: about 10 minutes per 24-minute episode on an M5 MacBook Air with `--fast` (measured). By default it runs in a gentle mode that keeps the computer usable meanwhile: smaller batches, short pauses, and a model restart if its memory grows, which takes roughly twice as long. Set `qwen_mode = "fast"` in `config.toml`, or tick the box in the web UI settings, to always run at full speed. Without a GPU it is several times slower, so use `edge` there.

**Fast path (no model download):** skip the setup and use the free Microsoft Edge voices. Male and female voices are picked automatically from the pitch of each original line. This needs an internet connection.

```bash
senpaidub "Episode 01.mkv" --tts edge
```

### Linux and Windows

Install ffmpeg (`sudo apt install ffmpeg` or `winget install Gyan.FFmpeg`) and Python 3.10 to 3.13, then follow the same steps. On Windows, activate the venv with `.venv\Scripts\activate`. If you have an NVIDIA GPU, install a CUDA build of PyTorch first (see pytorch.org); Demucs then uses it automatically. Everything except burned-in subtitle OCR should work, but only macOS on Apple Silicon is tested.

## Usage

```bash
senpaidub ep.mkv --list-tracks          # show audio and subtitle streams
senpaidub ep.mkv --dry-run              # list the lines that will be spoken
senpaidub ep.mkv --dry-run --verbose    # ...plus every skipped line and the reason
senpaidub ep.mkv                        # full dub
senpaidub ep.mkv --sub-track 3          # voice a specific subtitle stream
senpaidub ep.mkv --sub-file ep01.ass    # use an external subtitle file
senpaidub ep.mkv --set-default          # make the English track the default
senpaidub ep.mkv --no-separate          # skip Demucs; lower the original audio under speech instead
senpaidub "Season 1/" -o dubbed/        # every .mkv/.mp4 in a folder, in episode order
```

Folder runs skip episodes that already have an output (use `--force` to redo them) and keep going if one episode fails. Run `senpaidub --help` for all options.

### Choosing a voice engine

| `--tts` | Quality | Speed (one episode) | Cost | Setup |
|---|---|---|---|---|
| `qwen` (default) | Best. Clones each original voice | ~10 min with `--fast` on an Apple Silicon GPU (M5), about twice that in the default gentle mode; much slower without a GPU | Free, runs locally | `senpaidub setup qwen` (~4.5 GB) |
| `edge` | Good, generic voices; male/female picked by pitch | Fast | Free (Microsoft Edge online voices) | None, needs internet |
| `google` | Very natural generic voices (Chirp 3 HD); male/female picked by pitch | Fast | Free tier, then paid (see below) | `pip install -e ".[google]"` + API key |

`mock` also exists, for tests only.

**Edge voices.** To pick voices yourself, use `--voice en-US-GuyNeural` for every line, or map speakers to voices with `--voice-map voices.yaml`. Many `.ass` files put the character name in the Name/Actor field:

```yaml
default: en-US-AriaNeural
speakers:
  Rimuru: en-US-GuyNeural
  Shion: en-US-JennyNeural
```

List the available voices with `edge-tts --list-voices`.

**Google.** Enable the Cloud Text-to-Speech API in the Google Cloud console and create an API key. Then either set `GOOGLE_API_KEY`, save the key in a file named `google-api-key` next to `config.toml` (see [Configuration](#configuration)), or set `GOOGLE_APPLICATION_CREDENTIALS` to a service-account file. Chirp 3 HD voices have a monthly free tier (about 1M characters at the time of writing); check Google's current pricing.

### Subtitles

senpaidub prefers an embedded English text subtitle track (`.ass`, `.srt` and similar) and skips signs/forced tracks. It filters out signs, songs, OP/ED lyrics, karaoke, translator notes and on-screen text so only dialogue is voiced (`--voice-songs` voices the OP/ED lyrics too). Tune the filter with `--include-styles` / `--exclude-styles` and check the result with `--dry-run`.

- **Wrong track picked?** Find the right one with `--list-tracks` and pass `--sub-track N`.
- **No subtitles in the file?** Pass `--sub-file episode.srt` (or `.ass`).
- **Burned-in (hardsubbed) subtitles:** if a file has no subtitle tracks and you give no `--sub-file`, macOS reads the subtitles off the picture with Apple's Vision OCR. A small Swift helper is compiled on first use, which needs the Xcode command-line tools (`xcode-select --install`). It takes about a minute per episode and only counts centred white text with a dark outline at the bottom of the frame as dialogue.
- **Image-based subtitle tracks (PGS, VobSub) are not supported.** Use `--sub-file` with a text version.

### Series mode

`--series-name` files episodes as a series in a library folder (`~/Downloads/anime` by default):

```bash
senpaidub ep05.mkv --series-name "My Show"
```

```
~/Downloads/anime/My Show/
  My Show - 05 [EN dub].mkv
  original/
    <the original episode file>
    subs/My Show - 05.ass
```

The original episode and its subtitles are moved into `original/`, so deleting the series folder removes everything. Per-episode caches are deleted after each dub; add `--keep-cache` to keep them for faster re-runs.

### Web UI

Prefer clicking to typing? Start the local web app:

```bash
senpaidub serve            # opens http://127.0.0.1:7860 in your browser
```

Drag in an episode (or pick a whole season folder), choose the subtitle track and voice engine, and watch the progress live. Batches run in a queue you can pause, and finished dubs can be played, revealed in their folder or downloaded as a zip. The settings page edits the same `config.toml` as the command line, and a button installs the qwen voices (the same as `senpaidub setup qwen`).

It runs only on your computer: the server listens on `127.0.0.1`, and the link it prints carries a private key, so other devices and other websites can't use it. `--port`, `--library` and `--no-browser` are available. Unfinished jobs resume the next time you run `senpaidub serve`.

### Stremio mode

If you watch with [Stremio](https://www.stremio.com), senpaidub can dub the episodes Stremio is currently streaming. It reads the list of loaded files from Stremio's local streaming server (`127.0.0.1:11470`), matches them against a series name, downloads them from that local server and dubs them in episode order:

```bash
senpaidub --stremio "my show" --series-name "My Show"
```

Start each episode playing in Stremio; once it starts you can go back. senpaidub keeps watching for new episodes until you press Ctrl+C (`--once` processes what is loaded and exits). Stremio must stay open. You can also pass a single Stremio stream URL as the input, usually with `--sub-file` and `--name`, since most streams carry no subtitles.

## How it works

```
video ─┬─ subtitles ──> filter dialogue ──────────────────────┐
       │                                                      v
       └─ audio ──> Demucs ─┬─ vocals ──> voice references ─> TTS per line ─> fit to subtitle timing
                            │      └──────────────────────────────────┐            │
                            └─ music + effects ──────────────────> mix <───────────┘
                                                                     │
                                         loudness match ─> encode ─> mux as "English (TTS)"
```

1. **Subtitles.** Pick the English text track (or `--sub-file`, or OCR) and keep only dialogue lines.
2. **Separate.** Demucs `htdemucs` splits the original audio into vocals and background, in 60 s chunks. Results are cached.
3. **Synthesize.** `qwen` clones each line's voice from that line's original vocals, in batches on the GPU; `edge`/`google` use stock voices. A male/female check regenerates cloned lines that came out in the wrong register. Clips are cached, so re-runs make no new TTS calls.
4. **Fit.** Each clip starts at its subtitle time and may run into the silence before the next line. If it still doesn't fit, it is sped up (up to `--max-speedup`, default 1.4x), then allowed to overflow with a warning. Speech is never cut.
5. **Mix.** Each line is matched to the original line's volume and mixed over the background. The original vocals are removed only where English lines play and kept everywhere else.
6. **Encode and mux.** The track is matched to the original's loudness, peak-limited, encoded as AAC 192k (or `--codec opus`) and added to a new `.mkv` with all original streams copied.

## Configuration

Defaults can go in `config.toml` in your user config folder (macOS: `~/Library/Application Support/senpaidub/`, Linux: `~/.config/senpaidub/`, Windows: `%LOCALAPPDATA%\senpaidub\`). `SENPAIDUB_CONFIG` points to a different file. Command-line flags override it.

```toml
tts = "edge"
max_speedup = 1.4
set_default = true
exclude_styles = "flashback"
library = "~/Videos/anime"   # where series folders go
keep_cache = false
# also: voice, voice_map, separate, duck_db, include_styles, codec, device, concurrency, speech_gain
```

Environment variables:

| Variable | Purpose |
|---|---|
| `SENPAIDUB_CACHE` | Cache folder (default: the user cache folder) |
| `SENPAIDUB_QWEN_PYTHON` | Python with `qwen-tts` installed, if not `.venv-qwen` in the project folder or the one `senpaidub setup qwen` made |
| `SENPAIDUB_QWEN_DEVICE` | Device for Qwen (default: CUDA, else Apple GPU (`mps`), else CPU) |
| `SENPAIDUB_QWEN_MODEL` | Qwen model (default `Qwen/Qwen3-TTS-12Hz-1.7B-Base`; `...-0.6B-Base` is smaller, similar speed on a GPU) |
| `SENPAIDUB_QWEN_BATCH` | Lines voiced at once (default 8, which keeps memory moderate on a 16 GB Mac; 16 is ~25% faster but uses ~2 GB more) |
| `GOOGLE_API_KEY` | Key for `--tts google` |

## Troubleshooting

- **Japanese voices play before or between English lines.** The original vocals are kept wherever no English line plays, so a line missing from the subtitles stays in Japanese. Check with `--dry-run --verbose` whether the line was skipped by the filter, or use a more complete subtitle file.
- **"overflows its slot" warnings.** The English line was longer than the gap available even at `--max-speedup`, so it runs into the next line. A few per episode are normal; raise `--max-speedup` to squeeze harder.
- **Warning about unauthenticated requests to the Hugging Face Hub.** Harmless. The Qwen model downloads without a token.
- **The first run is slow or seems stuck.** It is downloading models (Demucs, and about 4.5 GB for Qwen if `senpaidub setup qwen` didn't already). Later runs use the local copies.
- **"The 'qwen' voice-cloning backend needs its own Python environment".** Run `senpaidub setup qwen`, or set `SENPAIDUB_QWEN_PYTHON`, or use `--tts edge`.
- **"The 'turbo' voices were replaced by 'qwen'".** Remove `tts = "turbo"` from `config.toml` (or set it to `qwen`).
- **"No dialogue lines left after filtering".** Run `--dry-run --verbose` to see why lines were skipped, then adjust `--include-styles` / `--exclude-styles` or pick another `--sub-track`.
- **Running out of disk.** Caches are deleted per episode in series mode; otherwise they live in the user cache folder (or `SENPAIDUB_CACHE`) and can be deleted at any time.

Exit codes: `0` ok, `1` usage or config error, `2` no usable subtitles, `3` external tool (ffmpeg, TTS, Demucs) failed.

## Limitations

- It is text-to-speech from fan or official subtitles, not a studio dub: timing follows the subtitles, and lip sync is not attempted.
- Lines without a subtitle stay in Japanese. Overlapping speakers, whispers and very short lines are hit or miss.
- Voice cloning is slow on CPU and occasionally produces an odd line.
- Burned-in subtitle OCR is macOS only. Image-based subtitle tracks (PGS, VobSub) are not read.
- Output is always Matroska (`.mkv`). DRM-protected files are not supported.
- Tested on macOS with Apple Silicon only.

## Development

```bash
pip install -e ".[dev]"
pytest && ruff check src tests
```

The tests build small synthetic videos with ffmpeg and use the offline `mock` TTS backend (`--tts mock`), so they need ffmpeg but no network or models.

## Responsible use

senpaidub is meant for personal viewing of media you have the right to use. Please don't redistribute the dubbed files. It does not search for, provide or host any anime: it works on files you already have, and Stremio mode only downloads what your own Stremio app is already streaming. Cloned voices imitate the original voice actors, so keep the results to yourself.

## License

MIT. See [LICENSE](LICENSE).

Models and services used at runtime have their own licenses and terms: [Demucs](https://github.com/facebookresearch/demucs), [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache-2.0), Microsoft Edge TTS and Google Cloud Text-to-Speech. This project isn't affiliated with any of them or with Stremio.
