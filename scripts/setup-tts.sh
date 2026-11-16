#!/usr/bin/env bash
# Create .venv-qwen with Qwen3-TTS (the 'qwen' voice-cloning backend) and download its model.
# Idempotent: safe to re-run. Override the interpreter with PYTHON=/path/to/python3.13.
# Same as `senpaidub setup qwen`, except the environment lives in the project folder.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PYTHON:-python3.13}"
MODEL="${SENPAIDUB_QWEN_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"

if ! command -v "$PY" >/dev/null 2>&1; then
    echo "error: '$PY' not found. Install Python 3.13 or set PYTHON=/path/to/python." >&2
    exit 1
fi

if [ ! -x .venv-qwen/bin/python ] && [ ! -f .venv-qwen/Scripts/python.exe ]; then
    "$PY" -m venv .venv-qwen
else
    echo ".venv-qwen already exists, reusing it"
fi

if [ -x .venv-qwen/bin/python ]; then VPY=.venv-qwen/bin/python; else VPY=.venv-qwen/Scripts/python.exe; fi
"$VPY" -m pip install qwen-tts soundfile
"$VPY" -I -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')"
echo "Done. Qwen voices are ready ($MODEL)."
