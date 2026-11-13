#!/usr/bin/env bash
# Create .venv-tts with Chatterbox (the 'turbo' voice-cloning backend).
# Idempotent: safe to re-run. Override the interpreter with PYTHON=/path/to/python3.13.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PYTHON:-python3.13}"

if ! command -v "$PY" >/dev/null 2>&1; then
    echo "error: '$PY' not found. Install Python 3.13 or set PYTHON=/path/to/python." >&2
    exit 1
fi

if [ ! -x .venv-tts/bin/python ] && [ ! -f .venv-tts/Scripts/python.exe ]; then
    "$PY" -m venv .venv-tts
else
    echo ".venv-tts already exists, reusing it"
fi

if [ -x .venv-tts/bin/pip ]; then PIP=.venv-tts/bin/pip; else PIP=.venv-tts/Scripts/pip.exe; fi
"$PIP" install chatterbox-tts "setuptools<80" soundfile
echo "Done. The first 'turbo' run downloads ~2 GB of model weights."
