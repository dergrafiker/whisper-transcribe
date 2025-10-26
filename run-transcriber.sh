#!/usr/bin/env bash
set -euo pipefail

# --- config ---
VENV_DIR=".venv"
REQ_FILE="requirements.txt"
SCRIPT="distil_whisper_transcribe.py"

# We need at least one argument: the audio file.
if [ $# -lt 1 ]; then
    echo "Usage: $0 /pfad/zur/audio.m4a [--model-size small|medium|large-v3] [--device auto|cpu|cuda]" >&2
    exit 1
fi

AUDIO_FILE="$1"
shift 1  # remove the first arg so "$@" now contains optional flags

# 1. check uv
if ! command -v uv >/dev/null 2>&1; then
    echo "Fehler: 'uv' nicht im PATH." >&2
    echo "Bitte uv installieren (https://astral.sh/uv/) und erneut versuchen." >&2
    exit 2
fi

# 2. create venv if missing
if [ ! -d "$VENV_DIR" ]; then
    echo "[+] Erzeuge venv in $VENV_DIR ..."
    uv venv "$VENV_DIR"
fi

# 3. install/update deps from requirements.txt into that venv
echo "[+] Synchronisiere Dependencies aus $REQ_FILE ..."
uv pip install -r "$REQ_FILE" --python "$VENV_DIR/bin/python"

# 4. run the Python transcriber inside that venv
echo "[+] Starte Transkription ..."
uv run --python "$VENV_DIR/bin/python" "$SCRIPT" "$AUDIO_FILE" "$@"
