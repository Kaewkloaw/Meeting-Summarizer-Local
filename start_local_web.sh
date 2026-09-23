#!/usr/bin/env bash
# Installs dependencies (if missing) and starts the local-only meeting
# summarizer web app at http://127.0.0.1:8765
#
# Everything this script installs/runs stays on this machine:
#   - ffmpeg, yt-dlp, Ollama, mlx-whisper: all local tools/models
#   - the web server binds to 127.0.0.1 only (not reachable from other devices)
#
# Requirements: macOS on Apple Silicon (M-series). Transcription runs on
# mlx-whisper, which depends on Apple's MLX framework and only works there.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"

echo "== Local Meeting Summarizer: setup =="

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: this app requires macOS on Apple Silicon (arm64)."
  echo "Transcription uses mlx-whisper, which only runs on Apple's MLX framework."
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install it from https://brew.sh, then re-run this script."
  exit 1
fi

# --- ffmpeg -------------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "-- Installing ffmpeg via Homebrew..."
  brew install ffmpeg
else
  echo "-- ffmpeg already installed"
fi

# --- Ollama ---------------------------------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
  echo "-- Installing Ollama via Homebrew..."
  brew install ollama
else
  echo "-- Ollama already installed"
fi

if ! curl -s -o /dev/null "http://127.0.0.1:11434/api/version"; then
  echo "-- Starting local Ollama server..."
  (nohup ollama serve >/tmp/local-meeting-web-ollama.log 2>&1 &)
  for i in $(seq 1 30); do
    if curl -s -o /dev/null "http://127.0.0.1:11434/api/version"; then
      break
    fi
    sleep 1
  done
fi

if ! ollama list 2>/dev/null | grep -q "qwen3:4b"; then
  echo "-- Pulling model qwen3:4b (first time only, may take a while)..."
  ollama pull qwen3:4b
else
  echo "-- Model qwen3:4b already available"
fi

# --- Python venv ------------------------------------------------------------
PYTHON_BIN="python3"
if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
elif command -v /opt/homebrew/bin/python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="/opt/homebrew/bin/python3.11"
fi
echo "-- Using Python: $PYTHON_BIN ($($PYTHON_BIN --version))"

# A venv created under a different path (e.g. the project folder was moved
# or renamed) leaves a stale shebang/symlink pointing at a Python binary
# that no longer exists at that path. Detect that and recreate instead of
# failing deep inside `pip install` with a confusing "No such file or
# directory" error.
if [[ -d "$VENV_DIR" ]] && ! "$VENV_DIR/bin/python3" --version >/dev/null 2>&1; then
  echo "-- Existing .venv is broken (likely moved/renamed project folder). Recreating..."
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "-- Creating virtual environment (.venv)..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "-- Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo ""
echo "== Starting the web app at http://127.0.0.1:8765 (reachable only from this machine) =="
echo "   Press Ctrl+C to stop"
echo ""

open "http://127.0.0.1:8765" >/dev/null 2>&1 || true

exec python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
