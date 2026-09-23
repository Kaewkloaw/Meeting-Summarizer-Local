"""Central configuration for the local-first meeting summarizer.

Everything here is local-only by design: the server binds to 127.0.0.1,
the LLM is a local Ollama model, and the only outbound network call this
app ever makes is an explicit yt-dlp download when the user pastes a
YouTube link. No audio, video, transcript, or summary is ever uploaded
anywhere.
"""

from pathlib import Path

# --- Network ---------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 8765

# --- Storage -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
ALLOWED_EXTENSIONS = {".mp3", ".m4a", ".wav", ".mp4", ".mov"}

# --- Transcription (mlx-whisper, Apple Silicon) -----------------------------
# Full-precision Whisper large-v3 (not the "turbo" distillation) for maximum
# accuracy on Thai conversational audio. Repo verified to exist on Hugging
# Face and to contain a valid mlx-whisper checkpoint (config.json + weights)
# before shipping this default: https://huggingface.co/mlx-community/whisper-large-v3-mlx
WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"
WHISPER_LANGUAGE = "th"

# --- Local LLM (Ollama) ------------------------------------------------------
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen3:4b"
OLLAMA_TIMEOUT_SECONDS = 600

# --- Output filenames (per job) ---------------------------------------------
TRANSCRIPT_FILENAME = "meeting_transcript_th.txt"
TRANSCRIPT_TIMESTAMPS_FILENAME = "meeting_transcript_th_timestamps.txt"
TRANSCRIPT_SEGMENTS_FILENAME = "meeting_transcript_th_segments.json"
SUMMARY_JSON_FILENAME = "meeting_summary.json"
SUMMARY_MD_FILENAME = "meeting_summary.md"
