# Local Meeting Summarizer

A local-first web app that summarizes Thai-language meetings **entirely on your own machine**.
Upload an audio/video file or paste a YouTube link, and the app will download (YouTube only) →
transcribe the Thai audio → let you review the transcript → summarize the meeting with a local
LLM. No manual copy-pasting of transcripts required, and nothing leaves your computer except an
explicit YouTube download.

## Requirements

| Component | Requirement |
|---|---|
| OS / hardware | **macOS on Apple Silicon (M-series)** — required, see below |
| Homebrew | https://brew.sh |
| Python | 3.11 (installed automatically via Homebrew if missing) |
| Disk space | A few GB free for the Whisper model, the Ollama model, and job data |

This app is Apple Silicon–only because transcription runs on
[`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper), which is built on
Apple's [MLX](https://github.com/ml-explore/mlx) framework and only runs on Apple GPUs. There is
currently no Linux/Windows/Intel-Mac path for the transcription stage — everything else
(FastAPI server, Ollama, ffmpeg, yt-dlp) is cross-platform, but this specific dependency is not.

## Quick start

```bash
git clone <this-repo-url>
cd local-meeting-web
./start_local_web.sh
```

The script is idempotent — safe to re-run any time — and on each run it will:

1. Check you're on macOS + Apple Silicon (exits with a clear error otherwise).
2. Install `ffmpeg` and `ollama` via Homebrew (skipped if already installed).
3. Start the local Ollama server and pull the `qwen3:4b` model (skipped if already present).
4. Create/repair a Python virtual environment (`.venv`) and install `requirements.txt`
   (including `mlx-whisper` for on-device transcription).
5. Open `http://127.0.0.1:8765` in your browser and start the server.

On first use, `mlx-whisper` downloads the `mlx-community/whisper-large-v3-mlx` model from
Hugging Face and caches it locally. This is the full (non-"turbo") Whisper large-v3 model —
larger and slower than the turbo distillation, but more accurate, which matters for Thai
conversational audio.

### Manual install (step by step)

If you'd rather not run the setup script, or want to understand/customize each step:

1. **Install Homebrew** (if you don't have it): https://brew.sh

2. **Install ffmpeg**, used to normalize audio to 16kHz mono WAV before transcription:
   ```bash
   brew install ffmpeg
   ```

3. **Install and start Ollama**, the local LLM runtime used for summarization:
   ```bash
   brew install ollama
   ollama serve &          # starts the local API at http://127.0.0.1:11434
   ollama pull qwen3:4b    # the model this app uses by default
   ```
   (Ollama also ships official installers for Linux and Windows at
   [ollama.com/download](https://ollama.com/download) — but see [Requirements](#requirements):
   the *transcription* stage of this app still needs macOS + Apple Silicon regardless of where
   Ollama itself can run.)

4. **Create a Python virtual environment and install dependencies**:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Run the server**:
   ```bash
   source .venv/bin/activate
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
   ```
   Then open http://127.0.0.1:8765 in your browser.

### Troubleshooting

- **`pip install` fails with "No such file or directory" pointing at a path inside `.venv`** —
  the virtual environment was created before the project folder was moved or renamed, so its
  Python symlink now points nowhere. Delete and recreate it:
  ```bash
  rm -rf .venv
  python3.11 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
  (`start_local_web.sh` detects and repairs this automatically on its next run.)
- **"connect to Ollama failed" / summarization step fails** — make sure `ollama serve` is
  running and `ollama list` shows `qwen3:4b`. Re-run `./start_local_web.sh` to fix both.
- **ffmpeg errors about a corrupt/incomplete file** — the source recording itself is likely
  truncated or invalid; open it in a media player first to confirm it plays back normally.

## How it works

1. Open `http://127.0.0.1:8765`.
2. Pick a file (MP3, M4A, WAV, MP4, MOV) or paste a YouTube link.
3. Click "Start transcription" and wait through: downloading (YouTube only) → transcribing.
4. Once transcription finishes, the app **stops and waits** — it does not summarize
   automatically. Download the transcript and review it first:
   - `meeting_transcript_th.txt` — plain text
   - `meeting_transcript_th_timestamps.txt` — with per-segment timestamps against the audio
   - `meeting_transcript_th_segments.json` — raw structured data (start/end/text per segment)
5. Once you've confirmed the transcript is accurate, click **"Summarize meeting"** to start the
   summarization stage (calls the local Ollama model).
6. Wait for it to complete, then download `meeting_summary.json` and `meeting_summary.md`.

Every job's output is kept locally at `local-meeting-web/jobs/<job_id>/`.

## Privacy

- The server only listens on `127.0.0.1:8765` — reachable from this machine only.
- **No** cloud ASR or cloud LLM is used anywhere. Transcription runs via `mlx-whisper`
  (`mlx-community/whisper-large-v3-mlx`, the full large-v3 model — not turbo — for maximum
  accuracy), and summarization runs via the local Ollama API (`qwen3:4b`), all on this machine.
- Audio, video, transcripts, and summaries are **never uploaded** anywhere.
- For YouTube links: the only outbound network call is `yt-dlp` downloading the source audio.
  Everything after that runs locally. **Only use content you have the right to download and
  process.**
- Every summary is generated automatically and **must be reviewed by a human** before real use —
  the app enforces a transcript-review step before you can trigger summarization (see
  "How it works" above).

## Summary output structure (`meeting_summary.md`)

- Key points
- Decisions
- Action items (task / owner / due date / evidence quoted from the transcript)
- Next steps
- Requirements
- Items flagged for further review

Rules enforced on the local LLM: only use information present in the transcript, never guess
names/dates/numbers/conclusions, use "not specified" when an owner or due date is missing, and
every action item must cite supporting evidence quoted from the transcript.

## Architecture

Transcription and summarization each run in their **own separate OS process**, always launched
fresh from the FastAPI server — this is a hard requirement, not an optimization:

- `mlx-whisper` (MLX) binds its GPU command stream to whichever thread first uses it. Calling it
  from FastAPI's `BackgroundTasks`, its request threadpool, or `asyncio.to_thread` raises
  `There is no Stream(cpu, 0) in current thread`, because none of those guarantee the same thread
  twice.
- `app/main.py` and `app/pipeline.py` (the FastAPI server) **never import `mlx_whisper`**. They
  only create a job and `subprocess.Popen` a new `app/worker.py` process for it.
- Two independent stages, each its own process:
  1. `--stage transcribe`: ffmpeg → mono 16kHz WAV → `mlx-whisper` transcription → explicitly
     release the ASR model (`transcribe.release_model()`) → process exits. This guarantees the
     model is fully released before summarization ever runs.
  2. `--stage summarize`: a brand-new process that reads the existing transcript and only calls
     Ollama — it never imports `mlx_whisper`.
- The server and worker processes communicate only through `status.json` in the job's directory
  (`app/job_status.py`) — there is no shared in-memory state between them.

## Project structure

```
local-meeting-web/
├── app/
│   ├── main.py          # FastAPI routes (never imports mlx_whisper)
│   ├── pipeline.py      # creates jobs + subprocess.Popen's worker.py (never imports mlx_whisper)
│   ├── worker.py        # standalone process: --stage transcribe or --stage summarize
│   ├── job_status.py    # reads/writes status.json (the only channel between server and worker)
│   ├── transcribe.py    # ffmpeg + mlx-whisper (only ever imported inside the worker process)
│   ├── summarize.py     # prompt construction + local Ollama API calls
│   ├── youtube.py       # yt-dlp (the only place that makes an outbound network call)
│   └── config.py
├── static/              # CSS/JS
├── templates/           # HTML
├── jobs/                # per-job output (created automatically at runtime)
├── requirements.txt
├── start_local_web.sh
└── README.md
```

## Limitations

- Requires macOS on Apple Silicon (M-series) — `mlx-whisper` depends on the MLX framework.
- Requires Ollama with the `qwen3:4b` model pulled locally (`start_local_web.sh` does this
  automatically).
- Summary quality depends on source audio quality and transcription accuracy — always review
  results before relying on them.
