"""Standalone worker process: download -> transcribe, and (separately) summarize.

WHY THIS FILE EXISTS: mlx-whisper (MLX) creates a GPU command stream that is
bound to the OS thread that first touches it. FastAPI's BackgroundTasks,
its request threadpool, and asyncio.to_thread all run work on worker threads
that are NOT the process's main thread and are not guaranteed to be the same
thread twice — calling mlx_whisper.transcribe() from any of them raises
"There is no Stream(cpu, 0) in current thread".

The fix used here: this script is launched as its own OS process via
`subprocess.Popen([sys.executable, "-m", "app.worker", ...])` from
app/pipeline.py. It does all the work on ITS OWN main thread, in ITS OWN
process, and communicates status back to the FastAPI server only by writing
`status.json` into the job directory (see app/job_status.py). The FastAPI
server process never imports mlx_whisper and never calls mlx_whisper.transcribe.

TWO STAGES, TWO SEPARATE PROCESSES: transcription (`--stage transcribe`) and
summarization (`--stage summarize`) are launched as independent invocations
of this script. This does two things at once:
  1. It guarantees the ASR model is fully released (the whole OS process
     exits) before the Ollama call for summarization ever runs.
  2. It lets the user review/download the raw transcript between the two
     stages and only trigger summarization once they're satisfied with the
     transcription quality (the FastAPI server only spawns the summarize
     stage when the user calls POST /api/jobs/{id}/summarize).

Everything still runs on this machine only: ffmpeg, mlx-whisper, and the
Ollama call to 127.0.0.1:11434. The only outbound network request is the
explicit yt-dlp download for a YouTube source.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from app import config, job_status, summarize, transcribe, youtube


class StatusWriter:
    """Tracks per-stage wall-clock time and writes status.json as we go."""

    def __init__(self, job_dir: Path, base_timings: dict | None = None):
        self.job_dir = job_dir
        self.timings: dict[str, float] = dict(base_timings or {})
        self._stage_started_at: dict[str, float] = {}
        self._overall_started_at = time.monotonic()

    def start_stage(self, stage: str) -> None:
        self._stage_started_at[stage] = time.monotonic()

    def end_stage(self, stage: str) -> None:
        started = self._stage_started_at.get(stage)
        if started is not None:
            self.timings[stage] = round(time.monotonic() - started, 2)

    def write(self, status: str, message: str, percent: int, error: str | None = None) -> None:
        job_status.write_status(self.job_dir, status, message, percent, error=error, timings=self.timings)

    def add_elapsed(self, key: str, seconds: float) -> None:
        self.timings[key] = round(seconds, 2)


def run_transcribe_from_youtube(job_dir: Path, url: str, source_label: str) -> None:
    sw = StatusWriter(job_dir)
    try:
        sw.start_stage("downloading")
        sw.write("downloading", "กำลังดาวน์โหลดไฟล์เสียงจาก YouTube...", 0)

        def on_download_progress(fraction: float) -> None:
            sw.write("downloading", "กำลังดาวน์โหลดไฟล์เสียงจาก YouTube...", job_status.stage_percent("downloading", fraction))

        audio_path = youtube.download_audio(url, job_dir, progress_callback=on_download_progress)
        sw.end_stage("downloading")
        sw.write("downloading", "ดาวน์โหลดเสร็จแล้ว", job_status.stage_percent("downloading", 1.0))
        _run_transcribe_stage(job_dir, audio_path, sw)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
        _fail(sw, exc)


def run_transcribe_from_file(job_dir: Path, file_path: Path, source_label: str) -> None:
    sw = StatusWriter(job_dir)
    try:
        _run_transcribe_stage(job_dir, file_path, sw)
    except Exception as exc:  # noqa: BLE001
        _fail(sw, exc)


def _run_transcribe_stage(job_dir: Path, audio_or_video_path: Path, sw: StatusWriter) -> None:
    sw.start_stage("transcribing")
    sw.write("transcribing", "กำลังแปลงไฟล์เสียงเป็น WAV mono 16kHz...", job_status.stage_percent("transcribing", 0.05))
    wav_path = transcribe.convert_to_wav(audio_or_video_path, job_dir)
    sw.write("transcribing", "กำลังถอดเสียงภาษาไทยในเครื่อง (Whisper large-v3)...", job_status.stage_percent("transcribing", 0.15))

    def on_transcribe_progress(fraction: float) -> None:
        sw.write(
            "transcribing",
            "กำลังถอดเสียงภาษาไทยในเครื่อง (Whisper large-v3)...",
            job_status.stage_percent("transcribing", 0.15 + 0.85 * fraction),
        )

    # This is the only call in the whole app that touches mlx_whisper, and it
    # runs on this worker process's own main thread.
    result = transcribe.transcribe_thai(wav_path, progress_callback=on_transcribe_progress)
    sw.end_stage("transcribing")

    # Release the ASR model and MLX/unified memory before this process exits,
    # so nothing from the transcription step lingers once summarization
    # (a separate process entirely) starts.
    transcribe.release_model()

    (job_dir / config.TRANSCRIPT_FILENAME).write_text(result.text, encoding="utf-8")
    (job_dir / config.TRANSCRIPT_TIMESTAMPS_FILENAME).write_text(
        transcribe.build_timestamped_transcript(result.segments), encoding="utf-8"
    )
    (job_dir / config.TRANSCRIPT_SEGMENTS_FILENAME).write_text(
        json.dumps(result.segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sw.write(
        "transcribed",
        "ถอดเสียงเสร็จแล้ว กรุณาตรวจสอบ Transcript ก่อนสร้างสรุป",
        job_status.stage_percent("transcribing", 1.0),
    )


def run_summarize(job_dir: Path, source_label: str) -> None:
    """Stage 2, launched only after the user reviews the transcript and asks
    for a summary. Runs as its own fresh process — never imports mlx_whisper."""
    existing = job_status.read_status(job_dir)
    sw = StatusWriter(job_dir, base_timings=existing.get("timings"))
    transcript_path = job_dir / config.TRANSCRIPT_FILENAME
    try:
        if not transcript_path.exists():
            raise summarize.SummarizationError("ไม่พบ Transcript กรุณาถอดเสียงก่อน")
        transcript_text = transcript_path.read_text(encoding="utf-8")

        sw.start_stage("summarizing")
        sw.write("summarizing", "กำลังสรุปการประชุมด้วย Local LLM (qwen3:4b)...", job_status.stage_percent("summarizing", 0.0))

        def on_summarize_progress(fraction: float) -> None:
            sw.write(
                "summarizing",
                "กำลังสรุปการประชุมด้วย Local LLM (qwen3:4b)...",
                job_status.stage_percent("summarizing", fraction),
            )

        summary = summarize.summarize_transcript(transcript_text, progress_callback=on_summarize_progress)
        sw.end_stage("summarizing")

        (job_dir / config.SUMMARY_JSON_FILENAME).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary_md = summarize.summary_to_markdown(summary, source_label)
        (job_dir / config.SUMMARY_MD_FILENAME).write_text(summary_md, encoding="utf-8")

        sw.write("completed", "เสร็จแล้ว: ดาวน์โหลดผลลัพธ์ได้ด้านล่าง", 100)
    except Exception as exc:  # noqa: BLE001
        _fail(sw, exc)


def _fail(sw: StatusWriter, exc: Exception) -> None:
    traceback.print_exc()  # goes to the worker.log file the server redirects stdout/stderr to
    sw.write("failed", f"เกิดข้อผิดพลาด: {exc}", 0, error=str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Local meeting-summarizer worker process")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--stage", choices=["transcribe", "summarize"], required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file")
    group.add_argument("--youtube")
    args = parser.parse_args()

    job_dir = Path(args.job_dir)

    if args.stage == "summarize":
        run_summarize(job_dir, args.source_label)
        return

    if args.file:
        run_transcribe_from_file(job_dir, Path(args.file), args.source_label)
    elif args.youtube:
        run_transcribe_from_youtube(job_dir, args.youtube, args.source_label)
    else:
        parser.error("--stage transcribe requires --file or --youtube")


if __name__ == "__main__":
    sys.exit(main() or 0)
