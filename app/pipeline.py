"""Job creation + dispatch.

IMPORTANT: this module (which runs inside the FastAPI server process) must
NEVER import mlx_whisper or call anything that does. All transcription and
summarization work happens in a separate OS process (app/worker.py),
launched here via subprocess.Popen. This is required because mlx-whisper's
MLX GPU stream is bound to the thread/process that first uses it, and
FastAPI's BackgroundTasks / threadpool / asyncio.to_thread do not guarantee
a stable dedicated thread — using any of them raised:
    "There is no Stream(cpu, 0) in current thread"

The server and the worker process only communicate through status.json in
the job's directory (see app/job_status.py) and the files the worker writes
there when finished.
"""

import json
import re
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config, job_status

_META_FILENAME = "meta.json"
_JOB_ID_RE = re.compile(r"^[0-9a-f]{8,32}$")


@dataclass
class JobMeta:
    id: str
    source_label: str
    dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.dir = config.JOBS_DIR / self.id
        self.dir.mkdir(parents=True, exist_ok=True)


_JOBS: dict[str, JobMeta] = {}
_LOCK = threading.Lock()


def get_job(job_id: str) -> Optional[JobMeta]:
    """Look up a job, falling back to its on-disk meta.json.

    _JOBS is only an in-process cache: it is empty again after every server
    restart. Job metadata (source_label) is also written to
    jobs/<id>/meta.json at creation time so that an in-progress job (in
    particular one waiting at "transcribed" for the user to click
    "สรุปการประชุม") survives a server restart instead of turning into a
    permanent 404 for a page that still has that job id (e.g. in its URL).
    """
    with _LOCK:
        job = _JOBS.get(job_id)
    if job is not None:
        return job

    if not _JOB_ID_RE.match(job_id):  # avoid path traversal via job_id
        return None
    job_dir = config.JOBS_DIR / job_id
    meta_path = job_dir / _META_FILENAME
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        source_label = meta["source_label"]
    except (json.JSONDecodeError, OSError, KeyError):
        source_label = "(ไม่ทราบแหล่งที่มา — กู้คืนหลัง server รีสตาร์ท)"

    job = JobMeta(id=job_id, source_label=source_label)
    with _LOCK:
        _JOBS[job_id] = job
    return job


def create_job(source_label: str) -> JobMeta:
    job = JobMeta(id=uuid.uuid4().hex[:12], source_label=source_label)
    with _LOCK:
        _JOBS[job.id] = job
    (job.dir / _META_FILENAME).write_text(
        json.dumps({"source_label": source_label}, ensure_ascii=False), encoding="utf-8"
    )
    job_status.write_status(job.dir, "queued", "รอเริ่มดำเนินการ", 0)
    return job


def job_to_dict(job: JobMeta) -> dict:
    status = job_status.read_status(job.dir)
    return {
        "job_id": job.id,
        "status": status["status"],
        "message": status["message"],
        "percent": status["percent"],
        "error": status["error"],
        "timings": status.get("timings", {}),
        "source_label": job.source_label,
        "downloads": _available_downloads(job.dir),
    }


def _available_downloads(job_dir: Path) -> list[str]:
    names = [
        config.TRANSCRIPT_FILENAME,
        config.TRANSCRIPT_TIMESTAMPS_FILENAME,
        config.TRANSCRIPT_SEGMENTS_FILENAME,
        config.SUMMARY_JSON_FILENAME,
        config.SUMMARY_MD_FILENAME,
    ]
    return [n for n in names if (job_dir / n).exists()]


def start_transcription_from_file(job: JobMeta, uploaded_path: Path) -> None:
    _spawn_worker(job, extra_args=["--stage", "transcribe", "--file", str(uploaded_path)])


def start_transcription_from_youtube(job: JobMeta, url: str) -> None:
    _spawn_worker(job, extra_args=["--stage", "transcribe", "--youtube", url])


def start_summarization(job: JobMeta) -> None:
    """Stage 2: only called once the user has reviewed the transcript and
    explicitly asks for a summary (POST /api/jobs/{id}/summarize). Spawns a
    brand-new process that never imports mlx_whisper, so the ASR model from
    stage 1 is guaranteed to already be fully released (that process exited)."""
    _spawn_worker(job, extra_args=["--stage", "summarize"])


def _spawn_worker(job: JobMeta, extra_args: list[str]) -> None:
    cmd = [
        sys.executable,
        "-m",
        "app.worker",
        "--job-dir",
        str(job.dir),
        "--source-label",
        job.source_label,
        *extra_args,
    ]
    log_path = job.dir / "worker.log"
    # Append (not truncate): the summarize stage's log should not erase the
    # transcribe stage's log from the same job.
    with log_path.open("ab") as log_file:
        log_file.write(f"\n--- launching: {' '.join(extra_args)} ---\n".encode())
        log_file.flush()
        # Popen returns immediately; the worker runs fully independently in
        # its own process. We deliberately do not wait() on it here.
        subprocess.Popen(
            cmd,
            cwd=str(config.BASE_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
