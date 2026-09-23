"""Shared status.json read/write helpers.

The FastAPI server process and the standalone worker process (app/worker.py)
communicate ONLY through this file on disk — never through in-process shared
state — because the worker runs as a separate OS process (see worker.py for
why: mlx-whisper's MLX stream is bound to the thread/process that created it,
so it must never run inside the server's threadpool/BackgroundTasks/
asyncio.to_thread).
"""

import json
from pathlib import Path
from typing import Optional

STATUS_FILENAME = "status.json"

# Overall progress is a weighted sum of per-stage progress (0-100 each stage).
STAGE_WEIGHTS = {
    "downloading": (0, 20),
    "transcribing": (20, 70),
    "summarizing": (70, 98),
}

DEFAULT_STATUS = {
    "status": "queued",
    "message": "รอเริ่มดำเนินการ",
    "percent": 0,
    "error": None,
    "timings": {},
}


def status_path(job_dir: Path) -> Path:
    return job_dir / STATUS_FILENAME


def write_status(
    job_dir: Path,
    status: str,
    message: str,
    percent: int,
    error: Optional[str] = None,
    timings: Optional[dict] = None,
) -> None:
    payload = {
        "status": status,
        "message": message,
        "percent": max(0, min(100, int(percent))),
        "error": error,
        "timings": timings or {},
    }
    target = status_path(job_dir)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)  # atomic on the same filesystem, avoids torn reads


def read_status(job_dir: Path) -> dict:
    path = status_path(job_dir)
    if not path.exists():
        return dict(DEFAULT_STATUS)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_STATUS)
    merged = dict(DEFAULT_STATUS)
    merged.update(data)
    return merged


def stage_percent(stage: str, fraction: float) -> int:
    lo, hi = STAGE_WEIGHTS[stage]
    fraction = max(0.0, min(1.0, fraction))
    return int(lo + (hi - lo) * fraction)
