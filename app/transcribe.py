"""Local, on-device Thai transcription via mlx-whisper.

Everything here runs on the machine's own Apple Silicon GPU. No audio or
transcript ever leaves the process.
"""

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import config

# Rough throughput of the full (non-turbo) mlx-whisper large-v3 model on
# Apple Silicon, used only to *estimate* progress while the (blocking,
# non-incremental) transcription call runs. It does not affect the actual
# transcription result. Full large-v3 is noticeably slower than the "turbo"
# distillation, hence the lower factor.
ASSUMED_REALTIME_FACTOR = 3.0


@dataclass
class TranscriptResult:
    text: str
    segments: list[dict] = field(default_factory=list)  # [{"start": float, "end": float, "text": str}, ...]


class TranscriptionError(RuntimeError):
    pass


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise TranscriptionError(
            "ไม่พบ ffmpeg กรุณารัน start_local_web.sh เพื่อติดตั้งก่อน"
        )


def convert_to_wav(src: Path, dest_dir: Path) -> Path:
    """Convert any supported audio/video file to 16kHz mono WAV via ffmpeg."""
    _require_ffmpeg()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "audio_16k.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not dest.exists():
        stderr = result.stderr or ""
        if "moov atom not found" in stderr:
            raise TranscriptionError(
                "ไฟล์ต้นฉบับเสียหายหรือบันทึก/ส่งออกไม่สมบูรณ์ (หา moov atom ไม่เจอ — "
                "มักเกิดกับไฟล์ MP4/M4A/MOV ที่แอปบันทึกเสียงปิดไม่สมบูรณ์ หรือไฟล์ถูกตัดตอนระหว่างคัดลอก/ส่งออก) "
                "กรุณาลองเปิดไฟล์ต้นฉบับด้วยโปรแกรมเล่นสื่อก่อนว่าเล่นได้ปกติหรือไม่ "
                "หรือส่งออก/บันทึกไฟล์ใหม่อีกครั้งแล้วลองอัปโหลดใหม่"
            )
        if "Invalid data found when processing input" in stderr:
            raise TranscriptionError(
                "ไฟล์ต้นฉบับไม่ใช่ไฟล์เสียง/วิดีโอที่ ffmpeg อ่านได้ หรือไฟล์เสียหาย "
                "กรุณาตรวจสอบว่าไฟล์เปิดเล่นได้ปกติแล้วลองอัปโหลดใหม่"
            )
        raise TranscriptionError(
            f"แปลงไฟล์เสียงด้วย ffmpeg ไม่สำเร็จ: {stderr[-800:]}"
        )
    return dest


def get_duration_seconds(path: Path) -> Optional[float]:
    """Best-effort audio duration via ffprobe; returns None if unavailable."""
    if shutil.which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return None


def transcribe_thai(
    wav_path: Path,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> TranscriptResult:
    """Run mlx-whisper locally on `wav_path` and return text + per-segment timestamps.

    mlx-whisper does not expose incremental progress, so while it runs we
    report an *estimated* progress (based on audio duration and an assumed
    processing speed) via `progress_callback`, capped below 100% until the
    call actually returns.

    IMPORTANT: only call this from a dedicated worker process (see
    app/worker.py), never from a FastAPI threadpool/BackgroundTasks/
    asyncio.to_thread — MLX's GPU command stream is bound to whichever
    thread first touches it, and calling this from an arbitrary worker
    thread raises "There is no Stream(cpu, 0) in current thread".
    """
    try:
        import mlx_whisper
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise TranscriptionError(
            "ไม่พบไลบรารี mlx-whisper กรุณารัน start_local_web.sh เพื่อติดตั้งก่อน "
            "(ต้องใช้เครื่อง Apple Silicon เท่านั้น)"
        ) from exc

    duration = get_duration_seconds(wav_path)
    stop_ticker = threading.Event()

    def _ticker() -> None:
        if progress_callback is None or not duration or duration <= 0:
            return
        estimated_total = max(3.0, duration / ASSUMED_REALTIME_FACTOR)
        started = time.monotonic()
        while not stop_ticker.is_set():
            elapsed = time.monotonic() - started
            fraction = min(0.97, elapsed / estimated_total)
            progress_callback(fraction)
            if stop_ticker.wait(0.5):
                break

    ticker_thread = threading.Thread(target=_ticker, daemon=True)
    ticker_thread.start()

    try:
        result = mlx_whisper.transcribe(
            str(wav_path),
            path_or_hf_repo=config.WHISPER_MODEL,
            language=config.WHISPER_LANGUAGE,
        )
    except Exception as exc:
        raise TranscriptionError(f"ถอดเสียงด้วย mlx-whisper ไม่สำเร็จ: {exc}") from exc
    finally:
        stop_ticker.set()
        ticker_thread.join(timeout=2)

    if progress_callback is not None:
        progress_callback(1.0)

    text = (result or {}).get("text", "").strip()
    if not text:
        raise TranscriptionError("ถอดเสียงสำเร็จแต่ไม่พบข้อความ (ไฟล์อาจไม่มีเสียงพูด)")

    segments = [
        {
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": str(s.get("text", "")).strip(),
        }
        for s in (result or {}).get("segments", [])
    ]

    return TranscriptResult(text=text, segments=segments)


def release_model() -> None:
    """Explicitly drop the cached Whisper model and free MLX/unified memory.

    Called right after transcription finishes and before any Ollama call, so
    the ASR model is fully released before the (separate) summarization step
    runs. In this app the transcribe and summarize stages already run as two
    separate OS processes (see app/worker.py --stage), so process exit alone
    guarantees this; this function makes the release explicit and immediate
    for any caller that keeps the process alive across both stages.
    """
    try:
        # NOT `import mlx_whisper.transcribe as X`: mlx_whisper/__init__.py
        # does `from .transcribe import transcribe`, which shadows the
        # `transcribe` submodule with a function on the mlx_whisper package
        # namespace. `import ... as` resolves through that shadowed
        # attribute, so it silently binds the function instead of the
        # module. importlib.import_module bypasses that and always returns
        # the actual submodule.
        import importlib

        mlx_transcribe_module = importlib.import_module("mlx_whisper.transcribe")
        mlx_transcribe_module.ModelHolder.model = None
        mlx_transcribe_module.ModelHolder.model_path = None
    except ImportError:
        pass

    try:
        import mlx.core as mx

        mx.clear_cache()
    except ImportError:
        pass

    import gc

    gc.collect()


def format_timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def build_timestamped_transcript(segments: list[dict]) -> str:
    lines = [
        f"[{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}] {s['text']}"
        for s in segments
        if s.get("text")
    ]
    return "\n".join(lines) + ("\n" if lines else "")
