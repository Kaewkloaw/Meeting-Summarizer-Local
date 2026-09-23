"""Download the source audio for a YouTube URL, then stop.

This is the ONLY place in the app that makes an outbound network request.
Everything downstream (transcription, summarization) runs entirely on the
local machine. The caller is responsible for making sure the user actually
has the right to download and process the given content.
"""

from pathlib import Path
from typing import Callable, Optional


class YoutubeDownloadError(RuntimeError):
    pass


def download_audio(
    url: str,
    dest_dir: Path,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Path:
    """Download the best available audio track for `url` into `dest_dir`.

    `progress_callback`, if given, is called with a float in [0, 1] as the
    download proceeds (post-processing/audio-extraction counts as the final
    portion of that range).

    Returns the path to the downloaded audio file. Raises
    YoutubeDownloadError on any failure (invalid URL, network error,
    geo/age restriction, etc).
    """
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise YoutubeDownloadError(
            "ไม่พบไลบรารี yt-dlp กรุณารัน start_local_web.sh เพื่อติดตั้ง dependency ก่อน"
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(dest_dir / "source_audio.%(ext)s")

    def _hook(d: dict) -> None:
        if progress_callback is None:
            return
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes") or 0
            if total:
                # downloading itself is the first 90% of this stage; the
                # remaining 10% covers ffmpeg audio extraction below.
                progress_callback(0.9 * min(1.0, downloaded / total))
        elif d.get("status") == "finished":
            progress_callback(0.9)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
        "progress_hooks": [_hook],
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as exc:  # yt_dlp raises its own DownloadError subclasses
        raise YoutubeDownloadError(f"ดาวน์โหลดจาก YouTube ไม่สำเร็จ: {exc}") from exc

    if progress_callback is not None:
        progress_callback(1.0)

    candidates = sorted(dest_dir.glob("source_audio.*"))
    if not candidates:
        raise YoutubeDownloadError("ดาวน์โหลดเสร็จแต่ไม่พบไฟล์เสียงผลลัพธ์")
    return candidates[0]
