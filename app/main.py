"""Local-first meeting summarizer web app.

Binds to 127.0.0.1 only. No Claude/Anthropic/cloud LLM is used anywhere in
this app. The only outbound network call is an explicit yt-dlp download when
the user supplies a YouTube URL; everything else (transcription with
mlx-whisper, summarization with a local Ollama model) runs on-device.
"""

import re
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import config, pipeline

app = FastAPI(title="Local Meeting Summarizer")

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

YOUTUBE_URL_RE = re.compile(
    r"^https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[\w\-]+"
)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"ollama_model": config.OLLAMA_MODEL, "whisper_model": config.WHISPER_MODEL},
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/jobs/file")
async def create_job_from_file(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"ไม่รองรับไฟล์นามสกุล {suffix or '(ไม่ทราบ)'} "
            f"(รองรับ: {', '.join(sorted(config.ALLOWED_EXTENSIONS))})",
        )

    job = pipeline.create_job(source_label=f"ไฟล์: {file.filename}")
    dest = job.dir / f"upload{suffix}"

    size = 0
    with dest.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > config.MAX_UPLOAD_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="ไฟล์ใหญ่เกินไป")
            out.write(chunk)

    pipeline.start_transcription_from_file(job, dest)
    return {"job_id": job.id}


@app.post("/api/jobs/youtube")
async def create_job_from_youtube(url: str = Form(...)):
    url = url.strip()
    if not YOUTUBE_URL_RE.match(url):
        raise HTTPException(status_code=400, detail="ลิงก์ YouTube ไม่ถูกต้อง")

    job = pipeline.create_job(source_label=f"YouTube: {url}")
    pipeline.start_transcription_from_youtube(job, url)
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    job = pipeline.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ไม่พบงานนี้")
    return JSONResponse(pipeline.job_to_dict(job))


@app.post("/api/jobs/{job_id}/summarize")
def summarize_job(job_id: str):
    """Stage 2, triggered only after the user has reviewed the transcript.

    Deliberately not run automatically after transcription: the whole point
    is to stop and let a human check meeting_transcript_th.txt /
    meeting_transcript_th_timestamps.txt before any Meeting Summary is
    generated from it.
    """
    job = pipeline.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ไม่พบงานนี้")

    current = pipeline.job_to_dict(job)
    if current["status"] not in ("transcribed", "failed"):
        raise HTTPException(
            status_code=409,
            detail=f"งานนี้ยังไม่พร้อมสรุป (สถานะปัจจุบัน: {current['status']})",
        )
    if not (job.dir / config.TRANSCRIPT_FILENAME).exists():
        raise HTTPException(status_code=409, detail="ไม่พบ Transcript กรุณาถอดเสียงก่อน")

    pipeline.start_summarization(job)
    return {"job_id": job.id}


_DOWNLOAD_NAMES = {
    "transcript": config.TRANSCRIPT_FILENAME,
    "transcript_timestamps": config.TRANSCRIPT_TIMESTAMPS_FILENAME,
    "transcript_segments": config.TRANSCRIPT_SEGMENTS_FILENAME,
    "summary_json": config.SUMMARY_JSON_FILENAME,
    "summary_md": config.SUMMARY_MD_FILENAME,
}


@app.get("/api/jobs/{job_id}/download/{kind}")
def download_output(job_id: str, kind: str):
    job = pipeline.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ไม่พบงานนี้")
    filename = _DOWNLOAD_NAMES.get(kind)
    if filename is None:
        raise HTTPException(status_code=400, detail="ประเภทไฟล์ไม่ถูกต้อง")
    path = job.dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="ไฟล์นี้ยังไม่พร้อม")
    return FileResponse(path, filename=filename)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, reload=False)
