const fileTabBtn = document.getElementById("tab-file-btn");
const ytTabBtn = document.getElementById("tab-yt-btn");
const fileForm = document.getElementById("file-form");
const ytForm = document.getElementById("yt-form");

const statusBox = document.getElementById("status-box");
const statusText = document.getElementById("status-text");
const statusPercent = document.getElementById("status-percent");
const statusSpinner = document.getElementById("status-spinner");
const progressFill = document.getElementById("progress-fill");
const reviewBox = document.getElementById("review-box");
const reviewDownloadsEl = document.getElementById("review-downloads");
const summarizeBtn = document.getElementById("summarize-btn");
const resultBox = document.getElementById("result-box");
const errorBox = document.getElementById("error-box");
const downloadsEl = document.getElementById("downloads");

// "transcribed" sits between transcribing and summarizing: the pipeline
// stops there and waits for the user to review the transcript before
// summarize-btn triggers stage 2.
const STEP_ORDER = ["downloading", "transcribing", "summarizing", "completed"];

let currentJobId = null;

function setCurrentJob(jobId) {
  currentJobId = jobId;
  const url = new URL(window.location.href);
  url.searchParams.set("job", jobId);
  window.history.replaceState({}, "", url);
}

fileTabBtn.addEventListener("click", () => {
  fileTabBtn.classList.add("active");
  ytTabBtn.classList.remove("active");
  fileForm.classList.remove("hidden");
  ytForm.classList.add("hidden");
});

ytTabBtn.addEventListener("click", () => {
  ytTabBtn.classList.add("active");
  fileTabBtn.classList.remove("active");
  ytForm.classList.remove("hidden");
  fileForm.classList.add("hidden");
});

function resetPanels() {
  statusBox.classList.remove("hidden");
  reviewBox.classList.add("hidden");
  resultBox.classList.add("hidden");
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
  reviewDownloadsEl.innerHTML = "";
  downloadsEl.innerHTML = "";
  statusSpinner.classList.remove("done");
  progressFill.classList.remove("done", "error");
  progressFill.style.width = "0%";
  statusPercent.textContent = "0%";
  document.querySelectorAll(".status-steps span").forEach((el) => {
    el.classList.remove("active", "complete");
  });
}

function setPercent(percent, variant) {
  const clamped = Math.max(0, Math.min(100, Math.round(percent || 0)));
  statusPercent.textContent = `${clamped}%`;
  progressFill.style.width = `${clamped}%`;
  progressFill.classList.remove("done", "error");
  if (variant) progressFill.classList.add(variant);
}

function setSubmitting(isSubmitting) {
  fileForm.querySelector("button").disabled = isSubmitting;
  ytForm.querySelector("button").disabled = isSubmitting;
}

function updateSteps(status) {
  // "transcribed" (awaiting review) visually counts as "transcribing" fully
  // complete, with summarizing/completed not yet reached.
  const effectiveStatus = status === "transcribed" ? "transcribing" : status;
  const idx = STEP_ORDER.indexOf(effectiveStatus);
  document.querySelectorAll(".status-steps span").forEach((el) => {
    const stepIdx = STEP_ORDER.indexOf(el.dataset.step);
    el.classList.remove("active", "complete");
    if (idx === -1) return;
    if (stepIdx < idx) el.classList.add("complete");
    else if (stepIdx === idx) el.classList.add(status === "transcribed" ? "complete" : "active");
  });
}

const DOWNLOAD_LABELS = {
  transcript: "Transcript (.txt)",
  transcript_timestamps: "Transcript พร้อม Timestamp (.txt)",
  transcript_segments: "Transcript ดิบ (.json)",
  summary_json: "Summary (.json)",
  summary_md: "Summary (.md)",
};

const DOWNLOAD_KIND_BY_FILENAME = {
  "meeting_transcript_th.txt": "transcript",
  "meeting_transcript_th_timestamps.txt": "transcript_timestamps",
  "meeting_transcript_th_segments.json": "transcript_segments",
  "meeting_summary.json": "summary_json",
  "meeting_summary.md": "summary_md",
};

function makeDownloadLink(jobId, kind) {
  const a = document.createElement("a");
  a.href = `/api/jobs/${jobId}/download/${kind}`;
  a.textContent = DOWNLOAD_LABELS[kind] || kind;
  return a;
}

async function pollJob(jobId) {
  try {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (!res.ok) throw new Error(`สถานะงานผิดพลาด (${res.status})`);
    const data = await res.json();

    statusText.textContent = data.message || data.status;
    updateSteps(data.status);
    setPercent(data.percent);

    if (data.status === "failed") {
      statusSpinner.classList.add("done");
      setPercent(data.percent, "error");
      errorBox.classList.remove("hidden");
      errorBox.textContent = data.error || "เกิดข้อผิดพลาดที่ไม่ทราบสาเหตุ";
      setSubmitting(false);
      return;
    }

    if (data.status === "transcribed") {
      // Don't auto-summarize: let the user download and review the
      // transcript first, then explicitly click "สรุปการประชุม". But DO keep
      // lightly re-polling in the background — summarization can also be
      // triggered from elsewhere (e.g. another tab, or a retry after this
      // page was reloaded mid-review), and this page should notice and
      // switch to the result view on its own instead of sitting frozen on
      // this screen forever waiting for a click that may never come.
      statusSpinner.classList.add("done");
      reviewBox.classList.remove("hidden");
      reviewDownloadsEl.innerHTML = "";
      (data.downloads || [])
        .map((filename) => DOWNLOAD_KIND_BY_FILENAME[filename])
        .filter((kind) => kind && kind.startsWith("transcript"))
        .forEach((kind) => reviewDownloadsEl.appendChild(makeDownloadLink(jobId, kind)));
      summarizeBtn.disabled = false;
      setSubmitting(false);
      setTimeout(() => pollJob(jobId), 5000);
      return;
    }

    if (data.status === "completed") {
      statusSpinner.classList.add("done");
      setPercent(100, "done");
      resultBox.classList.remove("hidden");
      downloadsEl.innerHTML = "";
      (data.downloads || [])
        .map((filename) => DOWNLOAD_KIND_BY_FILENAME[filename])
        .filter(Boolean)
        .forEach((kind) => downloadsEl.appendChild(makeDownloadLink(jobId, kind)));
      setSubmitting(false);
      return;
    }

    setTimeout(() => pollJob(jobId), 1000);
  } catch (err) {
    errorBox.classList.remove("hidden");
    errorBox.textContent = `เชื่อมต่อกับเซิร์ฟเวอร์ไม่สำเร็จ: ${err.message}`;
    setSubmitting(false);
  }
}

summarizeBtn.addEventListener("click", async () => {
  if (!currentJobId) {
    errorBox.classList.remove("hidden");
    errorBox.textContent =
      "ไม่พบ job ที่กำลังทำงาน (อาจเกิดจากรีโหลดหน้าแล้วข้อมูลหาย) กรุณาเริ่มถอดเสียงใหม่อีกครั้ง";
    return;
  }
  summarizeBtn.disabled = true;
  errorBox.classList.add("hidden");
  statusText.textContent = "กำลังเริ่มสรุปการประชุม...";

  try {
    const res = await fetch(`/api/jobs/${currentJobId}/summarize`, { method: "POST" });
    if (res.status === 409) {
      // The job already moved past "transcribed" (e.g. summarize was
      // triggered elsewhere while this page sat idle, or this click was a
      // stale double-submit). Don't show that as an error — just resync
      // this page to whatever the job's real current status is.
      pollJob(currentJobId);
      return;
    }
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "เริ่มสรุปไม่สำเร็จ");
    pollJob(currentJobId);
  } catch (err) {
    summarizeBtn.disabled = false;
    errorBox.classList.remove("hidden");
    errorBox.textContent = err.message;
  }
});

fileForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("file-input");
  if (!input.files.length) return;

  resetPanels();
  setSubmitting(true);
  statusText.textContent = "กำลังอัปโหลดไฟล์ไปยังเซิร์ฟเวอร์ในเครื่อง...";

  const formData = new FormData();
  formData.append("file", input.files[0]);

  try {
    const res = await fetch("/api/jobs/file", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "อัปโหลดไฟล์ไม่สำเร็จ");
    setCurrentJob(data.job_id);
    pollJob(currentJobId);
  } catch (err) {
    errorBox.classList.remove("hidden");
    errorBox.textContent = err.message;
    setSubmitting(false);
  }
});

ytForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("yt-input");
  const url = input.value.trim();
  if (!url) return;

  resetPanels();
  setSubmitting(true);
  statusText.textContent = "กำลังส่งลิงก์ YouTube...";

  const formData = new FormData();
  formData.append("url", url);

  try {
    const res = await fetch("/api/jobs/youtube", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "ส่งลิงก์ไม่สำเร็จ");
    setCurrentJob(data.job_id);
    pollJob(currentJobId);
  } catch (err) {
    errorBox.classList.remove("hidden");
    errorBox.textContent = err.message;
    setSubmitting(false);
  }
});

// Resume polling an in-progress job after a page reload (e.g. the user
// reviewed the transcript, reloaded the tab, then came back to click
// "สรุปการประชุม") instead of silently losing track of it.
(function resumeJobFromUrl() {
  const jobId = new URL(window.location.href).searchParams.get("job");
  if (!jobId) return;
  currentJobId = jobId;
  resetPanels();
  statusText.textContent = "กำลังตรวจสอบสถานะงานล่าสุด...";
  pollJob(jobId);
})();
