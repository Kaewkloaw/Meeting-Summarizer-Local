"""Summarize a Thai meeting transcript with a local Ollama LLM.

The only network call here is to http://127.0.0.1:11434 (Ollama running on
the same machine). No transcript content is ever sent anywhere else.
"""

import json
from pathlib import Path
from typing import Callable, Optional

import requests

from . import config

SUMMARY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "key_points": {"type": "array", "items": {"type": "string"}},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "owner": {"type": "string"},
                    "due_date": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["task", "owner", "due_date", "evidence"],
            },
        },
        "next_steps": {"type": "array", "items": {"type": "string"}},
        "requirements": {"type": "array", "items": {"type": "string"}},
        "needs_review": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "key_points",
        "decisions",
        "action_items",
        "next_steps",
        "requirements",
        "needs_review",
    ],
}

SYSTEM_PROMPT = """\
คุณคือผู้ช่วยสรุปการประชุมที่ทำงานอย่างเคร่งครัดตามกฎต่อไปนี้:
1. ใช้ข้อมูลเฉพาะที่ปรากฏใน Transcript ที่ให้มาเท่านั้น ห้ามเพิ่มข้อมูลใหม่และห้ามเดาโดยเด็ดขาด
2. ห้ามเดาชื่อคน ชื่อทีม วันที่ กำหนดส่ง ตัวเลข หรือข้อสรุปใด ๆ ที่ไม่ได้ถูกพูดถึงตรง ๆ ใน Transcript
3. หากรายการ Action Item ไม่มีผู้รับผิดชอบระบุไว้ชัดเจน ให้ใส่ owner เป็น "ไม่ระบุ"
4. หากรายการ Action Item ไม่มีกำหนดส่งระบุไว้ชัดเจน ให้ใส่ due_date เป็น "ไม่ระบุ"
5. ทุก Action Item ต้องมี evidence เป็นข้อความสั้น ๆ ที่คัดมาจาก Transcript จริงเพื่อสนับสนุนรายการนั้น
6. หากไม่มี Decision ใด ๆ ถูกพูดถึง ให้ decisions เป็น ["ไม่ระบุ"]
7. ใส่ประเด็นที่ไม่ชัดเจน ขัดแย้งกันเอง หรือควรให้มนุษย์ตรวจทานเพิ่มเติมไว้ใน needs_review
8. ตอบกลับเป็น JSON ล้วนตามสคีมาที่กำหนดเท่านั้น ห้ามมีข้อความอื่นนอก JSON
"""

USER_PROMPT_TEMPLATE = """\
นี่คือ Transcript ภาษาไทยจากการประชุม กรุณาสรุปตามกฎในระบบ:

---
{transcript}
---
"""


class SummarizationError(RuntimeError):
    pass


def _stream_chat(payload: dict, progress_callback: Optional[Callable[[float], None]], estimated_chars: int) -> str:
    url = f"{config.OLLAMA_BASE_URL}/api/chat"
    response = requests.post(
        url, json=payload, timeout=config.OLLAMA_TIMEOUT_SECONDS, stream=True
    )
    response.raise_for_status()

    content_parts: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        chunk = json.loads(raw_line)
        if chunk.get("error"):
            raise SummarizationError(f"Ollama ส่งข้อผิดพลาดกลับมา: {chunk['error']}")
        delta = (chunk.get("message") or {}).get("content", "")
        if delta:
            content_parts.append(delta)
            if progress_callback is not None:
                accumulated = sum(len(p) for p in content_parts)
                progress_callback(min(0.97, accumulated / estimated_chars))
        if chunk.get("done"):
            break

    return "".join(content_parts)


def _is_repeat_loop_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "repeat limit" in text or "token repeat" in text


def _call_once(base_payload: dict, progress_callback, estimated_chars) -> str:
    last_error: Exception | None = None
    for payload in (
        {**base_payload, "think": False},  # disable "thinking" mode if supported
        base_payload,  # fallback for older Ollama versions that reject "think"
    ):
        try:
            return _stream_chat(payload, progress_callback, estimated_chars)
        except requests.exceptions.ConnectionError as exc:
            raise SummarizationError(
                "เชื่อมต่อ Ollama ไม่สำเร็จ กรุณาตรวจสอบว่า Ollama กำลังทำงานที่ "
                f"{config.OLLAMA_BASE_URL} และมีโมเดล {config.OLLAMA_MODEL} แล้ว"
            ) from exc
        except requests.exceptions.RequestException as exc:
            last_error = exc
            continue
    raise SummarizationError(f"เรียก Ollama ไม่สำเร็จ: {last_error}")


def summarize_transcript(
    transcript: str,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> dict:
    """Call the local Ollama model (streaming) and return the parsed summary dict.

    `progress_callback`, if given, is called with an estimated fraction in
    [0, 1] as the model streams its JSON response back.

    On longer/real transcripts, structured JSON decoding at very low
    temperature can occasionally get stuck in a token repetition loop, which
    Ollama aborts itself with a "token repeat limit reached" error. We retry
    a couple of times with a higher temperature and an explicit repeat
    penalty, which reliably avoids the loop, before giving up.
    """
    # Rough heuristic for how long the JSON summary will be, purely to scale
    # the progress bar; it does not affect the actual output.
    estimated_chars = max(600, int(len(transcript) * 0.35))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(transcript=transcript)},
    ]

    # (temperature, repeat_penalty) per attempt: start conservative for
    # factual accuracy, back off only if we actually hit a repetition loop.
    attempts = [(0.15, 1.15), (0.35, 1.3), (0.5, 1.4)]

    content = ""
    last_exc: Optional[SummarizationError] = None
    for temperature, repeat_penalty in attempts:
        base_payload = {
            "model": config.OLLAMA_MODEL,
            "messages": messages,
            "stream": True,
            "format": SUMMARY_JSON_SCHEMA,
            "options": {
                "temperature": temperature,
                "repeat_penalty": repeat_penalty,
                "repeat_last_n": 256,
            },
        }
        try:
            content = _call_once(base_payload, progress_callback, estimated_chars)
            last_exc = None
            break
        except SummarizationError as exc:
            if not _is_repeat_loop_error(exc):
                raise
            last_exc = exc
            continue

    if last_exc is not None:
        raise SummarizationError(
            "Ollama สรุปไม่สำเร็จ (วนซ้ำโทเคนซ้ำ ๆ) แม้ลองปรับพารามิเตอร์แล้วหลายครั้ง "
            f"ลองอีกครั้ง หรือใช้ transcript ที่สั้นลง — รายละเอียด: {last_exc}"
        )

    content = content.strip()
    if not content:
        raise SummarizationError("Ollama ตอบกลับว่าง ไม่พบผลสรุป")

    try:
        summary = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SummarizationError(f"แปลงผลลัพธ์จาก Ollama เป็น JSON ไม่สำเร็จ: {exc}") from exc

    for key in SUMMARY_JSON_SCHEMA["required"]:
        summary.setdefault(key, [])

    if progress_callback is not None:
        progress_callback(1.0)

    return summary


def summary_to_markdown(summary: dict, source_label: str) -> str:
    def bullet_list(items: list[str]) -> str:
        items = [str(i).strip() for i in items if str(i).strip()]
        if not items:
            return "- ไม่ระบุ\n"
        return "".join(f"- {item}\n" for item in items)

    lines: list[str] = []
    lines.append("# Meeting Summary\n")
    lines.append(f"_ที่มา: {source_label}_\n")
    lines.append(
        "\n> ⚠️ สรุปนี้สร้างโดย Local LLM โดยอัตโนมัติ "
        "**ต้องมีมนุษย์ตรวจทานก่อนนำไปใช้งานจริง**\n"
    )

    lines.append("\n## ประเด็นสำคัญ\n")
    lines.append(bullet_list(summary.get("key_points", [])))

    lines.append("\n## Decisions / ข้อสรุป\n")
    lines.append(bullet_list(summary.get("decisions", [])))

    lines.append("\n## Action Items\n")
    action_items = summary.get("action_items", [])
    if not action_items:
        lines.append("- ไม่ระบุ\n")
    else:
        lines.append("| งาน | ผู้รับผิดชอบ | กำหนดส่ง | หลักฐานจาก Transcript |\n")
        lines.append("|---|---|---|---|\n")
        for item in action_items:
            task = str(item.get("task", "ไม่ระบุ")).replace("|", "\\|") or "ไม่ระบุ"
            owner = str(item.get("owner", "ไม่ระบุ")).replace("|", "\\|") or "ไม่ระบุ"
            due = str(item.get("due_date", "ไม่ระบุ")).replace("|", "\\|") or "ไม่ระบุ"
            evidence = str(item.get("evidence", "ไม่ระบุ")).replace("|", "\\|") or "ไม่ระบุ"
            lines.append(f"| {task} | {owner} | {due} | {evidence} |\n")

    lines.append("\n## Next Steps\n")
    lines.append(bullet_list(summary.get("next_steps", [])))

    lines.append("\n## Requirements\n")
    lines.append(bullet_list(summary.get("requirements", [])))

    lines.append("\n## จุดที่ต้องตรวจสอบเพิ่ม\n")
    lines.append(bullet_list(summary.get("needs_review", [])))

    lines.append(
        "\n---\n**ข้อมูลที่ต้องตรวจทานโดยมนุษย์**: สรุปนี้สร้างจากการถอดเสียงและโมเดลภาษาอัตโนมัติ "
        "ทั้งหมดทำงานในเครื่องนี้ อาจมีความคลาดเคลื่อนจากการถอดเสียงหรือการตีความ "
        "กรุณาตรวจทานความถูกต้องของชื่อ วันที่ ตัวเลข ผู้รับผิดชอบ และข้อสรุปทั้งหมดก่อนส่งต่อหรือใช้งานจริง\n"
    )

    return "".join(lines)
