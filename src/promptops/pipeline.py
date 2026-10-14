"""Document intake: OCR -> PII redaction -> versioned prompt -> Bedrock model -> audit record."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .aws import TEXTRACT_SUFFIXES, AuditLog, extract_text, redact_pii
from .models import message_text
from .prompts import CATEGORIES, URGENCIES
from .registry import ResolvedPrompt


@dataclass
class TriageResult:
    filename: str
    prompt_id: str
    prompt_source: str
    model: str
    valid: bool
    output: dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    error: str | None = None
    latency_s: float = 0.0
    pii_counts: dict[str, int] = field(default_factory=dict)
    audit_key: str | None = None


def parse_triage_json(raw: str) -> dict[str, Any]:
    """Extract the JSON object from a model reply (tolerates code fences and leading prose)."""
    text = re.sub(r"```(?:json)?", "", raw).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start : end + 1])


def validate_triage(obj: dict[str, Any]) -> list[str]:
    problems = []
    if obj.get("category") not in CATEGORIES:
        problems.append(f"category {obj.get('category')!r} not in enum")
    if obj.get("urgency") not in URGENCIES:
        problems.append(f"urgency {obj.get('urgency')!r} not in enum")
    if not isinstance(obj.get("summary"), str) or not obj.get("summary"):
        problems.append("summary missing")
    if not isinstance(obj.get("action_items"), list):
        problems.append("action_items not a list")
    return problems


class IntakePipeline:
    def __init__(
        self,
        *,
        llm: Any,
        model_name: str,
        textract: Any | None = None,
        comprehend: Any | None = None,
        audit: AuditLog | None = None,
        redact: bool = True,
    ):
        self.llm = llm
        self.model_name = model_name
        self.textract = textract
        self.comprehend = comprehend
        self.audit = audit
        self.redact = redact

    def load_text(self, path: Path) -> str:
        if path.suffix.lower() in TEXTRACT_SUFFIXES:
            if self.textract is None:
                raise RuntimeError("Textract client required for image/PDF input")
            return extract_text(self.textract, path.read_bytes())
        return path.read_text(encoding="utf-8")

    def process(self, path: Path, resolved: ResolvedPrompt) -> TriageResult:
        text = self.load_text(path)
        pii: dict[str, int] = {}
        if self.redact and self.comprehend is not None:
            red = redact_pii(self.comprehend, text)
            text, pii = red.text, red.entity_counts

        messages = resolved.prompt.invoke({"filename": path.name, "document": text})
        started = time.perf_counter()
        result = TriageResult(
            filename=path.name,
            prompt_id=resolved.identifier,
            prompt_source=resolved.source,
            model=self.model_name,
            valid=False,
            pii_counts=pii,
        )
        try:
            result.raw = message_text(self.llm.invoke(messages))
            result.output = parse_triage_json(result.raw)
            problems = validate_triage(result.output)
            result.valid = not problems
            result.error = "; ".join(problems) or None
        except Exception as exc:  # parse or provider failure is recorded, not raised
            result.error = f"{type(exc).__name__}: {exc}"
        result.latency_s = round(time.perf_counter() - started, 2)

        if self.audit is not None:
            result.audit_key = self.audit.write(
                {
                    "filename": path.name,
                    "prompt": resolved.identifier,
                    "prompt_ref": resolved.ref,
                    "prompt_source": resolved.source,
                    "model": self.model_name,
                    "valid": result.valid,
                    "error": result.error,
                    "pii_redacted": pii,
                    "output": result.output,
                    "latency_s": result.latency_s,
                }
            )
        return result
