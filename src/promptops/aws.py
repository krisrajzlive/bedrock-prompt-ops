"""Thin wrappers over the AWS AI services the pipeline uses: Textract, Comprehend and S3."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

TEXTRACT_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".tif", ".tiff"}
_COMPREHEND_CHUNK = 4500  # characters; the API limit is 100 KB per request, chunking keeps spans exact


def extract_text(textract: Any, data: bytes) -> str:
    """OCR an image or single-page PDF with Textract."""
    response = textract.detect_document_text(Document={"Bytes": data})
    lines = [b["Text"] for b in response.get("Blocks", []) if b.get("BlockType") == "LINE"]
    return "\n".join(lines)


@dataclass
class RedactionResult:
    text: str
    entity_counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.entity_counts.values())


# Dates carry the deadlines triage depends on; they are not identifying on their own.
KEEP_ENTITY_TYPES = frozenset({"DATE_TIME"})


def redact_pii(
    comprehend: Any,
    text: str,
    language: str = "en",
    min_score: float = 0.8,
    keep: frozenset[str] = KEEP_ENTITY_TYPES,
) -> RedactionResult:
    """Replace Comprehend PII spans with [REDACTED:TYPE] tokens."""
    counts: dict[str, int] = {}
    out: list[str] = []
    for start in range(0, len(text), _COMPREHEND_CHUNK):
        chunk = text[start : start + _COMPREHEND_CHUNK]
        entities = comprehend.detect_pii_entities(Text=chunk, LanguageCode=language)["Entities"]
        spans = sorted(
            (e for e in entities if e["Score"] >= min_score and e["Type"] not in keep),
            key=lambda e: e["BeginOffset"],
            reverse=True,
        )
        for e in spans:
            kind = e["Type"]
            chunk = chunk[: e["BeginOffset"]] + f"[REDACTED:{kind}]" + chunk[e["EndOffset"] :]
            counts[kind] = counts.get(kind, 0) + 1
        out.append(chunk)
    return RedactionResult("".join(out), counts)


class AuditLog:
    """One JSON object per processed document under audit/<date>/<id>.json."""

    def __init__(self, s3_client: Any, bucket: str):
        self.s3 = s3_client
        self.bucket = bucket

    def write(self, record: dict[str, Any]) -> str:
        now = datetime.now(timezone.utc)
        key = f"audit/{now:%Y-%m-%d}/{now:%H%M%S}-{uuid.uuid4().hex[:8]}.json"
        body = json.dumps({"recorded_at": now.isoformat(), **record}, indent=2).encode("utf-8")
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType="application/json")
        return key


def create_private_bucket(s3: Any, bucket: str, region: str) -> None:
    kwargs: dict[str, Any] = {"Bucket": bucket}
    if region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**kwargs)
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_encryption(
        Bucket=bucket,
        ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
        },
    )
