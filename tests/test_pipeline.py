import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from conftest import GOOD_REPLY, FakeLLM
from promptops.aws import AuditLog, redact_pii
from promptops.compare import compare_matrix, format_table
from promptops.models import message_text
from promptops.pipeline import IntakePipeline, parse_triage_json, validate_triage
from promptops.registry import PromptRegistry


class FakeComprehend:
    def detect_pii_entities(self, Text, LanguageCode):
        i = Text.find("Priya")
        ents = [] if i < 0 else [{"Type": "NAME", "Score": 0.99, "BeginOffset": i, "EndOffset": i + 5}]
        low = Text.find("maybe")
        if low >= 0:
            ents.append({"Type": "ADDRESS", "Score": 0.4, "BeginOffset": low, "EndOffset": low + 5})
        return {"Entities": ents}


def test_redaction_replaces_confident_spans_only():
    r = redact_pii(FakeComprehend(), "Bill Priya maybe")
    assert r.text == "Bill [REDACTED:NAME] maybe"
    assert r.entity_counts == {"NAME": 1}


def test_parse_handles_fences_and_prose():
    assert parse_triage_json('Sure!\n```json\n{"a": 1}\n```')["a"] == 1
    with pytest.raises(ValueError):
        parse_triage_json("no json here")


def test_validate_flags_bad_enum():
    assert validate_triage(json.loads(GOOD_REPLY)) == []
    assert any("category" in p for p in validate_triage({"category": "spam"}))


def test_message_text_drops_reasoning_blocks():
    msg = type("M", (), {"content": [{"type": "reasoning_content", "reasoning_content": "x"},
                                      {"type": "text", "text": "answer"}]})()
    assert message_text(msg) == "answer"


def test_pipeline_redacts_before_model_and_records_prompt(ls, tmp_path):
    doc = tmp_path / "inv.txt"
    doc.write_text("Bill Priya for 10 USD", encoding="utf-8")
    llm = FakeLLM(GOOD_REPLY)
    resolved = PromptRegistry(ls).resolve("p", "staging")
    res = IntakePipeline(llm=llm, model_name="fake", comprehend=FakeComprehend()).process(doc, resolved)
    sent = llm.seen[0].to_string()
    assert "Priya" not in sent and "[REDACTED:NAME]" in sent
    assert res.valid and res.prompt_id.startswith("p:")


def test_invalid_reply_is_recorded_not_raised(ls, tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("hello", encoding="utf-8")
    res = IntakePipeline(llm=FakeLLM("cannot comply"), model_name="fake").process(
        doc, PromptRegistry(ls).resolve("p", "prod"))
    assert not res.valid and "no JSON" in res.error


@mock_aws
def test_audit_record_written_to_s3(ls, tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="audit-test-bucket")
    doc = tmp_path / "a.txt"
    doc.write_text("hello", encoding="utf-8")
    pipe = IntakePipeline(llm=FakeLLM(GOOD_REPLY), model_name="fake", audit=AuditLog(s3, "audit-test-bucket"))
    res = pipe.process(doc, PromptRegistry(ls).resolve("p", "prod"))
    body = json.loads(s3.get_object(Bucket="audit-test-bucket", Key=res.audit_key)["Body"].read())
    assert body["prompt_ref"] == "prod" and body["model"] == "fake" and body["valid"]


def test_compare_matrix_rows(ls, tmp_path):
    d = tmp_path / "a.txt"
    d.write_text("x", encoding="utf-8")
    reg = PromptRegistry(ls)
    rows = compare_matrix(
        refs=["prod", "staging"], models=["m"], docs=[Path(d)],
        resolve=lambda r: reg.resolve("p", r),
        pipeline_for=lambda m: IntakePipeline(llm=FakeLLM(GOOD_REPLY), model_name=m),
    )
    assert [r["ref"] for r in rows] == ["prod", "staging"]
    assert rows[1]["agreement_with_first_ref"] == 1.0
    assert "valid_json" in format_table(rows)
