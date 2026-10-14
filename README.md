# bedrock-prompt-ops

Prompt lifecycle management on **LangSmith** (versioned commits, environment tags, promotion, rollback, diff)
with prompts executed on **Amazon Bedrock**, and documents prepared by **Textract** (OCR) and **Comprehend** (PII redaction).
Application code never embeds a template: it resolves `document-intake-triage:prod`, so changing behaviour is a tag move.

```
document ──► Textract (images/PDF) ──► Comprehend PII redaction ──► prompt pulled by tag from LangSmith
                                                                      │
                          S3 audit record  ◄── JSON validation ◄── Bedrock (Converse)
                          S3/local prompt cache (fallback if LangSmith is down)
```

## Commands

| command | does |
|---|---|
| `promptops whoami` | confirms AWS and LangSmith credentials |
| `promptops bootstrap` | creates the private, encrypted S3 bucket (cache + audit) |
| `promptops seed` | publishes prompt v1 and v2; `dev`/`staging` -> v2, `prod` -> v1 |
| `promptops status` | which commit each environment tag points at, plus history |
| `promptops diff prod staging` | unified diff of two refs |
| `promptops promote prod --from staging` | move a tag to another ref's commit |
| `promptops rollback prod` | move the tag to the parent commit |
| `promptops run <file-or-dir> --ref prod --model nova-lite` | triage documents |
| `promptops compare <dir> --refs prod staging --models nova-lite gpt-oss-120b` | same documents across refs and models |

Results from a live run: [docs/live-run-results.md](docs/live-run-results.md).

## Setup

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # or .venv/bin/pip on Linux/macOS
cp .env.example .env        # fill LANGSMITH_API_KEY; AWS credentials go in a named profile:
aws configure --profile sandbox
pytest                      # 17 offline tests, no network
promptops whoami && promptops bootstrap   # then put the bucket name in .env
```

Required: `LANGSMITH_API_KEY`, AWS credentials with Bedrock, Textract, Comprehend and S3 access in `us-east-1`.
No OpenAI or Anthropic key is used. Tracing is off by default (`LANGSMITH_TRACING=false`); prompt push and pull work without it.

## AWS sandbox notes

Built to run from a developer machine against a time-limited AWS sandbox (new account id each restart: re-run `bootstrap`).
Compute is not hosted in AWS here: the sandbox allows creating Lambda functions and IAM roles are not creatable, so no
execution role with Bedrock/S3 access exists. The AWS side is therefore Bedrock, Textract, Comprehend and S3 only.

## Design notes

- Manifests from LangSmith are deserialised with an explicit allowlist of prompt classes, never models or tools.
- Redaction keeps `DATE_TIME` entities so deadlines stay visible to the triage prompt.
- The cache is keyed by tag and by commit hash, so a pinned `name:<hash>` also survives an outage.
- A model reply that is not valid JSON or leaves the enum is recorded in the result and audit log, not raised.
