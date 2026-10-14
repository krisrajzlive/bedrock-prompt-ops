from __future__ import annotations

import warnings

import argparse
import json
import sys
from pathlib import Path

from .aws import AuditLog, create_private_bucket
from .cache import ChainedCache, DirCache, S3Cache
from .config import get_settings
from .models import MODEL_ALIASES, build_llm
from .pipeline import IntakePipeline
from .prompts import SEED_PROMPTS
from .registry import ENVIRONMENTS, PromptRegistry


warnings.filterwarnings("ignore", message=".*loads.*beta.*")


def _registry(settings, session=None) -> PromptRegistry:
    from langsmith import Client

    layers = [DirCache()]
    if settings.s3_bucket and session is not None:
        layers.append(S3Cache(session.client("s3"), settings.s3_bucket))
    return PromptRegistry(Client(), ChainedCache(*layers))


def _pipeline(settings, session, model: str, *, redact: bool = True) -> IntakePipeline:
    audit = AuditLog(session.client("s3"), settings.s3_bucket) if settings.s3_bucket else None
    return IntakePipeline(
        llm=build_llm(model, session),
        model_name=model,
        textract=session.client("textract"),
        comprehend=session.client("comprehend"),
        audit=audit,
        redact=redact,
    )


def _documents(target: str) -> list[Path]:
    p = Path(target)
    return sorted(x for x in p.iterdir() if x.is_file()) if p.is_dir() else [p]


def cmd_whoami(a, s):
    session = s.session()
    ident = session.client("sts").get_caller_identity()
    print(f"aws: {ident['Arn']} region={session.region_name}")
    from langsmith import Client

    n = len(list(Client().list_prompts(limit=1)))
    print(f"langsmith: ok ({n} prompt(s) visible)")


def cmd_bootstrap(a, s):
    session = s.session()
    acct = session.client("sts").get_caller_identity()["Account"]
    bucket = a.bucket or f"promptops-{acct}-{session.region_name}"
    create_private_bucket(session.client("s3"), bucket, session.region_name)
    print(f"created private bucket {bucket}\nset PROMPTOPS_S3_BUCKET={bucket} in .env")


def cmd_seed(a, s):
    reg = _registry(s, s.session())
    url = reg.push(s.prompt_name, SEED_PROMPTS["v1"], description="Document intake triage",
                   commit_tags=["v1", *ENVIRONMENTS], commit_description="v1: loose first draft")
    print(f"pushed v1 tagged {', '.join(ENVIRONMENTS)}: {url}")
    reg.push(s.prompt_name, SEED_PROMPTS["v2"], commit_tags=["v2"],
             commit_description="v2: strict JSON contract, document-as-data rule")
    # LangSmith rejects re-assigning an existing tag on push (409), so environments move explicitly.
    for env in ("dev", "staging"):
        reg.promote(s.prompt_name, env, "v2")
    print("pushed v2; dev and staging moved to v2, prod stays on v1")


def cmd_status(a, s):
    reg = _registry(s, s.session())
    for tag, h in reg.tag_map(s.prompt_name).items():
        print(f"{tag:8} -> {h[:8] if h else '(unset)'}")
    for c in reg.history(s.prompt_name, 10):
        print(f"  {c.commit_hash[:8]}  {c.created_at}  {c.description or ''}")


def cmd_promote(a, s):
    c = _registry(s, s.session()).promote(s.prompt_name, a.tag, a.source)
    print(f"{a.tag} -> {c.commit_hash[:8]} (from {a.source})")


def cmd_rollback(a, s):
    c = _registry(s, s.session()).rollback(s.prompt_name, a.tag)
    print(f"{a.tag} rolled back to {c.commit_hash[:8]}")


def cmd_diff(a, s):
    print(_registry(s, s.session()).diff(s.prompt_name, a.ref_a, a.ref_b))


def cmd_run(a, s):
    session = s.session()
    resolved = _registry(s, session).resolve(s.prompt_name, a.ref)
    pipe = _pipeline(s, session, a.model, redact=not a.no_redact)
    for doc in _documents(a.path):
        r = pipe.process(doc, resolved)
        print(json.dumps({"file": r.filename, "prompt": r.prompt_id, "src": r.prompt_source,
                          "valid": r.valid, "pii": r.pii_counts, "latency_s": r.latency_s,
                          "output": r.output, "error": r.error}, indent=2))


def cmd_compare(a, s):
    from .compare import compare_matrix, format_table

    session = s.session()
    reg = _registry(s, session)
    docs = _documents(a.path)
    rows = compare_matrix(
        refs=a.refs, models=a.models, docs=docs,
        resolve=lambda ref: reg.resolve(s.prompt_name, ref),
        pipeline_for=lambda m: _pipeline(s, session, m),
    )
    print(format_table(rows))
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("promptops", description="Prompt lifecycle on LangSmith + AWS AI")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("whoami").set_defaults(fn=cmd_whoami)
    b = sub.add_parser("bootstrap"); b.add_argument("--bucket"); b.set_defaults(fn=cmd_bootstrap)
    sub.add_parser("seed").set_defaults(fn=cmd_seed)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    pr = sub.add_parser("promote"); pr.add_argument("tag", choices=ENVIRONMENTS)
    pr.add_argument("--from", dest="source", required=True); pr.set_defaults(fn=cmd_promote)
    rb = sub.add_parser("rollback"); rb.add_argument("tag", choices=ENVIRONMENTS); rb.set_defaults(fn=cmd_rollback)
    d = sub.add_parser("diff"); d.add_argument("ref_a"); d.add_argument("ref_b"); d.set_defaults(fn=cmd_diff)
    r = sub.add_parser("run"); r.add_argument("path")
    r.add_argument("--ref", default=None); r.add_argument("--model", default=None)
    r.add_argument("--no-redact", action="store_true"); r.set_defaults(fn=cmd_run)
    c = sub.add_parser("compare"); c.add_argument("path")
    c.add_argument("--refs", nargs="+", default=["prod", "staging"])
    c.add_argument("--models", nargs="+", default=["nova-lite"])
    c.add_argument("--out"); c.set_defaults(fn=cmd_compare)
    sub.add_parser("models").set_defaults(fn=lambda a, s: print("\n".join(f"{k:16} {v}" for k, v in MODEL_ALIASES.items())))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    s = get_settings()
    if getattr(args, "ref", "x") is None:
        args.ref = s.default_ref
    if getattr(args, "model", "x") is None:
        args.model = s.default_model
    args.fn(args, s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
