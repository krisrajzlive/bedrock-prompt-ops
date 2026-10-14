"""Run the same documents across prompt refs and models and summarise the differences."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def compare_matrix(
    *,
    refs: list[str],
    models: list[str],
    docs: list[Path],
    resolve: Callable[[str], Any],
    pipeline_for: Callable[[str], Any],
) -> list[dict[str, Any]]:
    """One row per (ref, model); `agreement` is the share of docs whose category matches the first ref."""
    resolved = {ref: resolve(ref) for ref in refs}
    results: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for model in models:
        pipe = pipeline_for(model)
        for ref in refs:
            for doc in docs:
                results[(ref, model)].append(pipe.process(doc, resolved[ref]))

    rows = []
    for (ref, model), rs in results.items():
        base = results[(refs[0], model)]
        agree = sum(
            1 for r, b in zip(rs, base)
            if r.valid and b.valid and r.output.get("category") == b.output.get("category")
        )
        rows.append({
            "ref": ref,
            "prompt": resolved[ref].identifier,
            "model": model,
            "docs": len(rs),
            "valid_json": round(sum(r.valid for r in rs) / len(rs), 2),
            "agreement_with_first_ref": round(agree / len(rs), 2),
            "review_flagged": sum(bool(r.output.get("requires_human_review")) for r in rs),
            "avg_latency_s": round(sum(r.latency_s for r in rs) / len(rs), 2),
        })
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no rows)"
    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    line = lambda r: "  ".join(str(r[c]).ljust(widths[c]) for c in cols)  # noqa: E731
    head = line({c: c for c in cols})
    return "\n".join([head, "  ".join("-" * widths[c] for c in cols), *(line(r) for r in rows)])
