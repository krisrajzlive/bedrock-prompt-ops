"""Prompt lifecycle on LangSmith: push, resolve by tag/hash, promote, roll back, diff.

Environments are LangSmith *commit tags* (dev / staging / prod). Application code resolves
``name:prod`` and never a hard-coded template, so a promotion or rollback is a tag move, not a deploy.
"""
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from typing import Any

from langchain_core.load import loads
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from .cache import PromptCache

ENVIRONMENTS = ("dev", "staging", "prod")

# Manifests are deserialised with an explicit allowlist: prompt classes only, never models or tools.
_PROMPT_CLASSES = [
    ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate,
    HumanMessagePromptTemplate, AIMessagePromptTemplate, MessagesPlaceholder,
]


class PromptNotFound(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedPrompt:
    name: str
    ref: str
    commit_hash: str
    prompt: ChatPromptTemplate
    source: str  # "langsmith" or "cache"

    @property
    def identifier(self) -> str:
        return f"{self.name}:{self.commit_hash[:8]}"


@dataclass(frozen=True)
class CommitInfo:
    commit_id: str
    commit_hash: str
    parent_hash: str | None
    created_at: str | None
    description: str | None


def template_text(prompt: ChatPromptTemplate) -> str:
    """Human-readable form of a chat prompt, used for diffs."""
    lines: list[str] = []
    for msg in prompt.messages:
        role = type(msg).__name__.replace("MessagePromptTemplate", "").lower() or "message"
        inner = getattr(msg, "prompt", None)
        body = getattr(inner, "template", None) or str(msg)
        lines.append(f"[{role}]")
        lines.extend(body.splitlines())
    return "\n".join(lines)


class PromptRegistry:
    def __init__(self, client: Any, cache: PromptCache | None = None):
        self.client = client
        self.cache = cache

    # ---- write side -------------------------------------------------------------------------
    def push(
        self,
        name: str,
        prompt: ChatPromptTemplate,
        *,
        description: str | None = None,
        commit_tags: list[str] | None = None,
        commit_description: str | None = None,
    ) -> str:
        return self.client.push_prompt(
            name,
            object=prompt,
            description=description,
            commit_tags=commit_tags or None,
            commit_description=commit_description,
        )

    def promote(self, name: str, tag: str, source_ref: str) -> CommitInfo:
        """Point ``tag`` at the commit that ``source_ref`` currently resolves to."""
        resolved = self.resolve(name, source_ref, use_cache=False)
        commit = self._commit_by_hash(name, resolved.commit_hash)
        self._point_tag(name, tag, commit.commit_id)
        return commit

    def rollback(self, name: str, tag: str) -> CommitInfo:
        """Move ``tag`` to the parent of the commit it currently points at."""
        current = self.resolve(name, tag, use_cache=False)
        commit = self._commit_by_hash(name, current.commit_hash)
        if not commit.parent_hash:
            raise PromptNotFound(f"{name}:{tag} points at the first commit; nothing to roll back to")
        parent = self._commit_by_hash(name, commit.parent_hash)
        self._point_tag(name, tag, parent.commit_id)
        return parent

    def _point_tag(self, name: str, tag: str, commit_id: str) -> None:
        """Create the tag, or move it if it already exists."""
        try:
            self.client._create_commit_tags(f"-/{name}", commit_id, tag)
            return
        except Exception as create_err:
            if "409" not in str(create_err) and "exist" not in str(create_err).lower():
                raise
        response = self.client.request_with_retries(
            "PATCH", f"/repos/-/{name}/tags/{tag}", json={"commit_id": commit_id}
        )
        if getattr(response, "status_code", 200) >= 400:
            # Older deployments have no PATCH route: delete then recreate.
            self.client.request_with_retries("DELETE", f"/repos/-/{name}/tags/{tag}")
            self.client._create_commit_tags(f"-/{name}", commit_id, tag)

    # ---- read side --------------------------------------------------------------------------
    def history(self, name: str, limit: int = 20) -> list[CommitInfo]:
        return [
            CommitInfo(
                commit_id=str(c.id),
                commit_hash=c.commit_hash or "",
                parent_hash=c.parent_commit_hash,
                created_at=c.created_at.isoformat() if c.created_at else None,
                description=c.description,
            )
            for c in self.client.list_prompt_commits(name, limit=limit)
        ]

    def _commit_by_hash(self, name: str, commit_hash: str) -> CommitInfo:
        for commit in self.history(name, limit=100):
            if commit.commit_hash == commit_hash:
                return commit
        raise PromptNotFound(f"commit {commit_hash[:8]} not found in the last 100 commits of {name}")

    def tag_map(self, name: str, tags: tuple[str, ...] = ENVIRONMENTS) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for tag in tags:
            try:
                out[tag] = self.resolve(name, tag, use_cache=False).commit_hash
            except Exception:
                out[tag] = None
        return out

    def resolve(self, name: str, ref: str = "prod", *, use_cache: bool = True) -> ResolvedPrompt:
        """Fetch ``name:ref`` (tag or hash). Falls back to the cache if LangSmith fails."""
        try:
            commit = self.client.pull_prompt_commit(f"{name}:{ref}", skip_cache=True)
            manifest = commit.manifest
            entry = {"commit_hash": commit.commit_hash, "manifest": manifest}
            if self.cache:
                self.cache.put(name, ref, entry)
                self.cache.put(name, commit.commit_hash, entry)
            return self._build(name, ref, entry, "langsmith")
        except Exception as exc:
            cached = self.cache.get(name, ref) if (self.cache and use_cache) else None
            if cached is None:
                raise PromptNotFound(f"could not resolve {name}:{ref}: {exc}") from exc
            return self._build(name, ref, cached, "cache")

    @staticmethod
    def _build(name: str, ref: str, entry: dict[str, Any], source: str) -> ResolvedPrompt:
        prompt = loads(json.dumps(entry["manifest"]), allowed_objects=_PROMPT_CLASSES)
        return ResolvedPrompt(name, ref, entry["commit_hash"], prompt, source)

    def diff(self, name: str, ref_a: str, ref_b: str) -> str:
        a = self.resolve(name, ref_a, use_cache=False)
        b = self.resolve(name, ref_b, use_cache=False)
        return "\n".join(
            difflib.unified_diff(
                template_text(a.prompt).splitlines(),
                template_text(b.prompt).splitlines(),
                fromfile=f"{ref_a} ({a.commit_hash[:8]})",
                tofile=f"{ref_b} ({b.commit_hash[:8]})",
                lineterm="",
            )
        )
