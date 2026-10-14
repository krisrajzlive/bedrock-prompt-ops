from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.load import dumps
from langchain_core.messages import AIMessage

from promptops.prompts import TRIAGE_V1, TRIAGE_V2


class FakeLangSmith:
    """In-memory stand-in for the parts of langsmith.Client the registry uses."""

    def __init__(self):
        self.commits: list[SimpleNamespace] = []
        self.tags: dict[str, str] = {}  # tag -> commit_id
        self.down = False
        self.tag_conflicts = False

    def add(self, prompt, tags=()):
        parent = self.commits[-1].commit_hash if self.commits else None
        c = SimpleNamespace(
            id=uuid4(), commit_hash=uuid4().hex, parent_commit_hash=parent,
            created_at=datetime.now(timezone.utc), description=None,
            manifest=json.loads(dumps(prompt)),
        )
        self.commits.append(c)
        for t in tags:
            self.tags[t] = str(c.id)
        return c

    def _find(self, ref):
        if ref == "latest":
            return self.commits[-1]
        if ref in self.tags:
            return next(c for c in self.commits if str(c.id) == self.tags[ref])
        for c in self.commits:
            if c.commit_hash.startswith(ref):
                return c
        raise KeyError(ref)

    def pull_prompt_commit(self, identifier, *, skip_cache=False, **_):
        if self.down:
            raise ConnectionError("langsmith unreachable")
        c = self._find(identifier.split(":", 1)[1])
        return SimpleNamespace(commit_hash=c.commit_hash, manifest=c.manifest)

    def list_prompt_commits(self, identifier, limit=None, **_):
        return reversed(self.commits)

    def _create_commit_tags(self, owner_name, commit_id, tag):
        if tag in self.tags and self.tag_conflicts:
            raise RuntimeError("409 Conflict: tag already exists")
        self.tags[tag] = commit_id

    def request_with_retries(self, method, path, json=None):
        assert method == "PATCH"
        self.tags[path.rsplit("/", 1)[1]] = json["commit_id"]
        return SimpleNamespace(status_code=200)


@pytest.fixture
def ls():
    client = FakeLangSmith()
    client.add(TRIAGE_V1, tags=["v1", "dev", "staging", "prod"])
    client.add(TRIAGE_V2, tags=["v2", "dev", "staging"])
    return client


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.seen = []

    def invoke(self, messages):
        self.seen.append(messages)
        return AIMessage(content=self.reply)


GOOD_REPLY = json.dumps({
    "category": "invoice", "urgency": "high", "summary": "Overdue invoice.",
    "action_items": ["Pay invoice"], "requires_human_review": False,
})
