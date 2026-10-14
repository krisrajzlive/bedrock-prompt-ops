"""Pull-through caches so a prompt that was resolved once keeps working if LangSmith is unreachable."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol


def _key(name: str, ref: str) -> str:
    safe = lambda s: re.sub(r"[^A-Za-z0-9._-]", "_", s)  # noqa: E731
    return f"prompt-cache/{safe(name)}/{safe(ref)}.json"


class PromptCache(Protocol):
    def get(self, name: str, ref: str) -> dict[str, Any] | None: ...
    def put(self, name: str, ref: str, entry: dict[str, Any]) -> None: ...


class DirCache:
    def __init__(self, root: Path | str = ".prompt-cache"):
        self.root = Path(root)

    def get(self, name: str, ref: str) -> dict[str, Any] | None:
        path = self.root / _key(name, ref)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, name: str, ref: str, entry: dict[str, Any]) -> None:
        path = self.root / _key(name, ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry), encoding="utf-8")


class S3Cache:
    def __init__(self, s3_client: Any, bucket: str):
        self.s3 = s3_client
        self.bucket = bucket

    def get(self, name: str, ref: str) -> dict[str, Any] | None:
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=_key(name, ref))
        except self.s3.exceptions.NoSuchKey:
            return None
        return json.loads(obj["Body"].read())

    def put(self, name: str, ref: str, entry: dict[str, Any]) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=_key(name, ref),
            Body=json.dumps(entry).encode("utf-8"),
            ContentType="application/json",
        )


class ChainedCache:
    """Writes go to every layer; reads return the first hit."""

    def __init__(self, *layers: PromptCache):
        self.layers = layers

    def get(self, name: str, ref: str) -> dict[str, Any] | None:
        for layer in self.layers:
            try:
                hit = layer.get(name, ref)
            except Exception:  # a broken cache layer must never break a resolve
                continue
            if hit:
                return hit
        return None

    def put(self, name: str, ref: str, entry: dict[str, Any]) -> None:
        for layer in self.layers:
            try:
                layer.put(name, ref, entry)
            except Exception:
                continue
