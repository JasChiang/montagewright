"""Keep uploaded media addressable across calls and across runs.

The File API holds a file for 48 hours. Uploading the same 74 proxies again
for the second planning call, and again for every review round after that,
costs minutes of wall time and achieves nothing -- the bytes are already
there. What changes between calls is the question, not the material.

The cache is keyed by content hash, so a re-encoded proxy is a different entry
and an unchanged one is a hit however many runs have passed. Entries are
checked against the service before use, because a cache that lies about what
is still live is worse than no cache: the call fails deep inside a paid
request rather than at the point of upload.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The service expires files at 48 hours. Treating anything past 46 as gone
# leaves room for a long call to finish rather than losing its inputs midway.
LIFETIME_SECONDS = 46 * 3600


def default_cache_path() -> Path:
    """Where the cache lives when nobody says otherwise.

    Keyed by content, so it belongs to the material rather than to a run.
    Storing it under the output directory, which is what this did first, meant
    a second cut of the same footage into a new folder re-uploaded all
    seventy-five files -- roughly eight minutes spent proving the bytes had not
    changed.
    """

    root = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ) / "jascue-auto"
    return root / "uploads.json"


def content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class UploadCache:
    """asset hash -> File API URI, with the service as the final word."""

    path: Path
    entries: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, path: Path) -> "UploadCache":
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entries = {}
        return cls(path=path, entries=entries)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.entries, indent=1), encoding="utf-8"
        )

    def _live(self, entry: dict[str, Any], client: Any) -> bool:
        if time.time() - float(entry.get("uploaded_at", 0)) > LIFETIME_SECONDS:
            return False
        try:
            remote = client.files.get(name=entry["name"])
        except Exception:
            return False
        state = getattr(remote.state, "name", str(remote.state))
        return state == "ACTIVE"

    def uri_for(
        self, path: Path, client: Any, *, mime_type: str
    ) -> tuple[str, bool]:
        """Return a live URI for this file, uploading only if there is none."""

        key = content_hash(path)
        entry = self.entries.get(key)
        if entry and self._live(entry, client):
            return entry["uri"], True

        uploaded = upload_now(path, client)

        self.entries[key] = {
            "uri": uploaded.uri,
            "name": uploaded.name,
            "mime_type": mime_type,
            "source": str(path),
            "uploaded_at": time.time(),
        }
        self.save()
        return uploaded.uri, False


def upload_now(path: Path, client: Any) -> Any:
    """Upload and wait until the file can actually be used.

    An upload comes back before the service has finished with it, and using
    the URI in that window fails with "not in an ACTIVE state". The cached
    path always waited; the uncached path in five other modules did not, so
    it worked on small files and failed on the first long one. One function,
    because it is one fact about the API.
    """

    uploaded = client.files.upload(file=str(path))
    while getattr(uploaded.state, "name", str(uploaded.state)) == "PROCESSING":
        time.sleep(2.0)
        uploaded = client.files.get(name=uploaded.name)
    state = getattr(uploaded.state, "name", str(uploaded.state))
    if state != "ACTIVE":
        raise RuntimeError(f"{Path(path).name} ended upload in state {state}")
    return uploaded
