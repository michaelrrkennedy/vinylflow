"""Inbox — ingest recordings dropped into a watched folder.

A remote recorder (e.g. a Raspberry Pi next to the turntable) copies a
full-resolution recording into ``VINYLFLOW_INBOX_DIR``; VinylFlow picks it
up as a Session without a browser upload.

Drop protocol:
  1. Optionally copy a sidecar ``<name>.json`` first.
  2. Copy the audio ``<name>.wav`` (or .aif/.aiff).  ``rsync`` writes to a
     hidden temp file and renames on completion, so any *visible* audio
     file is treated as complete.  Other tools should do the same
     (write ``.<name>.part`` then rename).

Sidecar keys (all optional)::

    {
      "discogs_release_id": 249504,      # pre-selects the Discogs release
      "artist": "...", "title": "...",   # shown in the queue / search box
      "callback_url": "http://pi:5001/api/vinylflow/complete",
      ...                                 # anything else is passed back
    }

Inbox Sessions read the source in place — nothing is copied.  Once a
Session completes, the source and sidecar move to ``processed_dir`` and the
optional callback receives a JSON POST describing the output.  Removing an
inbox Session from the queue moves its files to ``processed_dir/skipped``
so it is not re-ingested.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Iterable, Optional

import requests

logger = logging.getLogger(__name__)


class Inbox:
    def __init__(self, inbox_dir: Path, processed_dir: Optional[Path] = None,
                 extensions: Iterable[str] = (".wav", ".aif", ".aiff")) -> None:
        self.inbox_dir = Path(inbox_dir)
        self.processed_dir = Path(processed_dir) if processed_dir else self.inbox_dir / "processed"
        self.extensions = {e.lower() for e in extensions}
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def pending(self) -> list[Path]:
        """Complete audio files waiting in the inbox, oldest first."""
        files = [
            p for p in self.inbox_dir.iterdir()
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix.lower() in self.extensions
        ]
        return sorted(files, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def sidecar_path(audio: Path) -> Path:
        return audio.with_suffix(".json")

    def read_sidecar(self, audio: Path) -> dict:
        path = self.sidecar_path(audio)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.warning("Ignoring unreadable sidecar %s: %s", path, e)
            return {}
        if not isinstance(data, dict):
            return {}
        if "discogs_release_id" in data:
            try:
                data["discogs_release_id"] = int(data["discogs_release_id"])
            except (TypeError, ValueError):
                data.pop("discogs_release_id")
        return data

    def archive(self, audio: Path, subdir: str = "") -> Path:
        """Move ``audio`` and its sidecar out of the inbox.  Returns the new
        audio path.  Never overwrites: clashing names get a numeric suffix."""
        dest_dir = self.processed_dir / subdir if subdir else self.processed_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = _unique(dest_dir / audio.name)
        shutil.move(str(audio), str(dest))
        sidecar = self.sidecar_path(audio)
        if sidecar.exists():
            shutil.move(str(sidecar), str(dest.with_suffix(".json")))
        return dest


def _unique(path: Path) -> Path:
    if not path.exists():
        return path
    n = 1
    while True:
        candidate = path.with_name(f"{path.stem}.{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def notify(callback_url: str, payload: dict, timeout: float = 10.0) -> bool:
    """POST ``payload`` to ``callback_url``.  Failures are logged, not raised —
    the recorder being offline must never fail a finished Pipeline run."""
    try:
        r = requests.post(callback_url, json=payload, timeout=timeout)
        r.raise_for_status()
        return True
    except requests.RequestException as e:
        logger.warning("Inbox callback to %s failed: %s", callback_url, e)
        return False
