"""Tests for the watched inbox (backend/inbox.py + its api.py wiring)."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.api as api
from backend.inbox import Inbox
from backend.sessions import ProcessingRun, SessionState


def _drop(inbox_dir: Path, name: str, sidecar: dict | None = None) -> Path:
    if sidecar is not None:
        (inbox_dir / name).with_suffix(".json").write_text(json.dumps(sidecar))
    audio = inbox_dir / name
    audio.write_bytes(b"RIFF....WAVEfake")
    return audio


# --- Inbox -----------------------------------------------------------------

def test_pending_skips_hidden_partial_and_non_audio(tmp_path):
    inbox = Inbox(tmp_path / "inbox")
    _drop(inbox.inbox_dir, "side-a.wav")
    _drop(inbox.inbox_dir, ".side-b.wav.Xy12ab")  # rsync temp file
    (inbox.inbox_dir / "notes.txt").write_text("x")
    _drop(inbox.inbox_dir, "side-c.AIFF")

    assert sorted(p.name for p in inbox.pending()) == ["side-a.wav", "side-c.AIFF"]


def test_read_sidecar_coerces_release_id_and_tolerates_garbage(tmp_path):
    inbox = Inbox(tmp_path / "inbox")
    a = _drop(inbox.inbox_dir, "a.wav", {"discogs_release_id": "249504", "artist": "X"})
    b = _drop(inbox.inbox_dir, "b.wav", {"discogs_release_id": "nope"})
    c = _drop(inbox.inbox_dir, "c.wav")
    (inbox.inbox_dir / "d.json").write_text("{not json")
    d = _drop(inbox.inbox_dir, "d.wav")

    assert inbox.read_sidecar(a) == {"discogs_release_id": 249504, "artist": "X"}
    assert inbox.read_sidecar(b) == {}
    assert inbox.read_sidecar(c) == {}
    assert inbox.read_sidecar(d) == {}


def test_archive_moves_sidecar_and_never_overwrites(tmp_path):
    inbox = Inbox(tmp_path / "inbox", tmp_path / "done")
    first = inbox.archive(_drop(inbox.inbox_dir, "a.wav", {"x": 1}))
    second = inbox.archive(_drop(inbox.inbox_dir, "a.wav", {"x": 2}))

    assert first == tmp_path / "done" / "a.wav"
    assert second == tmp_path / "done" / "a.1.wav"
    assert json.loads(second.with_suffix(".json").read_text()) == {"x": 2}
    assert list(inbox.inbox_dir.iterdir()) == []


# --- api wiring ------------------------------------------------------------

@pytest.fixture
def api_inbox(tmp_path, monkeypatch):
    inbox = Inbox(tmp_path / "inbox", tmp_path / "processed")
    monkeypatch.setattr(api, "inbox", inbox)
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api.audio_processor, "get_audio_duration", lambda p: 600.0)

    async def no_preconvert(file_id, path):
        return None

    monkeypatch.setattr(api, "preconvert_to_mp3", no_preconvert)
    yield inbox
    for s in api.session_store.list():
        api.session_store.remove(s.id)


def test_ingest_creates_session_once_with_sidecar_meta(api_inbox):
    audio = _drop(api_inbox.inbox_dir, "rec.wav",
                  {"discogs_release_id": 42, "artist": "A", "title": "T"})

    assert asyncio.run(api.ingest_inbox_once()) == 1
    assert asyncio.run(api.ingest_inbox_once()) == 0  # idempotent

    (session,) = api.session_store.list()
    assert session.origin == "inbox"
    assert session.source_audio == audio
    assert session.source_duration == 600.0

    queue = TestClient(api.app).get("/api/queue").json()["uploaded"]
    assert queue[0]["discogs_release_id"] == 42
    assert queue[0]["artist"] == "A" and queue[0]["origin"] == "inbox"


def test_reaped_session_is_reingested(api_inbox):
    _drop(api_inbox.inbox_dir, "rec.wav")
    asyncio.run(api.ingest_inbox_once())
    (session,) = api.session_store.list()
    api.session_store.remove(session.id)

    assert asyncio.run(api.ingest_inbox_once()) == 1


def test_finalize_archives_source_and_calls_back(api_inbox, monkeypatch):
    _drop(api_inbox.inbox_dir, "rec.wav",
          {"discogs_release_id": 42, "callback_url": "http://pi/cb", "recording": "rec.wav"})
    asyncio.run(api.ingest_inbox_once())
    (session,) = api.session_store.list()
    session.link_release({"id": 42})
    session.set_boundaries([{"number": 1}])
    session.mark_processing(ProcessingRun(started_at=datetime.now()))
    session.mark_complete(Path("/out/A - T"), ["A1-x.flac"])

    calls = []
    monkeypatch.setattr(api, "notify_inbox_callback", lambda url, payload: calls.append((url, payload)))
    api.finalize_inbox_session(session)

    assert session.state == SessionState.COMPLETE
    assert session.source_audio == api_inbox.processed_dir / "rec.wav"
    assert session.source_audio.exists()
    assert (api_inbox.processed_dir / "rec.json").exists()
    assert api_inbox.pending() == []

    (url, payload), = calls
    assert url == "http://pi/cb"
    assert payload["status"] == "complete"
    assert payload["recording"] == "rec.wav"
    assert payload["discogs_release_id"] == 42
    assert payload["tracks"] == ["A1-x.flac"]
    assert "callback_url" not in payload


def test_delete_inbox_session_moves_source_to_skipped(api_inbox):
    _drop(api_inbox.inbox_dir, "rec.wav")
    asyncio.run(api.ingest_inbox_once())
    (session,) = api.session_store.list()

    resp = TestClient(api.app).delete(f"/api/queue/{session.id}")

    assert resp.status_code == 200
    assert (api_inbox.processed_dir / "skipped" / "rec.wav").exists()
    assert asyncio.run(api.ingest_inbox_once()) == 0
