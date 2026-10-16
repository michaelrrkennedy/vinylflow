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


# --- library ---------------------------------------------------------------

def test_library_serves_output_files_only(tmp_path, monkeypatch):
    monkeypatch.setattr(api.config, "default_output_dir", str(tmp_path / "music"))
    album = tmp_path / "music" / "A - T"
    album.mkdir(parents=True)
    (album / "A1-x.flac").write_bytes(b"fLaC")
    (tmp_path / "secret.flac").write_bytes(b"no")
    client = TestClient(api.app)

    assert client.get("/api/library/A - T/A1-x.flac").content == b"fLaC"
    assert client.get("/api/library/A - T/missing.flac").status_code == 404
    assert client.get("/api/library/..%2Fsecret.flac").status_code == 404


def test_queue_exposes_sidecar_side(api_inbox):
    _drop(api_inbox.inbox_dir, "rec.wav", {"discogs_release_id": 42, "side": "B"})
    asyncio.run(api.ingest_inbox_once())

    queue = TestClient(api.app).get("/api/queue").json()["uploaded"]
    assert queue[0]["side"] == "B"


def test_parse_sample_rate():
    from audio_processor import parse_sample_rate
    stderr = "  Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 48000 Hz, 2 channels, s16, 1536 kb/s"
    assert parse_sample_rate(stderr) == 48000
    assert parse_sample_rate("no audio here") is None


# --- smart rescan edges ------------------------------------------------------

def test_refine_rescan_moves_edges_out_to_strict_scan_and_pads_separators():
    from track_boundaries import Gap, Track, refine_rescan_boundaries
    # Rescan (-30 dB) trimmed a quiet 15 s intro; strict scan finds music at 4.8 s
    tracks = [Track(1, 20.4, 179.7), Track(2, 181.5, 347.9), Track(3, 351.0, 940.0)]
    strict = [Gap(0.6, 1.5), Gap(3.5, 4.8), Gap(542.4, 543.8), Gap(944.8, 947.0), Gap(950.0, None)]
    out = refine_rescan_boundaries(tracks, strict, total_duration=953.75, min_track_length=30)

    assert out[0].start == 4.8                      # intro restored
    assert out[-1].end == 944.8                     # fade-out restored
    assert round(out[0].end, 2) == 180.6 and round(out[1].start, 2) == 180.6   # split mid-gap (1.8 s gap)
    assert out[1].end == 348.9 and out[2].start == 350.0                        # capped at 1 s
    assert all(t.duration == t.end - t.start for t in out)


def test_refine_rescan_never_moves_edges_inward():
    from track_boundaries import Gap, Track, refine_rescan_boundaries
    tracks = [Track(1, 2.0, 100.0)]
    strict = [Gap(1.0, 6.0)]  # strict lead-in ends after the rescan start
    out = refine_rescan_boundaries(tracks, strict, total_duration=200, min_track_length=30)
    assert out[0].start == 2.0 and out[0].end == 100.0


# --- linear normalisation ----------------------------------------------------

def _loudnorm_stderr(i, tp):
    return f'[Parsed_loudnorm_2 @ 0x1]\n{{\n\t"input_i" : "{i}",\n\t"input_tp" : "{tp}",\n\t"input_lra" : "9.30"\n}}\n'


@pytest.mark.parametrize("i, tp, gain", [
    ("-20.69", "-3.88", 2.88),   # peak-limited: stops at -1 dBTP, never compresses
    ("-22.00", "-12.00", 8.00),  # loudness-limited: reaches -14 LUFS
    ("-inf", "-inf", None),      # silent: leave alone
])
def test_normalise_gain_is_linear_and_peak_capped(monkeypatch, i, tp, gain):
    from types import SimpleNamespace
    import audio_processor as ap_mod
    from audio_processor import AudioProcessor, Track
    monkeypatch.setattr(ap_mod, "run_ffmpeg", lambda *a, **k: SimpleNamespace(stderr=_loudnorm_stderr(i, tp)))
    result = AudioProcessor()._normalise_gain(Path("x.wav"), Track(1, 0, 100), "highpass=f=15")
    assert (round(result, 2) if result is not None else None) == gain
