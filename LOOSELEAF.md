# Looseleaf fork

This fork adds a **watched inbox** so recordings made on the Looseleaf Pi
appear in VinylFlow's queue without a browser upload, with the Discogs
release already selected. Everything else tracks upstream
(`git fetch upstream && git merge upstream/main`).

## Flow

```
Pi (records 24-bit WAV) ──rsync──▶ ubuntu:~/vinylflow/inbox/
                                    ├─ <name>.json   sidecar (sent first)
                                    └─ <name>.wav    mezzanine
VinylFlow ingests → you check splits in the UI → Process
  → tracks:  ~/music/vinyl/<Artist> - <Album>/   (MUSIC_DIR)
  → source:  ~/vinylflow/processed/<name>.wav  (+ .json)
  → POST sidecar.callback_url  (Pi marks the release processed)
```

Removing an inbox item from the queue moves it to `processed/skipped/`.
Sidecar format and drop rules: see `backend/inbox.py`.

## Running on the Ubuntu box

```bash
git clone https://github.com/michaelrrkennedy/vinylflow.git ~/vinylflow
cd ~/vinylflow && git checkout looseleaf-inbox
cat > .env <<ENV
DISCOGS_USER_TOKEN=...
BIND_ADDR=$(tailscale ip -4)
UID=$(id -u)
GID=$(id -g)
ENV
mkdir -p inbox processed temp_uploads config ~/music/vinyl
docker compose -f docker-compose.looseleaf.yml up -d --build
```

Then open `http://<ubuntu tailscale name>:8000` from any device on the tailnet.

## Track previews

`GET /api/library/<Artist> - <Album>/<file>` serves finished tracks from the
output folder so the Looseleaf collection page can play them. AIFF is
transcoded to a cached MP3 (in `temp_uploads/library_previews/`) since most
browsers can't play it.
