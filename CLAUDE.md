# CLAUDE.md

## What this is

Flask + ffmpeg/OpenCV/Pillow media converter used by **both** Ropewalk (agents service `VIDEO_PROCESSING_URL`, back's file/OG pipelines) and the Aiphoria project. Runs on the shared prod server as container **`aiphoria_audio_converter`** (port 5000, bound to 127.0.0.1 on the host; other containers reach it via the docker network).

**Because it is shared, every route/field change must be additive and backward-compatible.**

## Deploy

Push to `master` → GitHub Action `AIPHORIA_CONVERTER_DEPLOY` SSHes to the prod server, `git pull` in `~/aiphoria/audio-converter`, `docker-compose up -d --build --force-recreate`. GitHub repo is `fedikan/aiphoria-audio-converter` (this checkout's remote `fedikan/media-file-converter` redirects there). Note: every push restarts the shared container — batch changes when possible.

## Testing

`tests/smoke.sh` — end-to-end suite (fixtures generated with local ffmpeg, `docker cp`'d into the container because the SSRF guard correctly blocks private-network URLs). URL-taking endpoints accept container-local paths ONLY with `CONVERTER_ALLOW_LOCAL_PATHS=1` (smoke container only — never in prod, it would let any caller make ffmpeg read files off the box).

`tests/test_ssrf_guard.py` — unit tests for `ssrf_guard.py` (`python -m unittest discover -s tests`, needs `requests`): private/scheme refusal, the connection pinned to the checked IP (DNS rebinding), playlist / concat / SDP detection. Downloads are refused when the bytes are a playlist or reference list, and saved only under plain media extensions, because ffmpeg follows the URLs inside those with no guard.

```bash
docker build -t converter-smoke .
docker run -d --name converter-smoke-test -e CONVERTER_ALLOW_LOCAL_PATHS=1 -p 127.0.0.1:5999:5000 converter-smoke
CONTAINER=converter-smoke-test BASE_URL=http://127.0.0.1:5999 ./tests/smoke.sh
```

`tests/test_app_upload.py` — image stack + upload cap (needs requirements.txt; markitdown is lazy-imported): Pillow's libwebp ≥ 1.3.2 and OpenCV ≥ 4.8.1 (CVE-2023-4863 — never pin Pillow/OpenCV back below that), WebP/HEIF/OpenCV round trips, `/convert-image`, and the 413 from `MAX_CONTENT_LENGTH` (`CONVERTER_MAX_CONTENT_MB`, default 512 — just above the host nginx 500M upload cap that bounds back's forwarded uploads; raise it, don't remove it, if a caller needs more).

## Key routes (agent-facing)

- `/merge-audio` — video+audio mux; `options.durationPolicy`: `trimAudio` (default), `freezeLastFrame`, `loopVideo`; `options.audioDelayMs`.
- `/mix-audio` — overlay/concat 2-10 audio tracks (per-track volume/offsetMs).
- `/concat-n` — URL-based N-video concat (normalize → concat demuxer). The older pairwise multipart `/concat` remains.
- `/collage`, `/video-to-gif`, `/extract-first-frame` (`timestampSec` optional), `/extract-last-frame`.
- `/transform-video`, `/transform-audio` — re-encode/downscale/trim; `start_sec` (input-side `-ss`) skips the head of the file and `max_duration` caps what follows; transform-audio also extracts audio from video (`-vn`). Skipping into a track is `start_sec` here — `audioDelayMs` on `/merge-audio` is the opposite (it inserts leading silence).
- All caller-supplied URLs go through `ssrf_guard.safe_get` — never bypass it.
