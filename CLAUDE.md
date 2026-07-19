# CLAUDE.md

## What this is

Flask + ffmpeg/OpenCV/Pillow media converter used by **both** Ropewalk (agents service `VIDEO_PROCESSING_URL`, back's file/OG pipelines) and the Aiphoria project. Runs on the shared prod server as container **`aiphoria_audio_converter`** (port 5000, bound to 127.0.0.1 on the host; other containers reach it via the docker network).

**Because it is shared, every route/field change must be additive and backward-compatible.**

## Deploy

Push to `master` → GitHub Action `AIPHORIA_CONVERTER_DEPLOY` SSHes to the prod server, `git pull` in `~/aiphoria/audio-converter`, `docker-compose up -d --build --force-recreate`. GitHub repo is `fedikan/aiphoria-audio-converter` (this checkout's remote `fedikan/media-file-converter` redirects there). Note: every push restarts the shared container — batch changes when possible.

## Testing

`tests/smoke.sh` — end-to-end suite (fixtures generated with local ffmpeg, `docker cp`'d into the container because the SSRF guard correctly blocks private-network URLs; URL-taking endpoints accept container-local paths).

```bash
docker build -t converter-smoke .
docker run -d --name converter-smoke-test -p 127.0.0.1:5999:5000 converter-smoke
CONTAINER=converter-smoke-test BASE_URL=http://127.0.0.1:5999 ./tests/smoke.sh
```

## Key routes (agent-facing)

- `/merge-audio` — video+audio mux; `options.durationPolicy`: `trimAudio` (default), `freezeLastFrame`, `loopVideo`; `options.audioDelayMs`.
- `/mix-audio` — overlay/concat 2-10 audio tracks (per-track volume/offsetMs).
- `/concat-n` — URL-based N-video concat (normalize → concat demuxer). The older pairwise multipart `/concat` remains.
- `/collage`, `/video-to-gif`, `/extract-first-frame` (`timestampSec` optional), `/extract-last-frame`.
- `/transform-video`, `/transform-audio` — re-encode/downscale/trim; transform-audio also extracts audio from video (`-vn`).
- All caller-supplied URLs go through `ssrf_guard.safe_get` — never bypass it.
