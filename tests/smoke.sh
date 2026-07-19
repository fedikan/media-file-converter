#!/usr/bin/env bash
# Smoke test for the media converter. Generates fixtures with local ffmpeg,
# copies them INTO the container (the SSRF guard rightly blocks private-network
# URLs, but the download helper accepts container-local paths), then curls each
# endpoint and validates output durations with ffprobe.
# Usage:
#   CONTAINER=<container-name> BASE_URL=http://127.0.0.1:5000 ./tests/smoke.sh
set -uo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:5000}"
CONTAINER="${CONTAINER:-aiphoria_audio_converter}"
WORK="$(mktemp -d)"
FIX="/tmp/smoke-fixtures"
PASS=0; FAIL=0
trap 'rm -rf "$WORK"' EXIT

need() { command -v "$1" >/dev/null || { echo "missing dependency: $1"; exit 1; }; }
need ffmpeg; need ffprobe; need curl; need docker; need python3

echo "== generating fixtures in $WORK"
ffmpeg -y -loglevel error -f lavfi -i "testsrc=duration=5:size=320x240:rate=24" \
  -pix_fmt yuv420p "$WORK/video5s.mp4"
ffmpeg -y -loglevel error -f lavfi -i "sine=frequency=440:duration=10" "$WORK/audio10s.mp3"
ffmpeg -y -loglevel error -f lavfi -i "sine=frequency=880:duration=3" "$WORK/audio3s.mp3"

echo "== copying fixtures into container $CONTAINER:$FIX"
docker exec "$CONTAINER" mkdir -p "$FIX"
docker cp "$WORK/video5s.mp4" "$CONTAINER:$FIX/"
docker cp "$WORK/audio10s.mp3" "$CONTAINER:$FIX/"
docker cp "$WORK/audio3s.mp3" "$CONTAINER:$FIX/"

dur() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1" 2>/dev/null; }
fsize() { stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null || echo 0; }
check_close() { # name actual expected tolerance
  local a e t; a="${2:-0}"; e="$3"; t="${4:-1.0}"
  if python3 -c "import sys; sys.exit(0 if abs(float('$a' or 0)-$e)<=$t else 1)" 2>/dev/null; then
    echo "PASS $1 (duration $a ≈ $e)"; PASS=$((PASS+1))
  else
    echo "FAIL $1 (duration '$a', expected ≈ $e)"; FAIL=$((FAIL+1))
  fi
}
check_ok() { # name file
  if [ -s "$2" ] && [ "$(fsize "$2")" -gt 1000 ]; then
    echo "PASS $1 ($(fsize "$2") bytes)"; PASS=$((PASS+1))
  else
    echo "FAIL $1 (missing/too small; body: $(head -c 300 "$2" 2>/dev/null))"; FAIL=$((FAIL+1))
  fi
}

merge() { # policy expected_duration
  local out="$WORK/merged_$1.mp4"
  curl -sS -X POST "$BASE_URL/merge-audio" -H 'Content-Type: application/json' \
    -d "{\"videoUrl\":\"$FIX/video5s.mp4\",\"audioUrl\":\"$FIX/audio10s.mp3\",\"options\":{\"durationPolicy\":\"$1\"}}" \
    -o "$out"
  check_ok "merge-audio [$1]" "$out"
  check_close "merge-audio [$1] duration" "$(dur "$out")" "$2"
}

echo "== /merge-audio duration policies (video 5s + audio 10s)"
merge trimAudio 5
merge freezeLastFrame 10
merge loopVideo 10

echo "== /merge-audio default policy (no option) -> trimAudio"
curl -sS -X POST "$BASE_URL/merge-audio" -H 'Content-Type: application/json' \
  -d "{\"videoUrl\":\"$FIX/video5s.mp4\",\"audioUrl\":\"$FIX/audio10s.mp3\",\"options\":{}}" \
  -o "$WORK/merged_default.mp4"
check_ok "merge-audio [default]" "$WORK/merged_default.mp4"
check_close "merge-audio [default] duration" "$(dur "$WORK/merged_default.mp4")" 5

echo "== /mix-audio mix mode (10s + 3s offset 2s -> 10s)"
curl -sS -X POST "$BASE_URL/mix-audio" -H 'Content-Type: application/json' \
  -d "{\"tracks\":[{\"url\":\"$FIX/audio10s.mp3\",\"volume\":1.0},{\"url\":\"$FIX/audio3s.mp3\",\"volume\":0.5,\"offsetMs\":2000}],\"mode\":\"mix\"}" \
  -o "$WORK/mixed.mp3"
check_ok "mix-audio [mix]" "$WORK/mixed.mp3"
check_close "mix-audio [mix] duration" "$(dur "$WORK/mixed.mp3")" 10

echo "== /mix-audio concat mode (10s + 3s -> 13s)"
curl -sS -X POST "$BASE_URL/mix-audio" -H 'Content-Type: application/json' \
  -d "{\"tracks\":[{\"url\":\"$FIX/audio10s.mp3\"},{\"url\":\"$FIX/audio3s.mp3\"}],\"mode\":\"concat\"}" \
  -o "$WORK/concat.mp3"
check_ok "mix-audio [concat]" "$WORK/concat.mp3"
check_close "mix-audio [concat] duration" "$(dur "$WORK/concat.mp3")" 13

echo "== /concat-n (3 clips of 5s -> 15s)"
curl -sS -X POST "$BASE_URL/concat-n" -H 'Content-Type: application/json' \
  -d "{\"videoUrls\":[\"$FIX/video5s.mp4\",\"$FIX/video5s.mp4\",\"$FIX/video5s.mp4\"]}" \
  -o "$WORK/concat3.mp4"
check_ok "concat-n [3 clips]" "$WORK/concat3.mp4"
check_close "concat-n [3 clips] duration" "$(dur "$WORK/concat3.mp4")" 15

echo "== /collage (grid of 4)"
ffmpeg -y -loglevel error -f lavfi -i "testsrc=duration=0.1:size=400x300:rate=1" -frames:v 1 "$WORK/img.png"
docker cp "$WORK/img.png" "$CONTAINER:$FIX/"
curl -sS -X POST "$BASE_URL/collage" -H 'Content-Type: application/json' \
  -d "{\"images\":[\"$FIX/img.png\",\"$FIX/img.png\",\"$FIX/img.png\",\"$FIX/img.png\"],\"layout\":\"grid\"}" \
  -o "$WORK/collage.webp"
check_ok "collage [grid 4]" "$WORK/collage.webp"

echo "== /video-to-gif (2s segment)"
curl -sS -X POST "$BASE_URL/video-to-gif" -H 'Content-Type: application/json' \
  -d "{\"videoUrl\":\"$FIX/video5s.mp4\",\"fps\":10,\"maxWidth\":240,\"startSec\":1,\"durationSec\":2}" \
  -o "$WORK/out.gif"
check_ok "video-to-gif" "$WORK/out.gif"
file "$WORK/out.gif" 2>/dev/null | grep -qi gif && { echo "PASS gif magic"; PASS=$((PASS+1)); } \
  || { echo "FAIL gif magic"; FAIL=$((FAIL+1)); }

echo "== /extract-first-frame with timestampSec (multipart)"
curl -sS -X POST "$BASE_URL/extract-first-frame" \
  -F "file=@$WORK/video5s.mp4" -F "timestampSec=3" -o "$WORK/frame3s.webp"
check_ok "extract-first-frame [t=3s]" "$WORK/frame3s.webp"

echo "== /transform-video + /transform-audio regression"
curl -sS -X POST "$BASE_URL/transform-video" \
  -F "file=@$WORK/video5s.mp4" -F "output_format=mp4" -F "max_width=160" -F "max_duration=2" \
  -o "$WORK/tv.mp4"
check_ok "transform-video" "$WORK/tv.mp4"
check_close "transform-video duration" "$(dur "$WORK/tv.mp4")" 2
curl -sS -X POST "$BASE_URL/transform-audio" \
  -F "file=@$WORK/audio10s.mp3" -F "output_format=wav" -F "max_duration=4" \
  -o "$WORK/ta.wav"
check_ok "transform-audio" "$WORK/ta.wav"
check_close "transform-audio duration" "$(dur "$WORK/ta.wav")" 4

echo "== regression: existing endpoints still up"
HEALTH=$(curl -sS "$BASE_URL/health" | head -c 100)
echo "health: $HEALTH"
curl -sS -X POST "$BASE_URL/video-duration" -H 'Content-Type: application/json' \
  -d "{\"url\":\"$FIX/video5s.mp4\"}" -o "$WORK/vd.json"
grep -q 'duration' "$WORK/vd.json" && { echo "PASS video-duration"; PASS=$((PASS+1)); } \
  || { echo "FAIL video-duration: $(cat "$WORK/vd.json")"; FAIL=$((FAIL+1)); }

docker exec "$CONTAINER" rm -rf "$FIX" 2>/dev/null

echo "== RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
