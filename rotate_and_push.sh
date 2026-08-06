#!/bin/bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${REPO_DIR}/rotation.log"
LOCK_DIR="${REPO_DIR}/.rotation.lock"
PYTHON="${DAILY_PHOTO_PYTHON:-${REPO_DIR}/.venv-grounding/bin/python}"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "Another daily photo rotation is already running; exiting safely." >&2
    exit 2
fi
cleanup() { rmdir "$LOCK_DIR" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

exec > >(tee -a "$LOG_FILE") 2>&1

cd "$REPO_DIR"
echo "========================================="
echo "Starting daily photo rotation: $(date)"
echo "========================================="

echo "Running rotate.py..."
"$PYTHON" rotate.py

echo "Running local_analyze.py with local Ollama vision model..."
"$PYTHON" local_analyze.py

echo "Checking git status..."

if [ -n "$(git status --porcelain)" ]; then
    echo "Changes detected. Staging changes..."
    git add README.md photo.jpg state.json index.html local_analyze.py grounding_localizer.py rotate.py rotate_and_push.sh analysis_report.json display_profiles.json crop_geometry.py qa_summary.md qa_source_preview.jpg
    NEW_PHOTO=$("$PYTHON" -c "import json; print(json.load(open('state.json'))['last_shown'])")
    git commit -m "Rotate to photo: ${NEW_PHOTO}"
    git push origin main
    echo "Successfully updated and pushed daily photo: ${NEW_PHOTO}!"
else
    echo "No changes detected. The photo is already up to date."
fi

echo "Rotation complete."
echo "========================================="
