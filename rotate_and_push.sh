#!/bin/bash
# Exit on error
set -e

# Export standard paths so git, sips, and python3 can be found from cron environment
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

REPO_DIR="/Users/karl/web_reports/daily-photo"
LOG_FILE="${REPO_DIR}/rotation.log"

# Redirect stdout and stderr to log file for easy debugging, while keeping a copy in stdout
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================="
echo "Starting daily photo rotation: $(date)"
echo "========================================="

# Navigate to repo directory
cd "${REPO_DIR}"

# Run the rotation script
echo "Running rotate.py..."
/Users/karl/.hermes/hermes-agent/venv/bin/python3 rotate.py

# Run the local vision analysis and crop calculation
echo "Running local_analyze.py with local Ollama vision model..."
/Users/karl/.hermes/hermes-agent/venv/bin/python3 local_analyze.py

# Check git status
echo "Checking git status..."
if [ -n "$(git status --porcelain)" ]; then
    echo "Changes detected. Staging changes..."
    git add photo.jpg state.json index.html local_analyze.py rotate_and_push.sh
    
    # Get the name of the newly shown photo from state.json
    NEW_PHOTO=$(/Users/karl/.hermes/hermes-agent/venv/bin/python3 -c "import json; print(json.load(open('state.json'))['last_shown'])")
    
    echo "Committing changes..."
    git commit -m "Rotate to photo: ${NEW_PHOTO}"
    
    echo "Pushing to GitHub..."
    git push origin main
    echo "Successfully updated and pushed daily photo: ${NEW_PHOTO}!"
else
    echo "No changes detected. The photo is already up to date."
fi

echo "Rotation complete."
echo "========================================="