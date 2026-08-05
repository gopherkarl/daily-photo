#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
import tempfile

POOL_DIR = os.environ.get("DAILY_PHOTO_POOL", "/Users/karl/Pictures/daily_photo_pool")
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(REPO_DIR, "state.json")
OUTPUT_PHOTO = os.path.join(REPO_DIR, "photo.jpg")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tiff"}


def get_pool_photos(pool_dir=POOL_DIR):
    if not os.path.exists(pool_dir):
        os.makedirs(pool_dir, exist_ok=True)
        return []
    return sorted(
        name for name in os.listdir(pool_dir)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    )


def load_state(path=STATE_FILE):
    if not os.path.exists(path):
        return {"last_shown": None, "history": []}
    try:
        with open(path) as handle:
            state = json.load(handle)
        history = state.get("history", [])
        return {"last_shown": state.get("last_shown"), "history": history if isinstance(history, list) else []}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error reading state file: {exc}. Resetting state.", file=sys.stderr)
        return {"last_shown": None, "history": []}


def save_state(state, path=STATE_FILE):
    directory = os.path.dirname(path) or "."
    fd, temp_path = tempfile.mkstemp(prefix="state-", suffix=".json", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(state, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def select_photo(photos, state):
    history = [name for name in state.get("history", []) if name in photos]
    last_shown = state.get("last_shown")
    unshown = [photo for photo in photos if photo not in history]
    if not unshown:
        history = [last_shown] if last_shown in photos else []
        unshown = [photo for photo in photos if photo not in history] or list(photos)
    selected = unshown[0]
    history.append(selected)
    return selected, {"last_shown": selected, "history": history}


def copy_or_convert(source, destination):
    extension = os.path.splitext(source)[1].lower()
    if extension in {".heic", ".heif"}:
        result = subprocess.run(
            ["sips", "-s", "format", "jpeg", source, "--out", destination],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "sips conversion failed")
    else:
        shutil.copy2(source, destination)


def main():
    photos = get_pool_photos()
    if not photos:
        print(f"No photographs found in directory: {POOL_DIR}", file=sys.stderr)
        return 1
    state = load_state()
    selected, new_state = select_photo(photos, state)
    source = os.path.join(POOL_DIR, selected)
    print(f"Selected next photo in rotation: {selected}")
    try:
        copy_or_convert(source, OUTPUT_PHOTO)
        save_state(new_state)
        print("Photo copied and state written atomically.")
        return 0
    except (OSError, RuntimeError) as exc:
        print(f"Error processing image rotation: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

# Testable functions: get_pool_photos, select_photo, and save_state.
# The production process is serialized by rotate_and_push.sh.

# pyright: ignore
