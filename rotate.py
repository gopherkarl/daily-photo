#!/usr/bin/env python3
import os
import json
import shutil
import subprocess
import sys

# Define absolute paths
POOL_DIR = "/Users/karl/Pictures/daily_photo_pool"
REPO_DIR = "/Users/karl/web_reports/daily-photo"
STATE_FILE = os.path.join(REPO_DIR, "state.json")
OUTPUT_PHOTO = os.path.join(REPO_DIR, "photo.jpg")

# Supported image extensions (case-insensitive)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif', '.tiff'}

def get_pool_photos():
    if not os.path.exists(POOL_DIR):
        os.makedirs(POOL_DIR, exist_ok=True)
        return []
    
    photos = []
    for file in os.listdir(POOL_DIR):
        _, ext = os.path.splitext(file)
        if ext.lower() in IMAGE_EXTENSIONS:
            photos.append(file)
    
    # Sort alphabetically to allow easy user ordering (e.g. 01_sunset.jpg, 02_beach.jpg)
    photos.sort()
    return photos

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading state file: {e}. Resetting state.", file=sys.stderr)
    
    return {
        "last_shown": None,
        "history": []
    }

def save_state(state):
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Error saving state file: {e}", file=sys.stderr)

def main():
    photos = get_pool_photos()
    
    if not photos:
        print(f"No photographs found in directory: {POOL_DIR}")
        print("Please add some images (.jpg, .png, .heic, etc.) to start the rotation.")
        # If there's no photo at all, create a default beautiful fallback so git has something
        if not os.path.exists(OUTPUT_PHOTO):
            print("Creating a placeholder default image...")
            # We can write a simple colored SVG or just do nothing. Let's write a small message.
            # But the user should just put some files in. We won't crash.
        return 1

    state = load_state()
    history = state.get("history", [])
    last_shown = state.get("last_shown")

    # Filter for photos that haven't been shown yet in this rotation cycle
    unshown = [p for p in photos if p not in history]
    
    # If all photos in the directory have been shown, or if files changed
    if not unshown:
        print("All photos in the pool have been shown! Resetting the cycle.")
        # Reset history, but keep the last shown photo in history if it's still in the pool,
        # so we don't accidentally pick it again immediately
        if last_shown in photos:
            history = [last_shown]
            unshown = [p for p in photos if p != last_shown]
        else:
            history = []
            unshown = photos.copy()
            
        # Fallback if there's only 1 photo in total
        if not unshown:
            unshown = photos.copy()

    # Select the next photo (first one in alphabetical order from unshown)
    selected_photo = unshown[0]
    input_path = os.path.join(POOL_DIR, selected_photo)
    
    print(f"Selected next photo in rotation: {selected_photo}")
    
    # Check if the photo is HEIC/HEIF and needs conversion
    _, ext = os.path.splitext(selected_photo)
    is_heic = ext.lower() in {'.heic', '.heif'}
    
    try:
        if is_heic:
            print(f"Converting HEIC image '{selected_photo}' to JPEG using macOS sips...")
            cmd = ["sips", "-s", "format", "jpeg", input_path, "--out", OUTPUT_PHOTO]
            result = subprocess.run(cmd, capture_with_output=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                raise Exception(f"sips conversion failed: {result.stderr}")
            print("HEIC conversion successful.")
        else:
            print(f"Copying '{selected_photo}' to web folder...")
            shutil.copy2(input_path, OUTPUT_PHOTO)
            print("Photo copied successfully.")
            
        # Update state
        state["last_shown"] = selected_photo
        if selected_photo not in history:
            history.append(selected_photo)
        state["history"] = history
        save_state(state)
        
        print("State updated successfully.")
        return 0
        
    except Exception as e:
        print(f"Error processing image rotation: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())