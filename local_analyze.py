#!/usr/bin/env python3
import os
import json
import subprocess
import numpy as np
from PIL import Image

# Define absolute paths
REPO_DIR = "/Users/karl/web_reports/daily-photo"
PHOTO_PATH = os.path.join(REPO_DIR, "photo.jpg")
INDEX_PATH = os.path.join(REPO_DIR, "index.html")

def calculate_centroid(image_path):
    """Calculates the mathematical center of visual mass using Pillow (Left-Brain Math)."""
    img = Image.open(image_path)
    w, h = img.size
    
    # Resize to 100x100 to reduce high-frequency noise
    img_small = img.resize((100, 100))
    pixels = np.array(img_small, dtype=float)
    
    r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    
    # Relative Luminance (ITU-R BT.601)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Color Saturation Approximation
    avg_color = (r + g + b) / 3.0
    sat = np.sqrt(((r - avg_color)**2 + (g - avg_color)**2 + (b - avg_color)**2) / 3.0)
    
    # Local Contrast (deviation from 8 neighbors)
    contrast = np.zeros_like(lum)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            shifted_lum = np.roll(np.roll(lum, dy, axis=0), dx, axis=1)
            contrast += np.abs(lum - shifted_lum)
    contrast /= 8.0
    
    # Normalize components
    lum_norm = (lum - np.min(lum)) / (np.max(lum) - np.min(lum) + 1e-5)
    lum_dev = np.abs(lum_norm - np.mean(lum_norm))
    
    sat_norm = (sat - np.min(sat)) / (np.max(sat) - np.min(sat) + 1e-5)
    contrast_norm = (contrast - np.min(contrast)) / (np.max(contrast) - np.min(contrast) + 1e-5)
    
    # Combined Visual Weight: Contrast (50%), Saturation (30%), Luminance Deviation (20%)
    weight = 0.5 * contrast_norm + 0.3 * sat_norm + 0.2 * lum_dev
    
    # Project weight to find Center of Mass
    col_weights = np.sum(weight, axis=0)
    row_weights = np.sum(weight, axis=1)
    
    x_coords = np.arange(len(col_weights))
    center_x = np.sum(col_weights * x_coords) / (np.sum(col_weights) + 1e-5)
    
    y_coords = np.arange(len(row_weights))
    center_y = np.sum(row_weights * y_coords) / (np.sum(row_weights) + 1e-5)
    
    return int(center_x), int(center_y), w, h

def query_local_ollama(image_path):
    """Queries the local Ollama LLava model for visual description (Right-Brain Art)."""
    # Downscale image first to speed up Ollama inference
    temp_micro = "/tmp/ollama_analyzable.jpg"
    subprocess.run(["sips", "-Z", "600", image_path, "--out", temp_micro], capture_output=True)
    
    prompt = "What is this image? Describe its main subjects, colors, and setting in two brief sentences."
    cmd = ["ollama", "run", "llava", prompt, temp_micro]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Clean up
    if os.path.exists(temp_micro):
        os.remove(temp_micro)
        
    if result.returncode == 0:
        return result.stdout.strip()
    return "Error: Local Ollama analysis failed."

def update_html_crop(x, y):
    """Patches index.html with the smart crop coordinates."""
    if not os.path.exists(INDEX_PATH):
        print(f"Error: index.html not found at {INDEX_PATH}")
        return False
        
    with open(INDEX_PATH, 'r') as f:
        content = f.read()
        
    # Search for the photo image tag and replace object-position
    import re
    pattern = r'style="object-position:[^"]*"'
    replacement = f'style="object-position: {x}% {y}%;"'
    
    new_content, count = re.subn(pattern, replacement, content)
    if count > 0:
        with open(INDEX_PATH, 'w') as f:
            f.write(new_content)
        return True
    return False

def main():
    if not os.path.exists(PHOTO_PATH):
        print(f"Error: No photo found at {PHOTO_PATH}")
        return
        
    print("=========================================")
    print("Starting Local Zero-Hallucination Pipeline")
    print("=========================================")
    
    # 1. Left-Brain Math Centroid Calculation
    x, y, w, h = calculate_centroid(PHOTO_PATH)
    print(f"1. Image Resolution: {w}x{h}")
    print(f"2. Programmatic Centroid calculated: X={x}%, Y={y}%")
    
    # 2. Right-Brain Art Local Ollama Analysis
    print("3. Running local Ollama vision model (Llava)...")
    description = query_local_ollama(PHOTO_PATH)
    print(f"\n--- Local Vision Model Report ---")
    print(description)
    print("---------------------------------\n")
    
    # 3. Apply the crop
    success = update_html_crop(x, y)
    if success:
        print(f"4. Successfully updated index.html with crop: {x}% {y}%")
    else:
        print("4. Error: Failed to update crop in index.html")
        
    print("=========================================")

if __name__ == "__main__":
    main()
