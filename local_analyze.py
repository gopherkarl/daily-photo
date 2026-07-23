#!/usr/bin/env python3
import os
import json
import subprocess
import re
import urllib.request
import numpy as np
from PIL import Image

REPO_DIR = "/Users/karl/web_reports/daily-photo"
PHOTO_PATH = os.path.join(REPO_DIR, "photo.jpg")
INDEX_PATH = os.path.join(REPO_DIR, "index.html")
REPORT_PATH = os.path.join(REPO_DIR, "analysis_report.json")
OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_VISUAL_WEIGHT_PROMPT = """You are a master photography curator applying the Visual Composition Weight skill and Bryan Peterson's Learning to See principles.

The 9 Visual Weight Drivers:
1. People & Animals (Highest psychological weight - faces, human agents)
2. Sharpness & Focus (Tack sharp vs background blur)
3. Contrast & Tone (Dark objects on bright background or bright spots in dark)
4. Location & Orientation (Off-center placement holds higher visual weight)
5. Color Saturation (Vibrant warm colors anchor focus)
6. Framing & Isolation (Negative space surrounding subject)
7. Size & Scale (Dominant geometric forms)
8. Shape Complexity (Intricate vs flat background)
9. Quantity & Repetition (Pattern-breakers / anomalies)

The Dual-Core Rule:
"Anchor the mathematical luminance centroid, but nudge the crop box to frame the emotional story subject and fulfill asymmetrical visual balance."
"""

def calculate_centroid(image_path):
    """Stage 1: Left-Brain Quantitative Math Centroid & Metrics."""
    img = Image.open(image_path)
    w, h = img.size
    
    img_small = img.resize((100, 100))
    pixels = np.array(img_small, dtype=float)
    r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    avg_color = (r + g + b) / 3.0
    sat = np.sqrt(((r - avg_color)**2 + (g - avg_color)**2 + (b - avg_color)**2) / 3.0)
    
    contrast = np.zeros_like(lum)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(lum, dy, axis=0), dx, axis=1)
            contrast += np.abs(lum - shifted)
    contrast /= 8.0
    
    lum_norm = (lum - np.min(lum)) / (np.max(lum) - np.min(lum) + 1e-5)
    lum_dev = np.abs(lum_norm - np.mean(lum_norm))
    sat_norm = (sat - np.min(sat)) / (np.max(sat) - np.min(sat) + 1e-5)
    contrast_norm = (contrast - np.min(contrast)) / (np.max(contrast) - np.min(contrast) + 1e-5)
    
    weight = 0.5 * contrast_norm + 0.3 * sat_norm + 0.2 * lum_dev
    
    col_weights = np.sum(weight, axis=0)
    row_weights = np.sum(weight, axis=1)
    
    center_x = int(np.sum(col_weights * np.arange(100)) / (np.sum(col_weights) + 1e-5))
    center_y = int(np.sum(row_weights * np.arange(100)) / (np.sum(row_weights) + 1e-5))
    
    return {
        "x": center_x,
        "y": center_y,
        "width": w,
        "height": h,
        "aspect_ratio": round(w / h, 2)
    }

def query_llava_vision(image_path):
    """Stage 2: Right-Brain Qualitative Vision (Llava)."""
    temp_micro = "/tmp/ollama_analyzable.jpg"
    subprocess.run(["sips", "-Z", "800", image_path, "--out", temp_micro], capture_output=True)
    
    prompt = ("Describe this photograph in detail for a photographic editor. "
              "1. What is the primary subject and where is it located in the frame (e.g. top-left, center, lower-right)? "
              "2. Are there human figures or agents? Where are they located? "
              "3. Describe the lighting, contrast, and visual background context.")
    cmd = ["ollama", "run", "llava", prompt, temp_micro]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if os.path.exists(temp_micro):
        os.remove(temp_micro)
        
    return result.stdout.strip() if result.returncode == 0 else "Local vision analysis unavailable."

def run_synthesis_judge(lum_data, vision_desc, model="qwen2.5-coder:32b"):
    """Stage 3 & 4: Visual Weight Skill Reasoning + Synthesis Judge."""
    prompt = f"""
Image Quantitative Math Centroid:
- Dimensions: {lum_data['width']}x{lum_data['height']} (Aspect Ratio: {lum_data['aspect_ratio']})
- Quantitative Centroid: X={lum_data['x']}%, Y={lum_data['y']}%

Qualitative Vision Report (Llava):
"{vision_desc}"

Task:
Synthesize the Quantitative Math Centroid and Qualitative Vision Report using the Visual Composition Weight rules.
Determine the optimal CSS `object-position` crop coordinates (final_x, final_y) from 0% to 100%.

Return ONLY a JSON object with this exact shape:
{{
  "math_centroid": {{"x": {lum_data['x']}, "y": {lum_data['y']}}},
  "visual_weight_analysis": "<2-sentence analysis of focal drivers and fulcrum balance>",
  "final_x": <int 0-100>,
  "final_y": <int 0-100>,
  "justification": "<2-sentence summary explaining why this final crop was selected over the raw centroid>"
}}
"""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_VISUAL_WEIGHT_PROMPT,
        "stream": False,
        "options": {"num_predict": 350}
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            res = json.loads(resp.read().decode('utf-8'))
            raw_response = res.get("response", "")
            match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
    except Exception as e:
        print(f"Error calling Ollama judge: {e}")
        
    return {
        "math_centroid": {"x": lum_data['x'], "y": lum_data['y']},
        "visual_weight_analysis": "Fallback: unable to complete LLM synthesis.",
        "final_x": lum_data['x'],
        "final_y": lum_data['y'],
        "justification": "Fallback to raw mathematical centroid."
    }

def update_html_crop(x, y):
    """Patches index.html with the smart crop coordinates."""
    if not os.path.exists(INDEX_PATH):
        print(f"Error: index.html not found at {INDEX_PATH}")
        return False
        
    with open(INDEX_PATH, 'r') as f:
        content = f.read()
        
    pattern = r'style="object-position:[^"]*"'
    replacement = f'style="object-position: {x}% {y}%;"'
    
    new_content, count = re.subn(pattern, replacement, content)
    if count > 0:
        with open(INDEX_PATH, 'w') as f:
            f.write(new_content)
        return True
    return False

def main():
    print("=========================================")
    print("Starting Dual-Core Visual Weight Pipeline")
    print("=========================================")
    
    # 1. Left-Brain Math Centroid Calculation
    lum_data = calculate_centroid(PHOTO_PATH)
    print(f"1. Quantitative Centroid: X={lum_data['x']}%, Y={lum_data['y']}% ({lum_data['width']}x{lum_data['height']})")
    
    # 2. Right-Brain Art Local Ollama Analysis
    print("2. Running local Ollama vision model (Llava)...")
    vision_desc = query_llava_vision(PHOTO_PATH)
    print(f"   Vision Description: {vision_desc[:120]}...")
    
    # 3. Visual Composition Weight Skill Synthesis Judge
    print("3. Synthesizing visual weight principles via local LLM (qwen2.5-coder:32b)...")
    report = run_synthesis_judge(lum_data, vision_desc)
    
    final_x = report.get("final_x", lum_data['x'])
    final_y = report.get("final_y", lum_data['y'])
    
    print("\n--- Dual-Core Synthesis Report ---")
    print(json.dumps(report, indent=2))
    print("----------------------------------\n")
    
    # Save detailed report JSON
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
        
    # 4. Apply the crop to index.html
    success = update_html_crop(final_x, final_y)
    if success:
        print(f"4. Successfully updated index.html with smart crop: {final_x}% {final_y}%")
    else:
        print("4. Error: Failed to update crop in index.html")
        
    print("=========================================")

if __name__ == "__main__":
    main()
