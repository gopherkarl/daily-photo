#!/usr/bin/env python3
"""Analyze the selected photo and generate display-aware crop positions."""
import json
import os
import re
import subprocess
import tempfile
import hashlib
import base64
import urllib.request
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from crop_geometry import (
    bbox_inside_window,
    bbox_percent_to_pixels,
    crop_position_for_bbox,
    display_window,
    percent_window,
    visible_source_window,
    bbox_overlap_fraction,
    candidate_crop,
    score_candidate,
)
from grounding_localizer import canonical_category, locate as grounding_locate, match_detections_to_elements

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTO_PATH = os.path.join(REPO_DIR, "photo.jpg")
INDEX_PATH = os.path.join(REPO_DIR, "index.html")
REPORT_PATH = os.path.join(REPO_DIR, "analysis_report.json")
DISPLAY_PROFILES_PATH = os.path.join(REPO_DIR, "display_profiles.json")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
VISION_MODEL = os.environ.get("DAILY_PHOTO_VISION_MODEL", "qwen3-vl:8b")
JUDGE_MODEL = os.environ.get("DAILY_PHOTO_JUDGE_MODEL", "qwen3:32b")
USE_GROUNDING_DINO = os.environ.get("DAILY_PHOTO_USE_GROUNDING_DINO", "1") != "0"
SCENE_DISAGREEMENT_THRESHOLD = 0.35

SYSTEM_VISUAL_WEIGHT_PROMPT = """You are a photography composition reviewer. Analyze only the supplied image.
Do not infer content from filenames or prior context. Identify the primary subject,
its complete bounding box, and confidence. Coordinates are percentages: left/top are
0 and right/bottom are 100. Return valid JSON only."""


def load_display_profiles():
    with open(DISPLAY_PROFILES_PATH) as handle:
        data = json.load(handle)
    return data["profiles"], data["default_profile"]


def calculate_centroid(image_path):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    pixels = np.asarray(img.resize((100, 100)), dtype=float)
    r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    average = (r + g + b) / 3.0
    saturation = np.sqrt(((r - average) ** 2 + (g - average) ** 2 + (b - average) ** 2) / 3.0)
    contrast = np.zeros_like(luminance)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dy, dx) != (0, 0):
                shifted = np.roll(np.roll(luminance, dy, axis=0), dx, axis=1)
                contrast += np.abs(luminance - shifted)
    contrast /= 8.0

    def normalize(value):
        return (value - np.min(value)) / (np.max(value) - np.min(value) + 1e-5)

    lum_dev = np.abs(normalize(luminance) - np.mean(normalize(luminance)))
    weight = 0.5 * normalize(contrast) + 0.3 * normalize(saturation) + 0.2 * lum_dev
    columns, rows = np.sum(weight, axis=0), np.sum(weight, axis=1)
    x = int(np.sum(columns * np.arange(100)) / (np.sum(columns) + 1e-5))
    y = int(np.sum(rows * np.arange(100)) / (np.sum(rows) + 1e-5))
    return {"x": x, "y": y, "width": width, "height": height, "aspect_ratio": round(width / height, 2)}


def extract_json(text):
    # Ollama's interactive CLI may emit ANSI spinner/cursor sequences around output.
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = text.replace("\x1b", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("vision model returned no JSON object")
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Repair literal control characters only when they occur inside JSON strings;
        # structural newlines outside strings must remain unchanged.
        repaired_chars = []
        in_string = False
        escaped = False
        for char in candidate:
            if char == '"' and not escaped:
                in_string = not in_string
            if in_string and ord(char) < 32:
                char = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(char, " ")
            repaired_chars.append(char)
            escaped = (char == "\\" and not escaped)
            if char != "\\":
                escaped = False
        return json.loads("".join(repaired_chars))


def validate_bbox(bbox, x_scale=None, y_scale=None):
    if not isinstance(bbox, dict):
        raise ValueError("missing bbox_pct")
    required = ("left", "top", "right", "bottom")
    if any(key not in bbox for key in required):
        raise ValueError("bbox_pct missing coordinate")
    values = {key: float(bbox[key]) for key in required}
    if x_scale and y_scale:
        values["left"] = values["left"] / x_scale * 100
        values["right"] = values["right"] / x_scale * 100
        values["top"] = values["top"] / y_scale * 100
        values["bottom"] = values["bottom"] / y_scale * 100
    elif max(values.values()) > 100:
        values = {key: value / 10.0 for key, value in values.items()}
    # Permit small model overshoot at the image boundary, but reject grossly malformed boxes.
    if any(value < -20 or value > 120 for value in values.values()):
        raise ValueError("bbox_pct is outside image bounds")
    values = {key: max(0.0, min(100.0, value)) for key, value in values.items()}
    if not (values["left"] < values["right"] and values["top"] < values["bottom"]):
        raise ValueError("bbox_pct has no visible area")
    return {key: round(values[key], 2) for key in required}


def infer_bbox_scales(raw_elements):
    boxes = [item.get("bbox_pct", {}) for item in raw_elements if isinstance(item, dict)]
    x_values = [float(box[key]) for box in boxes for key in ("left", "right") if key in box]
    y_values = [float(box[key]) for box in boxes for key in ("top", "bottom") if key in box]
    max_x, max_y = max(x_values or [100]), max(y_values or [100])
    if max_x <= 100 and max_y <= 100:
        return None, None
    return max_x, max_y


def validate_vision_report(report):
    raw_elements = report.get("elements")
    if not isinstance(raw_elements, list) or not raw_elements:
        raise ValueError("vision response must contain elements")
    elements = []
    x_scale, y_scale = infer_bbox_scales(raw_elements)
    valid_roles = {"primary_anchor", "background_mass", "focal_anomaly", "context", "secondary_subject"}
    for raw in raw_elements:
        if not isinstance(raw, dict) or not str(raw.get("name", "")).strip():
            continue
        role = str(raw.get("role", "secondary_subject")).strip().lower()
        role_aliases = {
            "primary_subject": "primary_anchor",
            "main_subject": "primary_anchor",
            "subject": "primary_anchor",
            "focal_subject": "primary_anchor",
            "anomaly": "focal_anomaly",
            "pattern_breaker": "focal_anomaly",
            "isolated_subject": "focal_anomaly",
            "environmental_context": "context",
            "setting": "context",
            "scene_context": "context",
        }
        role = role_aliases.get(role, role)
        if role not in valid_roles:
            role = "secondary_subject"
        elements.append({
            "name": str(raw["name"]).strip(),
            "role": role,
            "bbox_pct": validate_bbox(raw.get("bbox_pct"), x_scale, y_scale),
            "confidence": max(0.0, min(1.0, float(raw.get("confidence", 0)))),
            "narrative_value": max(0.0, min(1.0, float(raw.get("narrative_value", 0.5)))),
            "description": str(raw.get("description", "")).strip(),
        })
    if not elements:
        raise ValueError("no valid visual elements")
    anchor = next((item for item in elements if item["role"] == "primary_anchor"), None)
    if anchor is None:
        eligible = [item for item in elements if item["role"] != "background_mass"]
        if eligible:
            anchor = max(eligible, key=lambda item: item["confidence"] * item["narrative_value"])
            anchor["role"] = "primary_anchor"
        else:
            raise ValueError("vision response must identify a primary_anchor")
    anomaly = next((item for item in elements if item["role"] == "focal_anomaly"), None)
    contexts = [item for item in elements if item["role"] == "context"]
    if not contexts:
        background = next((item for item in elements if item["role"] == "background_mass"), None)
        if background and background["name"] != anchor["name"]:
            background["role"] = "context"
            contexts = [background]
    if not contexts:
        # The anchor's surrounding scene is still useful context when the VLM
        # omits a separate environmental element. Use a broad, non-anchor box.
        anchor_box = anchor["bbox_pct"]
        contexts = [{
            "name": "surrounding scene",
            "role": "context",
            "bbox_pct": {"left": 0, "top": 0, "right": 100, "bottom": 100},
            "confidence": 0.4,
            "narrative_value": 0.4,
            "description": "surrounding visual environment",
            "synthetic": True,
        }]
    if not any(item["name"] == "surrounding scene" for item in elements):
        elements.extend(contexts)
    return {
        "elements": elements,
        "primary_anchor": anchor["name"],
        "focal_element": anomaly["name"] if anomaly else None,
        "confidence": anchor["confidence"],
        "description": str(report.get("description", "")).strip(),
        "lighting": str(report.get("lighting", "")).strip(),
        "background": str(report.get("background", "")).strip(),
    }


def query_vision(image_path, verification=False):
    prompt = """Analyze the attached photograph from its pixels only for a portrait-phone crop editor. Do not infer content from the filename, previous images, this prompt, or any example scene. Return ONLY valid JSON with this shape:
{
  "elements": [
    {"name":"literal element name", "role":"primary_anchor|background_mass|focal_anomaly|context|secondary_subject", "bbox_pct":{"left":0,"top":0,"right":0,"bottom":0}, "confidence":0.0, "narrative_value":0.0, "description":"literal visual description"}
  ],
  "description":"literal scene description",
  "lighting":"brief lighting description",
  "background":"brief background description"
}
Return at least one element. Classify dense repetitive material as background_mass. Classify an isolated person, animal, object, bright focal point, or pattern-breaker with high narrative value as focal_anomaly. Include physical surroundings, signage, architecture, or other elements needed to explain the scene as context. Every bbox must contain the COMPLETE element, with coordinates from the original image: left/top=0 and right/bottom=100. Do not use vague locations without numeric bounding boxes. First identify the actual scene and its distinct elements; then produce the JSON."""
    if verification:
        prompt += " Re-check every element independently, especially the focal anomaly and context boundaries."
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="daily-photo-", suffix=".jpg", delete=False) as temp:
            temp_path = temp.name
        result = subprocess.run(["sips", "-Z", "1024", image_path, "--out", temp_path], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "sips failed")
        source_bytes = open(image_path, "rb").read()
        source_hash = hashlib.sha256(source_bytes).hexdigest()[:12]
        prompt += f" Image identity hash for this run: {source_hash}. Analyze this attached image, not any prior image."
        image_b64 = base64.b64encode(open(temp_path, "rb").read()).decode("ascii")
        if VISION_MODEL.startswith("qwen3-vl"):
            endpoint = OLLAMA_URL.replace("/api/generate", "/api/chat")
            payload = {
                "model": VISION_MODEL,
                "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.1},
            }
        else:
            endpoint = OLLAMA_URL
            payload = {
                "model": VISION_MODEL,
                "prompt": prompt,
                "images": [image_b64],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.1},
            }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
        raw_response = (
            result.get("message", {}).get("content", "")
            if "message" in result
            else result.get("response", "")
        )
        try:
            return validate_vision_report(extract_json(raw_response))
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"invalid vision report: {exc}; raw={raw_response[:1200]}") from exc
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def call_ollama_json(model, prompt):
    payload = {"model": model, "prompt": prompt, "format": "json", "stream": False, "options": {"temperature": 0.1}}
    request = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))
    return extract_json(result.get("response", ""))


def scene_signature(vision):
    return " ".join(sorted(item["name"].lower() for item in vision["elements"]))


def scene_consistent(first, second):
    a = set(scene_signature(first).split())
    b = set(scene_signature(second).split())
    if not a or not b:
        return False
    return len(a & b) / max(1, len(a | b)) >= SCENE_DISAGREEMENT_THRESHOLD


def ground_vision_elements(image_path, vision):
    """Replace VLM-estimated boxes with independent Grounding DINO boxes."""
    # Grounding DINO localizes generic categories; the VLM retains attributes,
    # narrative roles, and semantic identity. This avoids exact phrase matching.
    labels = sorted({
        canonical_category(item["name"])
        for item in vision["elements"]
        if not item.get("synthetic", False)
    })
    detections = grounding_locate(image_path, labels)
    grounded = match_detections_to_elements(vision["elements"], detections)
    primary = next(item for item in grounded if item["name"] == vision["primary_anchor"])
    if "detector_score" not in primary and primary["role"] == "primary_anchor":
        raise RuntimeError(f"Grounding DINO could not localize primary element: {primary['name']}")
    return {**vision, "elements": grounded, "localization_model": "Grounding DINO-T + VLM assignment"}


def synthesize_analysis(lum_data, vision, display_context, candidates):
    prompt = json.dumps({
        "task": "Choose the best composition candidate. Return JSON only.",
        "rules": [
            "primary_anchor defines the photograph's identity",
            "context explains the scene",
            "focal_anomaly adds narrative value but must not displace the primary anchor",
            "background_mass supports but does not dominate",
            "prefer candidates with anchor visible and at least one context element visible",
        ],
        "math_centroid": {"x": lum_data["x"], "y": lum_data["y"]},
        "vision": vision,
        "candidates": candidates,
        "required_output": {"recommended_candidate": "candidate name", "primary_anchor": "name", "selected_context": ["name"], "selected_anomaly": "name or null", "confidence": 0.0, "composition_strategy": "brief explanation"},
    })
    try:
        return call_ollama_json(JUDGE_MODEL, prompt)
    except Exception:
        return {"recommended_candidate": max(candidates, key=lambda item: item["score"])["name"], "confidence": 0.0, "composition_strategy": "Deterministic fallback to highest-scoring candidate."}


def update_html_crop(positions):
    with open(INDEX_PATH) as handle:
        content = handle.read()
    pattern = r'(<img\b[^>]*class="photo"[^>]*)(?:\s+style="[^"]*")?([^>]*>)'

    def replace(match):
        prefix, suffix = match.groups()
        return f'{prefix} style="object-position: {positions["portrait_x"]}% {positions["portrait_y"]}%"{suffix}'

    new_content, count = re.subn(pattern, replace, content, count=1)
    if count != 1:
        raise RuntimeError("could not find the photo element in index.html")
    with open(INDEX_PATH, "w") as handle:
        handle.write(new_content)


def update_html_profiles(positions):
    with open(INDEX_PATH) as handle:
        content = handle.read()
    # Remove the old inline crop because inline style would override the landscape media query.
    pattern = r'(<img\b[^>]*class="photo")[^>]*>'

    def replace(match):
        return f'{match.group(1)} src="photo.jpg" alt="Daily Photograph" onerror="this.style.display=\'none\';">'

    new_content, count = re.subn(pattern, replace, content, count=1)
    if count != 1:
        raise RuntimeError("could not find the photo element in index.html")
    css_marker = "    /* Generated display-aware crop positions */"
    css = f'''{css_marker}
    .photo {{ object-position: {positions["portrait_x"]}% {positions["portrait_y"]}%; }}
    @media (orientation: landscape) {{ .photo {{ object-position: {positions["landscape_x"]}% {positions["landscape_y"]}%; }} }}
'''
    if css_marker in new_content:
        new_content = re.sub(r"    /\* Generated display-aware crop positions \*/.*?(?=  </style>)", css, new_content, flags=re.DOTALL)
    else:
        new_content = new_content.replace("  </style>", css + "  </style>")
    with open(INDEX_PATH, "w") as handle:
        handle.write(new_content)


def create_qa_artifact(lum_data, vision, positions, validation):
    try:
        img = Image.open(PHOTO_PATH).convert("RGB")
        width, height = img.size
        img.thumbnail((1200, 1200))
        img.save(os.path.join(REPO_DIR, "qa_source_preview.jpg"), quality=85)
    except Exception:
        pass
    with open(os.path.join(REPO_DIR, "qa_summary.md"), "w") as handle:
        handle.write(f"# Crop QA — {datetime.now(timezone.utc).isoformat()}\n\n")
        handle.write(f"- Source: `{os.path.basename(PHOTO_PATH)}` ({lum_data['width']}×{lum_data['height']})\n")
        handle.write(f"- Localization model: **{vision.get('localization_model', 'VLM-only')}**\n")
        handle.write(f"- Primary anchor: **{vision['primary_anchor']}**\n")
        handle.write(f"- Focal anomaly: **{vision['focal_element'] or 'none'}**\n")
        handle.write(f"- Elements: `{[(item['name'], item['role'], item['bbox_pct']) for item in vision['elements']]}`\n")
        handle.write(f"- Portrait crop: `{positions['portrait_x']}% {positions['portrait_y']}%`\n")
        handle.write(f"- Landscape crop: `{positions['landscape_x']}% {positions['landscape_y']}%`\n")
        handle.write(f"- Portrait anchor overlap: **{validation['portrait_primary_anchor_overlap']:.2f}**\n")
        handle.write(f"- Landscape anchor overlap: **{validation['landscape_primary_anchor_overlap']:.2f}**\n")
        handle.write(f"- Portrait composition valid: **{validation['portrait_composition_valid']}**\n")
        handle.write(f"- Landscape composition valid: **{validation['landscape_composition_valid']}**\n")
        handle.write(f"- Portrait anchor-priority fallback: **{validation['portrait_anchor_priority_fallback']}**\n")
        handle.write(f"- Portrait anchor-center fallback: **{validation.get('portrait_anchor_center_fallback', False)}**\n")
        handle.write(f"- Landscape anchor-priority fallback: **{validation['landscape_anchor_priority_fallback']}**\n")
        handle.write(f"- Landscape anchor-center fallback: **{validation.get('landscape_anchor_center_fallback', False)}**\n")
        handle.write("\nThe preview image is `qa_source_preview.jpg`; the production image is not replaced by this artifact.\n")


def main():
    lum_data = calculate_centroid(PHOTO_PATH)
    profiles, default_profile = load_display_profiles()
    vision = query_vision(PHOTO_PATH)
    confidence = vision["confidence"]
    has_context = any(item["role"] == "context" for item in vision["elements"])
    if confidence < 0.65 or not has_context:
        try:
            second = query_vision(PHOTO_PATH, verification=True)
        except (ValueError, RuntimeError, OSError):
            second = None
        if second is not None:
            second_has_context = any(item["role"] == "context" for item in second["elements"])
            if second["confidence"] > confidence or (second_has_context and not has_context):
                vision = second

    if USE_GROUNDING_DINO:
        vision = ground_vision_elements(PHOTO_PATH, vision)

    anchor = next(item for item in vision["elements"] if item["name"] == vision["primary_anchor"])
    anomaly = next((item for item in vision["elements"] if item["name"] == vision["focal_element"]), None)
    contexts = [item for item in vision["elements"] if item["role"] == "context"]
    anchor_px = bbox_percent_to_pixels(anchor["bbox_pct"], lum_data["width"], lum_data["height"])
    anomaly_px = bbox_percent_to_pixels(anomaly["bbox_pct"], lum_data["width"], lum_data["height"]) if anomaly else None
    context_px = [bbox_percent_to_pixels(item["bbox_pct"], lum_data["width"], lum_data["height"]) for item in contexts]
    profile_windows, positions, validation, candidates_by_profile = {}, {}, {}, {}
    for name, profile in profiles.items():
        window = display_window(lum_data["width"], lum_data["height"], profile["width"], profile["height"])
        raw_candidates = {
            "anchor_only": candidate_crop(window, anchor_px),
            "anchor_plus_context": candidate_crop(window, anchor_px, context_px),
            "anchor_plus_anomaly": candidate_crop(window, anchor_px, anomaly_bbox=anomaly_px),
            "anchor_context_anomaly": candidate_crop(window, anchor_px, context_px, anomaly_px),
        }
        candidates = []
        for candidate_name, candidate in raw_candidates.items():
            candidate["name"] = candidate_name
            candidate["score"] = score_candidate(candidate, lum_data, anomaly_present=anomaly is not None)
            candidates.append(candidate)
        candidates_by_profile[name] = candidates
        profile_windows[name] = {**window, "candidates": candidates}

    display_context = {"default_profile": default_profile, "profiles": profile_windows, "position_method": "candidate crops scored around primary anchor, context, and focal anomaly"}
    synthesis = synthesize_analysis(lum_data, vision, display_context, candidates_by_profile[default_profile])
    selected_name = synthesis.get("recommended_candidate", "anchor_plus_context")
    default_candidates = {item["name"]: item for item in candidates_by_profile[default_profile]}
    selected = default_candidates.get(selected_name, max(default_candidates.values(), key=lambda item: item["score"]))
    default_prefix = "portrait" if default_profile == "portrait_phone" else "landscape"
    if not selected["anchor_inside"]:
        # Anchor-priority fallback: preserve the photograph's identity even when
        # no candidate can retain both the anchor and contextual material.
        selected = default_candidates["anchor_only"]
        if selected["anchor_inside"]:
            selected["fallback"] = "anchor_priority"
        else:
            # If the anchor is wider than the portrait viewport, center the
            # viewport on the anchor's center rather than rejecting the run.
            # candidate_crop() already computes this center-based position;
            # containment is impossible by geometry, but the crop remains the
            # best identity-preserving portrait representation.
            selected["fallback"] = "anchor_center"
    for name, candidates in candidates_by_profile.items():
        selected_profile = selected if name == default_profile else max(candidates, key=lambda item: item["score"])
        prefix = "portrait" if name == "portrait_phone" else "landscape"
        positions[f"{prefix}_x"], positions[f"{prefix}_y"] = selected_profile["object_x"], selected_profile["object_y"]
        validation[f"{prefix}_primary_anchor_inside"] = selected_profile["anchor_inside"]
        validation[f"{prefix}_primary_anchor_overlap"] = selected_profile.get("anchor_overlap", 0.0)
        validation[f"{prefix}_context_inside"] = selected_profile["context_inside"]
        validation[f"{prefix}_focal_anomaly_inside"] = selected_profile["anomaly_inside"]
        # Anchor-only is an intentional valid fallback for narrow portrait
        # displays. Context is preferred, but must not displace the identity
        # anchor when the source geometry cannot fit both.
        validation[f"{prefix}_composition_valid"] = (
            selected_profile["anchor_inside"]
            or selected_profile.get("fallback") == "anchor_center"
        )
        validation[f"{prefix}_anchor_center_fallback"] = (
            selected_profile.get("fallback") == "anchor_center"
        )
        validation[f"{prefix}_context_preferred_but_unavailable"] = (
            selected_profile["anchor_inside"]
            and not any(selected_profile["context_inside"])
            and selected_profile["name"] == "anchor_only"
        )
        validation[f"{prefix}_anchor_priority_fallback"] = (
            selected_profile.get("fallback") == "anchor_priority"
        )
    print(f"Primary anchor: {vision['primary_anchor']}; selected candidate: {selected['name']}; portrait crop: {positions['portrait_x']}% {positions['portrait_y']}%")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_photo": os.path.basename(PHOTO_PATH),
        "math_centroid": {"x": lum_data["x"], "y": lum_data["y"]},
        "vision_model": VISION_MODEL,
        "vision": vision,
        "synthesis_model": JUDGE_MODEL,
        "synthesis": synthesis,
        "positions": positions,
        "display_context": display_context,
        "validation": validation,
    }
    with open(REPORT_PATH, "w") as handle:
        json.dump(report, handle, indent=2)
    update_html_profiles(positions)
    create_qa_artifact(lum_data, vision, positions, validation)
    print(json.dumps(report, indent=2))
    default_prefix = "portrait" if default_profile == "portrait_phone" else "landscape"
    if (
        not validation[f"{default_prefix}_primary_anchor_inside"]
        and not validation[f"{default_prefix}_anchor_center_fallback"]
    ):
        raise RuntimeError(
            f"primary anchor could not be preserved for default profile {default_profile}: "
            f"overlap={validation[f'{default_prefix}_primary_anchor_overlap']}"
        )


if __name__ == "__main__":
    main()
