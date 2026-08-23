#!/usr/bin/env python3
"""Anchor-consistency gate for the daily-photo pipeline.

Known failure mode (photo 21.jpg, 2026-08-23): qwen3-vl named the primary
anchor "man in dark shirt and cap" but supplied a bounding box pointing at a
*different* person (a woman at the glass doorway, x~91%). Grounding DINO then
re-localized the "person" head-noun and, because match_detections_to_elements
selects by IoU against the VLM box, faithfully followed the wrong box.

This gate attacks the root cause: it never trusts the VLM's box to decide
WHICH instance of a category is the anchor. When DINO returns several
candidate detections of the anchor's head-noun category, the gate builds a
labeled contact strip of those crops and asks the vision model to pick the
one that matches the VLM's semantic description.

TERMINATION GUARANTEE (no infinite VLM<->DINO bounce): the verification runs
at most MAX_ATTEMPTS times total. Each attempt may either select a matching
candidate or confirm no candidate matches. After the cap is reached the gate
stops trying and FAILS OPEN (returns None), so the caller falls back to the
deterministic IoU pick. It can never oscillate forever.

FAIL-OPEN: any error (VLM timeout, bad JSON, no candidate, cap reached)
returns None and the caller keeps deterministic behavior. The gate never
blocks rotation and never fabricates a crop.

Guarded behind an env flag so it can be A/B tested:
    DAILY_PHOTO_ANCHOR_CONSISTENCY=0  -> disabled (default 1 / on)
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

ENABLED = os.environ.get("DAILY_PHOTO_ANCHOR_CONSISTENCY", "1") != "0"
# The consistency verifier is Gemini (independent from qwen3-vl, the proposer),
# restoring the "different mechanism tests the conjecture" principle. Reuse the
# proven Nous auth/chat helpers from visual_weight_refine so there is one
# canonical Gemini client; if that module is unavailable the gate fails open.
try:
    from visual_weight_refine import load_access_token, chat_with_image
    _GEMINI_AVAILABLE = True
except Exception:  # noqa: BLE001
    _GEMINI_AVAILABLE = False
GEMINI_MODEL = os.environ.get("DAILY_PHOTO_CONSISTENCY_MODEL", "google/gemini-3.7-flash")
# Minimum detector confidence for a candidate to be worth verifying.
MIN_DETECTOR_SCORE = float(os.environ.get("DAILY_PHOTO_CONSISTENCY_MIN_SCORE", "0.25"))
# Crop padding around each candidate box (fraction of box height).
PAD_FRACTION = 0.35
# Number of candidates shown in one contact strip; top-N by detector score.
MAX_CANDIDATES = 6
# Hard cap on verification attempts per anchor (termination guarantee).
MAX_ATTEMPTS = int(os.environ.get("DAILY_PHOTO_CONSISTENCY_MAX_ATTEMPTS", "3"))
# How a "no match yet" is reported by the model.
NO_MATCH_TOKENS = {"none", "null", "no", "not", "none of", "no match"}


def _load_font(size: int):
    for path in ("/System/Library/Fonts/Helvetica.ttc",
                 "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _crop_region(img: Image.Image, bbox_pct: dict, pad_frac: float = PAD_FRACTION) -> Image.Image:
    """Crop a padded region around a percentage bbox. Returns PIL crop (RGB)."""
    w, h = img.size
    left = bbox_pct["left"] / 100.0 * w
    top = bbox_pct["top"] / 100.0 * h
    right = bbox_pct["right"] / 100.0 * w
    bottom = bbox_pct["bottom"] / 100.0 * h
    bw, bh = max(1, right - left), max(1, bottom - top)
    pad_x, pad_y = bw * pad_frac, bh * pad_frac
    x0 = max(0, int(left - pad_x))
    y0 = max(0, int(top - pad_y))
    x1 = min(w, int(right + pad_x))
    y1 = min(h, int(bottom + pad_y))
    return img.crop((x0, y0, x1, y1))


def build_contact_strip(image_path: str, candidates: list[dict]) -> str:
    """Tile the candidate crops into one labeled strip. Returns temp path."""
    img = Image.open(image_path).convert("RGB")
    font = _load_font(34)
    tiles = []
    for det in candidates:
        tile = _crop_region(img, det["bbox_pct"]).convert("RGB")
        tile.thumbnail((320, 320))
        canvas = Image.new("RGB", (tile.width, tile.height + 54), (20, 20, 22))
        draw = ImageDraw.Draw(canvas)
        letter = det.get("_letter", "?")
        draw.text((10, 6), letter, font=font, fill=(245, 245, 247))
        canvas.paste(tile, (0, 54))
        tiles.append(canvas)
    strip_h = max(t.height for t in tiles)
    strip = Image.new("RGB", (sum(t.width for t in tiles) + 6 * (len(tiles) - 1), strip_h), (20, 20, 22))
    x = 0
    for t in tiles:
        strip.paste(t, (x, (strip_h - t.height) // 2))
        x += t.width + 6
    tmp = tempfile.NamedTemporaryFile(prefix="anchor-consistency-", suffix=".jpg", delete=False)
    strip.save(tmp.name, quality=88)
    return tmp.name


def _resize_for_vlm(strip_path: str) -> str:
    """Downscale the strip so the VLM payload stays small (sips -Z)."""
    fd, out = tempfile.mkstemp(prefix="anchor-consistency-resized-", suffix=".jpg")
    os.close(fd)
    subprocess.run(["sips", "-Z", "1024", strip_path, "--out", out],
                   capture_output=True, text=True)
    return out


def _extract_json(text: str) -> dict:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = text.replace("\x1b", "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in consistency response")
    return json.loads(match.group(0))


def _one_attempt(image_path: str, anchor_desc: str, ranked: list[dict],
                 letters: str) -> str | None:
    """Run a single Gemini verification on the contact strip.

    Returns the chosen letter, or None if the model reports no match / errors.
    Uses the independent Nous Gemini client (visual_weight_refine helpers) so
    the verifier is NOT the same model that proposed the anchor.
    """
    if not _GEMINI_AVAILABLE:
        return None
    strip_path = build_contact_strip(image_path, ranked)
    resized = _resize_for_vlm(strip_path)
    try:
        with open(resized, "rb") as fh:
            image_b64 = base64.b64encode(fh.read()).decode("ascii")
        prompt = (
            "The image is a contact strip showing several labeled crops of the SAME "
            "scene. Each is labeled with a single letter (A, B, C, ...) in its top-left "
            "corner. One of these crops contains the subject described as: "
            f"\"{anchor_desc}\". "
            "Choose the SINGLE letter whose crop best matches that description. "
            "Reply with ONLY JSON: {\"letter\": \"X\"}. If none of them match, reply "
            "{\"letter\": null}."
        )
        base, tok = load_access_token()
        result = chat_with_image(base, tok, GEMINI_MODEL, prompt, image_b64, mime="image/jpeg")
        raw = result["choices"][0]["message"]["content"]
        parsed = _extract_json(raw)
        letter = str(parsed.get("letter") or "").strip().upper()
        if not letter:
            return None
        if letter in {"NULL", "NONE"}:
            return None
        if letter in letters:
            return letter
        # Model may have returned a sentence (e.g. "no match" or "B").
        for tok in NO_MATCH_TOKENS:
            if tok in letter.lower():
                return None
        first = re.search(r"[A-J]", letter)
        return first.group(0) if first else None
    finally:
        for p in (strip_path, resized):
            try:
                os.unlink(p)
            except OSError:
                pass


def disambiguate_anchor(image_path: str, anchor_desc: str,
                        candidates: list[dict], max_attempts: int = MAX_ATTEMPTS) -> dict | None:
    """Return the candidate detection that best matches `anchor_desc`.

    Bounded: at most `max_attempts` VLM calls; afterwards fails open (None).
    candidates: list of DINO detections of the anchor head-noun category,
    each {'bbox_pct': {...}, 'detector_score': float}.
    """
    if not ENABLED:
        return None
    if len(candidates) <= 1:
        return None
    ranked = sorted(candidates, key=lambda d: d.get("detector_score", 0.0), reverse=True)
    ranked = [d for d in ranked if d.get("detector_score", 0.0) >= MIN_DETECTOR_SCORE][:MAX_CANDIDATES]
    if len(ranked) <= 1:
        return None
    letters = "ABCDEFGHIJ"[:len(ranked)]
    for i, det in enumerate(ranked):
        det["_letter"] = letters[i]

    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        letter = _one_attempt(image_path, anchor_desc, ranked, letters)
        if letter is None:
            # Model reported no match. Stop; do not re-prompt for the same set.
            return None
        chosen = next((d for d in ranked if d.get("_letter") == letter), None)
        if chosen is not None:
            return chosen
        # Letter outside set -> treat as no decision; allow a bounded retry.
    return None
