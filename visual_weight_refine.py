#!/usr/bin/env python3
"""Production visual-weight crop refinement.

After the deterministic crop decision in local_analyze.main(), optionally call a
frontier vision LLM (Nous google/gemini-3.7-flash by default) to make a SMALL,
bounded pan for better composition. The +/-10%-of-pixels clamp is the only safety
guard; the anchor re-check is informational and is NOT a rejection gate.

FAIL-OPEN: every callable here raises on error; the caller (local_analyze.main)
catches and keeps the deterministic positions. This module never blocks or breaks
the daily rotation because of a transient network/auth/model issue.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

from crop_geometry import (
    display_window,
    visible_source_window,
    percent_window,
)

AUTH_PATH = Path(os.path.expanduser("~/.hermes/auth.json"))
DEFAULT_MODEL = os.environ.get("DAILY_PHOTO_VISUAL_WEIGHT_MODEL", "google/gemini-3.7-flash")
CALL_TIMEOUT = float(os.environ.get("DAILY_PHOTO_VISUAL_WEIGHT_TIMEOUT", "60"))

VISUAL_WEIGHT_GUIDE = """You are a photographic composition judge. Use visual-weight
principles to decide whether the current crop window can be improved by a SMALL,
bounded pan. Judge composition as competing visual forces, not just subject matter:

Perceptual tests first: squint (which large shapes / contrast junctions / saturated
blobs dominate?) and flip the frame upside-down mentally to strip semantic bias.

Nine visual-weight drivers (heavy vs light):
1. People & Animals  : faces/eyes/figures = heaviest; landscapes/texture = light.
2. Sharpness & Focus : crisp subjects on the focal plane vs soft bokeh.
3. Contrast & Tone   : dark-on-bright (or reverse) junction = strong highlight.
4. Location          : off-center / top-right holds more weight than center.
5. Color Saturation  : vibrant warm hues (red/orange/yellow) are heavy.
6. Framing/Isolation : isolated subject in negative space is amplified.
7. Size & Scale      : large dominant forms draw the eye.
8. Shape Complexity  : intricate/geometric silhouettes stand out.
9. Quantity          : a single unique element amid repetition is the anomaly.

Balance via the fulcrum rule: a small heavy element near center can be balanced by a
much larger light element near the edge; a small element near the right boundary can
balance a larger central-left shape. Perfect symmetry is static; slight asymmetry
(a rule-of-thirds placement) is more dynamic. Give directional subjects leading space
(room to move/look into). Dense repeating crowds read as low-weight background texture.

The crop window can only pan along the CRITICAL axis, and by at most the given max
delta. If the current composition is already good, return adjust:false. Never invent a
better composition that does not preserve the primary anchor."""


def load_access_token() -> tuple[str, str]:
    """Return (inference_base_url, access_token) from the Nous OAuth store."""
    if not AUTH_PATH.exists():
        raise RuntimeError(f"no auth store at {AUTH_PATH}")
    data = json.loads(AUTH_PATH.read_text())
    creds = data.get("credential_pool", {}).get("nous", [])
    if not creds:
        raise RuntimeError("no nous credential in auth.json")
    base = creds[0].get("inference_base_url") or "https://inference-api.nousresearch.com/v1"
    tok = creds[0].get("access_token")
    if not tok:
        raise RuntimeError("nous access_token missing")
    return base, tok


def chat_with_image(base: str, tok: str, model: str, prompt: str, image_b64: str,
                    mime: str = "image/png", max_tokens: int = 2000) -> dict:
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
            ],
        }],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {tok}",
            "User-Agent": "hermes-visual-weight-refine",
        },
    )
    with urllib.request.urlopen(req, timeout=CALL_TIMEOUT) as r:
        return json.load(r)


def build_overlay_input(source_path: str, window: dict, object_x: float,
                        object_y: float, anchor_px: dict | None) -> tuple[str, str]:
    """Downscale the source to ~1024px and dim everything outside the crop window.

    Returns (path, b64_data_url). Cache-busted unique path so the gateway never
    serves a stale frame.
    """
    source = Image.open(source_path).convert("RGB")
    target_w = 1024
    scale = target_w / source.width
    canvas = source.resize((target_w, int(source.height * scale)), Image.LANCZOS).convert("RGBA")

    vis = visible_source_window(window, object_x, object_y)
    box_px = {
        "left": vis["left"] * scale,
        "top": vis["top"] * scale,
        "right": vis["right"] * scale,
        "bottom": vis["bottom"] * scale,
    }
    mask = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    md = ImageDraw.Draw(mask)
    md.rectangle([0, 0, canvas.width - 1, canvas.height - 1], fill=(0, 0, 0, 168))
    md.rectangle([box_px["left"], box_px["top"], box_px["right"], box_px["bottom"]],
                 fill=(0, 0, 0, 0))
    canvas = Image.alpha_composite(canvas, mask)

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([box_px["left"], box_px["top"], box_px["right"], box_px["bottom"]],
                   outline=(0, 255, 255), width=7)
    if anchor_px:
        ab = {k: v * scale for k, v in anchor_px.items()}
        draw.rectangle([ab["left"], ab["top"], ab["right"], ab["bottom"]],
                       outline=(255, 40, 40), width=5)

    outdir = Path("/tmp/vw_refine")
    outdir.mkdir(exist_ok=True)
    path = outdir / f"vw_{int(time.time()*1000)}_{os.getpid()}_{id(canvas)%100000}.png"
    canvas.convert("RGB").save(path)
    return str(path), base64.b64encode(Path(path).read_bytes()).decode()


def build_prompt(profile: str, window: dict, object_x: float, object_y: float,
                 max_dx: float, max_dy: float, anchor_pct: dict | None) -> str:
    crit = "X (horizontal)" if max_dx > 0 else "Y (vertical)"
    ad = max_dx if max_dx > 0 else max_dy
    win = percent_window(window, object_x, object_y)
    anchor_line = ""
    if anchor_pct:
        anchor_line = (
            f"Primary anchor bbox in source %% coords: {anchor_pct}. "
            "It MUST remain fully inside the crop window."
        )
    return (
        VISUAL_WEIGHT_GUIDE
        + "\n\n"
        + "TASK:\n"
        + f"The image shows the FULL source frame with the current {profile} crop window "
        + "drawn as a bright CYAN rectangle; everything outside it is DARKENED. The red "
        + "box (if present) is the primary anchor.\n"
        + f"Current crop window in source %%: {win}. Current object-position: "
        + f"x={object_x}% y={object_y}%.\n"
        + f"You may pan along the {crit} axis only, by at most +/-{round(ad,1)} "
        + "object-position points (the 10%-of-pixels budget). The other axis stays fixed.\n"
        + anchor_line
        + "\n\nReply ONLY with JSON:\n"
        + '{"adjust": true|false, "object_x": <number or null>, '
        + '"object_y": <number or null>, "reason": "<1-2 sentences>"}\n'
        + "If adjust:false, object_x/object_y must be null. Never exceed the max delta; "
        + "never move the anchor out of the window."
    )


def clamp_object(current_x: float, current_y: float, adj_x, adj_y,
                 max_dx: float, max_dy: float) -> tuple[float, float]:
    x, y = current_x, current_y
    if adj_x is not None and max_dx > 0:
        x = max(current_x - max_dx, min(current_x + max_dx, float(adj_x)))
    if adj_y is not None and max_dy > 0:
        y = max(current_y - max_dy, min(current_y + max_dy, float(adj_y)))
    return round(x, 4), round(y, 4)


def _extract_json(content: str) -> dict:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object in model response")
    return json.loads(content[start:end + 1])


def refine_profile(source_path: str, profile: str, viewport_w: int, viewport_h: int,
                   photo_w: int, photo_h: int, object_x: float, object_y: float,
                   anchor_px: dict | None, anchor_pct: dict | None,
                   model: str = DEFAULT_MODEL) -> dict:
    """Run one judge call for a single profile. Returns a result dict; raises on error."""
    window = display_window(photo_w, photo_h, viewport_w, viewport_h)
    max_dx = 0.10 * photo_w / window["overflow_x"] * 100 if window["x_critical"] else 0.0
    max_dy = 0.10 * photo_h / window["overflow_y"] * 100 if window["y_critical"] else 0.0

    path, b64 = build_overlay_input(source_path, window, object_x, object_y, anchor_px)
    prompt = build_prompt(profile, window, object_x, object_y, max_dx, max_dy, anchor_pct)
    base, tok = load_access_token()
    raw = chat_with_image(base, tok, model, prompt, b64)
    content = raw["choices"][0]["message"]["content"]
    parsed = _extract_json(content)

    adjust = bool(parsed.get("adjust", False))
    new_x, new_y = clamp_object(object_x, object_y, parsed.get("object_x"),
                                parsed.get("object_y"), max_dx, max_dy)
    moved = abs(new_x - object_x) > 1e-6 or abs(new_y - object_y) > 1e-6
    return {
        "profile": profile,
        "model": model,
        "judge_adjust": adjust,
        "judge_reason": parsed.get("reason", ""),
        "judge_object_x": parsed.get("object_x"),
        "judge_object_y": parsed.get("object_y"),
        "max_delta_x_points": round(max_dx, 4),
        "max_delta_y_points": round(max_dy, 4),
        "clamped_object_x": new_x,
        "clamped_object_y": new_y,
        "applied": adjust and moved,
        "cost": raw.get("usage", {}).get("cost"),
        "usage": raw.get("usage"),
    }


def refine_positions(source_path: str, positions: dict, profiles: dict,
                     anchor_px: dict | None, anchor_pct: dict | None,
                     model: str = DEFAULT_MODEL) -> tuple[dict, list]:
    """Refine portrait/landscape positions. Returns (new_positions, per-profile log).

    Raises on error so the caller can fail open. Any profile whose call raises aborts
    the whole refinement (caller keeps ALL deterministic positions) -- consistency is
    preferred over partial application.
    """
    photo = Image.open(source_path)
    photo_w, photo_h = photo.size
    log = []
    new_positions = dict(positions)
    for name, prof in (("portrait", profiles["portrait_phone"]),
                       ("landscape", profiles["landscape_phone"])):
        res = refine_profile(
            source_path, name, prof["width"], prof["height"], photo_w, photo_h,
            float(positions[f"{name}_x"]), float(positions[f"{name}_y"]),
            anchor_px, anchor_pct, model,
        )
        log.append(res)
        if res["applied"]:
            new_positions[f"{name}_x"] = res["clamped_object_x"]
            new_positions[f"{name}_y"] = res["clamped_object_y"]
    return new_positions, log
