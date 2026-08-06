#!/usr/bin/env python3
"""Local text-conditioned localization using Hugging Face Grounding DINO-T."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from PIL import Image

MODEL_ID = os.environ.get(
    "DAILY_PHOTO_GROUNDING_MODEL",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".grounding-cache"),
)
BOX_THRESHOLD = float(os.environ.get("DAILY_PHOTO_BOX_THRESHOLD", "0.28"))
TEXT_THRESHOLD = float(os.environ.get("DAILY_PHOTO_TEXT_THRESHOLD", "0.20"))


@lru_cache(maxsize=1)
def _load_model():
    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Grounding DINO requires torch and transformers in the pipeline Python environment"
        ) from exc

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    return processor, model, device


def _normalise_label(label: str) -> str:
    return " ".join(str(label).lower().strip().replace(".", " ").split())


def locate(image_path: str, labels: list[str]) -> list[dict[str, Any]]:
    """Return detector boxes in percentage coordinates, grouped by label."""
    labels = list(dict.fromkeys(_normalise_label(label) for label in labels if str(label).strip()))
    if not labels:
        raise ValueError("Grounding DINO received no labels")

    processor, model, device = _load_model()
    import torch

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=[labels], return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )[0]

    width, height = image.size
    detections = []
    for box, score, result_label in zip(results["boxes"], results["scores"], results["labels"]):
        left, top, right, bottom = [float(value) for value in box.tolist()]
        detections.append({
            "name": _normalise_label(str(result_label)),
            "bbox_pct": {
                "left": round(max(0.0, min(100.0, left / width * 100)), 2),
                "top": round(max(0.0, min(100.0, top / height * 100)), 2),
                "right": round(max(0.0, min(100.0, right / width * 100)), 2),
                "bottom": round(max(0.0, min(100.0, bottom / height * 100)), 2),
            },
            "detector_score": round(float(score), 4),
        })
    return detections


def choose_best(detections: list[dict[str, Any]], labels: list[str]) -> dict[str, Any] | None:
    wanted = {_normalise_label(label) for label in labels}
    candidates = [item for item in detections if item["name"] in wanted]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["detector_score"])


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("labels", nargs="+")
    args = parser.parse_args()
    print(json.dumps(locate(args.image, args.labels), indent=2))
