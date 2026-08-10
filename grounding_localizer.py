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


def canonical_category(label: str) -> str:
    """Map free-form VLM descriptions to stable detector categories.

    Grounding DINO is text-conditioned, but it is much more reliable when
    prompted with the object's head noun than with a scene-specific caption.
    Keep this vocabulary semantic and general: descriptions such as
    ``"vase with foliage"`` and ``"central palm tree"`` must reduce to
    categories that the detector can recognize across unrelated photos.
    """
    value = _normalise_label(label)
    tokens = set(value.split())

    if tokens & {"woman", "women"}:
        return "woman"
    if tokens & {"man", "men", "person", "people", "human"}:
        return "person"
    if tokens & {"tree", "trees", "palm", "palms"}:
        return "trees"
    if tokens & {"building", "buildings", "facade", "facades"}:
        return "building"
    if tokens & {"bollard", "bollards"}:
        return "bollards"
    if tokens & {"umbrella", "umbrellas", "parasol", "parasols"}:
        return "umbrella"
    if tokens & {"vase", "vases", "urn", "urns", "planter", "planters", "pot", "pots"}:
        return "vase"
    if tokens & {"plant", "plants", "foliage", "flower", "flowers", "floral", "vegetation"}:
        return "plant"

    # Generic fallback: remove common descriptive/relational words and
    # normalize simple plurals without relying on any particular test image.
    head_tokens = [token for token in value.split() if token not in {
        "a", "an", "the", "with", "in", "on", "at", "near", "central",
        "left", "right", "foreground", "background", "low", "high",
    }]
    if not head_tokens:
        return value
    head = head_tokens[-1]
    if len(head) > 3 and head.endswith("ies"):
        head = head[:-3] + "y"
    elif len(head) > 3 and head.endswith("s") and not head.endswith("ss"):
        head = head[:-1]
    return head


def _bbox_iou(first: dict[str, float], second: dict[str, float]) -> float:
    left = max(first["left"], second["left"])
    top = max(first["top"], second["top"])
    right = min(first["right"], second["right"])
    bottom = min(first["bottom"], second["bottom"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(1e-9, (first["right"] - first["left"]) * (first["bottom"] - first["top"]))
    second_area = max(1e-9, (second["right"] - second["left"]) * (second["bottom"] - second["top"]))
    return intersection / (first_area + second_area - intersection + 1e-9)


def match_detections_to_elements(elements: list[dict[str, Any]], detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign generic detector instances to semantic VLM elements by category and IoU."""
    remaining = list(detections)
    assigned = []
    for element in elements:
        category = canonical_category(element["name"])
        candidates = [item for item in remaining if item.get("category") == category]
        if not candidates:
            assigned.append(dict(element))
            continue
        best = max(candidates, key=lambda item: (_bbox_iou(element["bbox_pct"], item["bbox_pct"]), item["detector_score"]))
        remaining.remove(best)
        item = dict(element)
        item["bbox_pct"] = best["bbox_pct"]
        item["detector_score"] = best["detector_score"]
        assigned.append(item)
    return assigned


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
    result_labels = results.get("text_labels", results.get("labels", []))
    for box, score, result_label in zip(results["boxes"], results["scores"], result_labels):
        left, top, right, bottom = [float(value) for value in box.tolist()]
        detections.append({
            "name": _normalise_label(str(result_label)),
            "category": canonical_category(str(result_label)),
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
    wanted = {canonical_category(label) for label in labels}
    candidates = [item for item in detections if item.get("category", canonical_category(item["name"])) in wanted]
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
