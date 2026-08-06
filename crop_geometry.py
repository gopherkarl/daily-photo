#!/usr/bin/env python3
"""Display geometry and subject-bounding-box crop calculations."""


def display_window(photo_w, photo_h, viewport_w, viewport_h):
    scale = max(viewport_w / photo_w, viewport_h / photo_h)
    displayed_w = photo_w * scale
    displayed_h = photo_h * scale
    visible_w = viewport_w / scale
    visible_h = viewport_h / scale
    overflow_x = max(0.0, photo_w - visible_w)
    overflow_y = max(0.0, photo_h - visible_h)
    return {
        "photo_width": photo_w,
        "photo_height": photo_h,
        "viewport_width": viewport_w,
        "viewport_height": viewport_h,
        "visible_width": visible_w,
        "visible_height": visible_h,
        "visible_fraction_x": visible_w / photo_w,
        "visible_fraction_y": visible_h / photo_h,
        "overflow_x": overflow_x,
        "overflow_y": overflow_y,
        "x_critical": overflow_x > 1e-6,
        "y_critical": overflow_y > 1e-6,
    }


def clamp(value, low=0.0, high=100.0):
    return max(low, min(high, value))


def crop_position_for_bbox(window, bbox):
    """Return CSS object-position percentages that center a bbox when possible.

    object-position distributes overflow, so the CSS percentage is not the same
    as the subject's source-image percentage.
    """
    left, top, right, bottom = [float(bbox[k]) for k in ("left", "top", "right", "bottom")]
    subject_x = (left + right) / 2.0
    subject_y = (top + bottom) / 2.0
    if window["overflow_x"] > 1e-6:
        x = clamp((subject_x - window["visible_width"] / 2) / window["overflow_x"] * 100)
    else:
        x = 50.0
    if window["overflow_y"] > 1e-6:
        y = clamp((subject_y - window["visible_height"] / 2) / window["overflow_y"] * 100)
    else:
        y = 50.0
    return round(x), round(y)


def visible_source_window(window, object_x, object_y):
    left = (object_x / 100.0) * window["overflow_x"]
    top = (object_y / 100.0) * window["overflow_y"]
    return {
        "left": left,
        "top": top,
        "right": left + window["visible_width"],
        "bottom": top + window["visible_height"],
    }


def bbox_inside_window(bbox, visible):
    return (
        bbox["left"] >= visible["left"] - 1e-6
        and bbox["right"] <= visible["right"] + 1e-6
        and bbox["top"] >= visible["top"] - 1e-6
        and bbox["bottom"] <= visible["bottom"] + 1e-6
    )


def bbox_overlap_fraction(bbox, visible):
    left = max(bbox["left"], visible["left"])
    top = max(bbox["top"], visible["top"])
    right = min(bbox["right"], visible["right"])
    bottom = min(bbox["bottom"], visible["bottom"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area = max(1e-9, (bbox["right"] - bbox["left"]) * (bbox["bottom"] - bbox["top"]))
    return intersection / area


def bbox_percent_to_pixels(bbox_pct, photo_w, photo_h):
    return {
        "left": bbox_pct["left"] / 100 * photo_w,
        "top": bbox_pct["top"] / 100 * photo_h,
        "right": bbox_pct["right"] / 100 * photo_w,
        "bottom": bbox_pct["bottom"] / 100 * photo_h,
    }


def percent_window(window, object_x, object_y):
    visible = visible_source_window(window, object_x, object_y)
    return {key: round(value / (window["photo_width"] if key in ("left", "right") else window["photo_height"]) * 100, 2) for key, value in visible.items()}


def union_bboxes(bboxes):
    if not bboxes:
        return None
    return {
        "left": min(box["left"] for box in bboxes),
        "top": min(box["top"] for box in bboxes),
        "right": max(box["right"] for box in bboxes),
        "bottom": max(box["bottom"] for box in bboxes),
    }


def candidate_crop(window, anchor_bbox, context_bboxes=(), anomaly_bbox=None):
    """Center a candidate around the anchor plus requested supporting boxes."""
    boxes = [anchor_bbox, *context_bboxes]
    if anomaly_bbox:
        boxes.append(anomaly_bbox)
    target = union_bboxes(boxes)
    x, y = crop_position_for_bbox(window, target)
    visible = visible_source_window(window, x, y)
    return {
        "object_x": x,
        "object_y": y,
        "visible_source_window": percent_window(window, x, y),
        "anchor_inside": bbox_inside_window(anchor_bbox, visible),
        "anchor_overlap": round(bbox_overlap_fraction(anchor_bbox, visible), 4),
        "context_inside": [bbox_inside_window(box, visible) for box in context_bboxes],
        "context_overlap": [round(bbox_overlap_fraction(box, visible), 4) for box in context_bboxes],
        "anomaly_inside": bbox_inside_window(anomaly_bbox, visible) if anomaly_bbox else False,
        "target_bbox": target,
    }


def score_candidate(candidate, centroid, anomaly_present=True):
    context_score = (sum(candidate.get("context_overlap", [])) / len(candidate["context_overlap"])) if candidate.get("context_overlap") else 0.0
    centroid_x = abs(candidate["object_x"] - centroid["x"]) / 100.0
    return round(
        0.40 * float(candidate["anchor_inside"])
        + 0.25 * context_score
        + 0.15 * (float(candidate["anomaly_inside"]) if anomaly_present else 0.0)
        + 0.10 * (1.0 - centroid_x)
        - 0.50 * (0.0 if candidate["anchor_inside"] else 1.0)
        - 0.25 * (0.0 if context_score > 0 else 1.0),
        4,
    )
