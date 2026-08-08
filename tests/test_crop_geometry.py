import unittest

from crop_geometry import (
    bbox_inside_window,
    bbox_percent_to_pixels,
    crop_position_for_bbox,
    display_window,
    visible_source_window,
)


class CropGeometryTests(unittest.TestCase):
    def test_landscape_portrait_phone_clips_width(self):
        window = display_window(1924, 1281, 402, 874)
        self.assertAlmostEqual(window["visible_fraction_x"], 0.306, places=2)
        self.assertAlmostEqual(window["visible_fraction_y"], 1.0, places=2)
        self.assertTrue(window["x_critical"])
        self.assertFalse(window["y_critical"])

    def test_bbox_crop_is_inside_visible_window(self):
        window = display_window(6654, 4436, 402, 874)
        bbox = bbox_percent_to_pixels(
            {"left": 72, "top": 42, "right": 84, "bottom": 58}, 6654, 4436
        )
        x, y = crop_position_for_bbox(window, bbox)
        visible = visible_source_window(window, x, y)
        self.assertTrue(bbox_inside_window(bbox, visible))

    def test_portrait_source_clips_height(self):
        window = display_window(600, 1600, 402, 874)
        self.assertTrue(window["y_critical"])
        self.assertFalse(window["x_critical"])

    def test_wide_subject_can_fail_but_anomaly_can_fit(self):
        window = display_window(1024, 1280, 402, 874)
        crowd = bbox_percent_to_pixels(
            {"left": 0, "top": 0, "right": 80, "bottom": 100}, 1024, 1280
        )
        player = bbox_percent_to_pixels(
            {"left": 74, "top": 55, "right": 88, "bottom": 70}, 1024, 1280
        )
        x, y = crop_position_for_bbox(window, player)
        visible = visible_source_window(window, x, y)
        self.assertFalse(bbox_inside_window(crowd, visible))
        self.assertTrue(bbox_inside_window(player, visible))


if __name__ == "__main__":
    unittest.main()

# pyright: ignore
