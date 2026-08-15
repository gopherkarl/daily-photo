"""Unit tests for visual_weight_refine geometry + fail-open behavior."""
import unittest

from crop_geometry import display_window
from visual_weight_refine import clamp_object, build_prompt


class ClampObjectTest(unittest.TestCase):
    def test_portrait_x_clamp_within_budget(self):
        # x-critical: current 50, max_dx 14.41, requested 40 -> clamped 40 (inside budget)
        x, y = clamp_object(50.0, 50.0, 40.0, None, 14.4112, 0.0)
        self.assertEqual(x, 40.0)
        self.assertEqual(y, 50.0)

    def test_portrait_x_clamp_caps_at_budget(self):
        # requested 20 exceeds 50-14.41=35.59 -> capped at 35.59
        x, _ = clamp_object(50.0, 50.0, 20.0, None, 14.4112, 0.0)
        self.assertAlmostEqual(x, 35.5888, places=3)

    def test_portrait_other_axis_fixed(self):
        # y not critical -> object_y stays, even if LLM suggests it
        x, y = clamp_object(50.0, 50.0, 45.0, 10.0, 14.4112, 0.0)
        self.assertEqual(x, 45.0)
        self.assertEqual(y, 50.0)

    def test_landscape_y_clamp(self):
        x, y = clamp_object(50.0, 100.0, None, 72.0, 0.0, 32.3786)
        self.assertEqual(x, 50.0)
        self.assertEqual(y, 72.0)

    def test_null_adjustment_keeps_current(self):
        x, y = clamp_object(50.0, 62.0, None, None, 14.4112, 32.3786)
        self.assertEqual((x, y), (50.0, 62.0))

    def test_does_not_exceed_budget_on_overshoot(self):
        x, y = clamp_object(50.0, 62.0, 0.0, 200.0, 14.4112, 32.3786)
        self.assertAlmostEqual(x, 35.5888, places=3)
        self.assertAlmostEqual(y, 94.3786, places=3)


class PromptContractTest(unittest.TestCase):
    def test_prompt_contains_json_contract(self):
        p = build_prompt("portrait", display_window(7629, 5077, 402, 874),
                         50.0, 50.0, 14.4112, 0.0, None)
        self.assertIn("adjust", p)
        self.assertIn("object_x", p)
        self.assertIn("10%-of-pixels", p)

    def test_prompt_anchor_line_when_present(self):
        p = build_prompt("landscape", display_window(7629, 5077, 874, 402),
                         50.0, 62.0, 0.0, 32.3786, {"left": 0.34, "top": 8.01})
        self.assertIn("Primary anchor bbox", p)


if __name__ == "__main__":
    unittest.main()
