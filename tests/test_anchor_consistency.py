import unittest
from unittest import mock

import anchor_consistency as ac


class AnchorConsistencyTests(unittest.TestCase):
    def test_returns_none_when_disabled(self):
        with mock.patch.object(ac, "ENABLED", False):
            self.assertIsNone(ac.disambiguate_anchor("x.jpg", "desc", [{"bbox_pct": {}, "detector_score": 0.9}]))

    def test_returns_none_with_single_candidate(self):
        # No ambiguity -> no gate needed.
        self.assertIsNone(ac.disambiguate_anchor("x.jpg", "desc",
                                                 [{"bbox_pct": {}, "detector_score": 0.9}]))

    def test_returns_none_when_candidates_below_min_score(self):
        # All below threshold -> no verifiable ambiguity.
        with mock.patch.object(ac, "MIN_DETECTOR_SCORE", 0.25):
            self.assertIsNone(ac.disambiguate_anchor(
                "x.jpg", "desc",
                [{"bbox_pct": {"left": 1, "top": 1, "right": 2, "bottom": 2}, "detector_score": 0.1},
                 {"bbox_pct": {"left": 3, "top": 1, "right": 4, "bottom": 2}, "detector_score": 0.2}]))

    def test_bounded_attempts_terminates(self):
        # The VLM keeps returning an out-of-set letter; the gate must stop
        # after max_attempts and return None, never loop forever.
        candidates = [
            {"bbox_pct": {"left": 0, "top": 0, "right": 10, "bottom": 90}, "detector_score": 0.8},
            {"bbox_pct": {"left": 20, "top": 0, "right": 30, "bottom": 90}, "detector_score": 0.7},
            {"bbox_pct": {"left": 40, "top": 0, "right": 50, "bottom": 90}, "detector_score": 0.6},
        ]
        with mock.patch.object(ac, "MAX_ATTEMPTS", 3), \
             mock.patch.object(ac, "MIN_DETECTOR_SCORE", 0.5), \
             mock.patch.object(ac, "_one_attempt", return_value="Q") as m:  # out-of-set letter every time
            result = ac.disambiguate_anchor("x.jpg", "desc", candidates)
        self.assertIsNone(result)
        self.assertEqual(m.call_count, 3)

    def test_returns_chosen_candidate_when_vlm_matches(self):
        candidates = [
            {"bbox_pct": {"left": 0, "top": 0, "right": 10, "bottom": 90}, "detector_score": 0.8},
            {"bbox_pct": {"left": 20, "top": 0, "right": 30, "bottom": 90}, "detector_score": 0.7},
            {"bbox_pct": {"left": 40, "top": 0, "right": 50, "bottom": 90}, "detector_score": 0.6},
        ]
        with mock.patch.object(ac, "MIN_DETECTOR_SCORE", 0.5), \
             mock.patch.object(ac, "_one_attempt", return_value="B"):
            chosen = ac.disambiguate_anchor("x.jpg", "desc", candidates)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["bbox_pct"]["left"], 20)

    def test_returns_none_when_vlm_reports_no_match(self):
        candidates = [
            {"bbox_pct": {"left": 0, "top": 0, "right": 10, "bottom": 90}, "detector_score": 0.8},
            {"bbox_pct": {"left": 20, "top": 0, "right": 30, "bottom": 90}, "detector_score": 0.7},
        ]
        with mock.patch.object(ac, "MIN_DETECTOR_SCORE", 0.5), \
             mock.patch.object(ac, "_one_attempt", return_value=None) as m:
            chosen = ac.disambiguate_anchor("x.jpg", "desc", candidates)
        self.assertIsNone(chosen)
        # No-match stops immediately; must not retry the same set.
        self.assertEqual(m.call_count, 1)

    def test_ranks_by_detector_score_top_n(self):
        candidates = [
            {"bbox_pct": {"left": i, "top": 0, "right": i + 1, "bottom": 90}, "detector_score": s}
            for i, s in enumerate([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        ]
        with mock.patch.object(ac, "MIN_DETECTOR_SCORE", 0.25), \
             mock.patch.object(ac, "MAX_CANDIDATES", 6), \
             mock.patch.object(ac, "_one_attempt", return_value="A"):
            chosen = ac.disambiguate_anchor("x.jpg", "desc", candidates)
        self.assertEqual(chosen["detector_score"], 0.9)


if __name__ == "__main__":
    unittest.main()

# pyright: ignore
