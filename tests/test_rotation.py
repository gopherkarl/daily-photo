import json
import tempfile
import unittest
from pathlib import Path

import rotate


class RotationTests(unittest.TestCase):
    def test_selection_is_alphabetical_and_skips_history(self):
        photos = ["01.jpg", "02.jpg", "03.jpg"]
        selected, state = rotate.select_photo(
            photos, {"last_shown": "01.jpg", "history": ["01.jpg"]}
        )
        self.assertEqual(selected, "02.jpg")
        self.assertEqual(state["last_shown"], "02.jpg")

    def test_cycle_reset_avoids_immediate_repeat(self):
        photos = ["01.jpg", "02.jpg"]
        selected, state = rotate.select_photo(
            photos, {"last_shown": "02.jpg", "history": ["01.jpg", "02.jpg"]}
        )
        self.assertEqual(selected, "01.jpg")
        self.assertEqual(state["history"], ["02.jpg", "01.jpg"])

    def test_state_write_is_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            rotate.save_state({"last_shown": "03.jpg", "history": ["03.jpg"]}, str(path))
            with path.open() as handle:
                self.assertEqual(json.load(handle)["last_shown"], "03.jpg")


if __name__ == "__main__":
    unittest.main()

# pyright: ignore
