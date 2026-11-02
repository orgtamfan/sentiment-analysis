"""Unit tests for prepare_labels module."""

import json
import tempfile
import unittest
from pathlib import Path

from src.prepare_labels import generate_label_suggestions


class TestPrepareLabels(unittest.TestCase):
    def test_invalid_json_format_raises_error(self):
        with tempfile.TemporaryDirectory() as td:
            in_file = Path(td) / "invalid.json"
            out_file = Path(td) / "review.csv"
            # Root JSON is a dictionary instead of a list of objects
            in_file.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain a list"):
                generate_label_suggestions(in_file, out_file)

    def test_json_with_non_dict_items_raises_error(self):
        with tempfile.TemporaryDirectory() as td:
            in_file = Path(td) / "invalid.json"
            out_file = Path(td) / "review.csv"
            # List contains numbers or strings instead of dict
            in_file.write_text(json.dumps(["comment1", 123]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be an object"):
                generate_label_suggestions(in_file, out_file)

    def test_unknown_game_raises_error(self):
        with tempfile.TemporaryDirectory() as td:
            in_file = Path(td) / "unknown_game.json"
            out_file = Path(td) / "review.csv"
            in_file.write_text(
                json.dumps([{"game": "ApexLegends", "comment_text": "Good game"}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported game value"):
                generate_label_suggestions(in_file, out_file)

    def test_empty_text_is_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            in_file = Path(td) / "comments.json"
            out_file = Path(td) / "review.csv"
            in_file.write_text(
                json.dumps([
                    {"game": "cs2", "comment_text": "   "},
                    {"game": "valorant", "comment_text": "Awesome clutch!"},
                ]),
                encoding="utf-8",
            )
            stats = generate_label_suggestions(in_file, out_file)
            self.assertEqual(stats["skipped_empty_text"], 1)
            self.assertEqual(stats["written"], 1)


if __name__ == "__main__":
    unittest.main()
