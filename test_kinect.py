"""Regression checks against published Kinect coordinates, not training metrics."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from ui_prmd_preprocess.kinect import reconstruct_kinect, load_triplets, discover_kinect_files
from ui_prmd_preprocess.data_loader import load_csv_positions


class KinectTests(unittest.TestCase):
    def test_reconstruction_matches_published_absolute_coordinates(self):
        fixture = json.loads((Path(__file__).parent / "tests/fixtures/kinect_reference.json").read_text())
        rows = np.array(fixture["rows"])
        positions = rows[None, :, :3].copy()
        expected_input = positions.copy()
        actual = reconstruct_kinect(positions, rows[None, :, 3:6])
        np.testing.assert_allclose(actual[0], rows[:, 6:9], atol=1e-3, rtol=0)
        np.testing.assert_array_equal(positions, expected_input)

    def test_malformed_recordings_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vicon_angles.txt"
            np.savetxt(path, np.zeros((2, 117)))
            with self.assertRaises(ValueError):
                load_csv_positions(path)
            np.savetxt(path, np.zeros((2, 65)), delimiter=",")
            with self.assertRaises(ValueError):
                load_triplets(path)

    def test_subject_identity_and_label_come_from_original_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "incorrect/positions/m04_s07_e02_positions_inc.txt"
            path.parent.mkdir(parents=True)
            path.touch()
            entry = discover_kinect_files(root)[0]
            self.assertEqual((entry["subject"], entry["exercise"], entry["rep"], entry["label"]),
                             (7, 4, 2, 0))


if __name__ == "__main__":
    unittest.main()
