"""Basic tests for synthetic dataset generation."""

import csv
import tempfile
import unittest
from pathlib import Path

from src.data_generator import generate_data


class DataGeneratorTests(unittest.TestCase):
    """Check output files and their requested row counts."""

    def test_generate_data_writes_expected_rows(self) -> None:
        """Each generated CSV has a header and the requested number of records."""
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_data(3, 2, 5, Path(directory), seed=10)
            expected = {"customers": 3, "products": 2, "orders": 5}
            for dataset, count in expected.items():
                with paths[dataset].open(newline="", encoding="utf-8") as source:
                    self.assertEqual(len(list(csv.DictReader(source))), count)


if __name__ == "__main__":
    unittest.main()
