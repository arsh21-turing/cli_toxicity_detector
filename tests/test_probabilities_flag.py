#!/usr/bin/env python3
"""
Test for the --probabilities flag in the toxicity detection CLI.
Verifies that all expected labels appear with numeric probability scores in both
plain-text and JSON modes.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path
from typing import List


class ProbabilitiesFlagTest(unittest.TestCase):
    expected_labels: List[str] = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    test_sentence = "This is a test sentence to verify probabilities output."

    # Helper to call the CLI with the active Python interpreter
    def _run_cli(self, extra_args: List[str]) -> subprocess.CompletedProcess[str]:
        root = Path(__file__).resolve().parent.parent
        cmd = [sys.executable, str(root / "main.py"), *extra_args]
        return subprocess.run(cmd, capture_output=True, text=True, check=True)

    # ------------------------------------------------------------------
    def test_probabilities_flag_output(self) -> None:
        res = self._run_cli(["--text", self.test_sentence, "--probabilities", "--quiet"])
        output = res.stdout
        # verify every label with numeric score between 0 and 1
        for label in self.expected_labels:
            pattern = rf"\b{re.escape(label)}:\s+([0-1]?\.\d+)"
            m = re.search(pattern, output)
            self.assertIsNotNone(m, f"Label '{label}' with score not found in output")
            score = float(m.group(1))
            self.assertTrue(0 <= score <= 1, f"Score for '{label}' out of bounds: {score}")

    # ------------------------------------------------------------------
    def test_probabilities_with_json_output(self) -> None:
        res = self._run_cli([
            "--text",
            self.test_sentence,
            "--probabilities",
            "--json",
            "--quiet",
        ])
        data = json.loads(res.stdout)
        self.assertIn("raw_probabilities", data, "raw_probabilities missing in JSON output")
        probs = data["raw_probabilities"]
        for label in self.expected_labels:
            self.assertIn(label, probs, f"{label} missing in raw_probabilities")
            score = probs[label]
            self.assertTrue(isinstance(score, (int, float)), f"{label} score not numeric")
            self.assertTrue(0 <= score <= 1, f"Score for '{label}' out of bounds: {score}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()