import sys, os
from pathlib import Path

# Ensure project root is on sys.path when running pytest from tests directory
ROOT_DIR = Path(__file__).resolve().parent.parent
if ROOT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, ROOT_DIR.as_posix())

import pytest
from unittest.mock import patch

from model_loader import predict_toxicity, ToxicityCategory


@pytest.fixture()
def _mock_groq(monkeypatch):
    """Patch the _groq_second_opinion helper to return a fixed map that disagrees
    with the stub local model (which always predicts non-toxic)."""

    fake_map = {cat.name: 0.1 for cat in ToxicityCategory}
    fake_map["INSULT"] = 0.8  # make Groq think it's an insult
    fake_map["NON_TOXIC"] = 0.2

    monkeypatch.setattr("model_loader._groq_second_opinion", lambda _txt: fake_map)
    yield


test_text = "This sentence should trigger the Groq fallback."  # content irrelevant


@pytest.mark.parametrize(
    "policy,expected_source,expected_groq_used",
    [
        ("prefer-groq", "groq", True),
        ("prefer-local", "local", False),
        ("highest-confidence", "local", False),  # local distance 0.5 > 0.3 so local wins
    ],
)
def test_tie_policy_end_to_end(policy, expected_source, expected_groq_used, _mock_groq):
    res = predict_toxicity(
        texts=[test_text],
        allow_groq_fallback=True,
        gray_min=0.0,
        gray_max=1.0,
        tie_policy=policy,
    )[0]

    assert res["tie_source"] == expected_source, f"policy {policy} chose {res['tie_source']}"
    assert res["groq_used"] is expected_groq_used 