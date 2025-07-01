import sys
from pathlib import Path

# Ensure project root on path before importing project modules
ROOT_DIR = Path(__file__).resolve().parent.parent
if ROOT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, ROOT_DIR.as_posix())

import pytest

from categories import (
    ToxicityCategory,
    CATEGORY_DESCRIPTIONS,
    ALL_CATEGORIES,
    TOXIC_CATEGORIES,
    get_category_by_name,
    is_valid_category,
    get_all_categories,
    get_toxic_categories,
    get_category_description,
)


# ---------------------------------------------------------------------------
# Basic enum integrity
# ---------------------------------------------------------------------------

def test_enum_members():
    names = {c.name for c in ToxicityCategory}
    expected = {
        "INSULT",
        "HATE",
        "OBSCENE",
        "THREAT",
        "SEXUAL",
        "SELF_HARM",
        "NON_TOXIC",
    }
    assert names == expected, "Enum members mismatch"
    assert len(ToxicityCategory) == 7


def test_descriptions_exist():
    for cat in ToxicityCategory:
        assert cat in CATEGORY_DESCRIPTIONS, f"Missing description for {cat}"
        assert CATEGORY_DESCRIPTIONS[cat], "Description should not be empty"


def test_collections_consistency():
    assert set(ALL_CATEGORIES) == set(ToxicityCategory), "ALL_CATEGORIES mismatch"
    assert ToxicityCategory.NON_TOXIC not in TOXIC_CATEGORIES
    assert set(TOXIC_CATEGORIES) == set(ToxicityCategory) - {ToxicityCategory.NON_TOXIC}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def test_get_category_by_name():
    assert get_category_by_name("insult") is ToxicityCategory.INSULT
    assert get_category_by_name("Obscene") is ToxicityCategory.OBSCENE
    assert get_category_by_name("SELFHARM") is ToxicityCategory.SELF_HARM
    assert get_category_by_name("clean") is ToxicityCategory.NON_TOXIC
    assert get_category_by_name("invalid") is None


def test_is_valid_category():
    assert is_valid_category("hate")
    assert is_valid_category(ToxicityCategory.THREAT)
    assert not is_valid_category("unknown")
    assert not is_valid_category(123)  # type: ignore[arg-type]


def test_get_lists_helpers():
    assert get_all_categories() == ALL_CATEGORIES
    assert get_toxic_categories() == TOXIC_CATEGORIES


def test_get_category_description():
    desc = get_category_description(ToxicityCategory.SEXUAL)
    assert isinstance(desc, str) and desc
    # via string
    assert get_category_description("sexual") == desc
    assert get_category_description("invalid") is None 