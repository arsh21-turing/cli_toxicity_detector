"""
Categories Module

This module defines the standard toxicity categories used throughout the application.
It serves as the single source of truth for all category-related information.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List, Optional, Union


class ToxicityCategory(Enum):
    """Enumeration of toxicity categories recognised by the application."""

    INSULT = auto()
    HATE = auto()
    OBSCENE = auto()
    THREAT = auto()
    SEXUAL = auto()
    SELF_HARM = auto()
    NON_TOXIC = auto()

    def __str__(self) -> str:  # pragma: no cover – utility
        return self.name


# ---------------------------------------------------------------------------
# Descriptions & collections -------------------------------------------------
# ---------------------------------------------------------------------------

CATEGORY_DESCRIPTIONS: Dict[ToxicityCategory, str] = {
    ToxicityCategory.INSULT: "Content that is rude, disrespectful, or belittles someone",
    ToxicityCategory.HATE: "Content expressing prejudice against protected characteristics",
    ToxicityCategory.OBSCENE: "Content that contains offensive or vulgar language",
    ToxicityCategory.THREAT: "Content expressing intent to inflict harm or violence",
    ToxicityCategory.SEXUAL: "Content with sexual references or implications",
    ToxicityCategory.SELF_HARM: "Content promoting, encouraging, or referring to self-harm",
    ToxicityCategory.NON_TOXIC: "Content that does not fall into any toxic category",
}

# List of toxic-only categories (excludes NON_TOXIC)
TOXIC_CATEGORIES: List[ToxicityCategory] = [
    ToxicityCategory.INSULT,
    ToxicityCategory.HATE,
    ToxicityCategory.OBSCENE,
    ToxicityCategory.THREAT,
    ToxicityCategory.SEXUAL,
    ToxicityCategory.SELF_HARM,
]

# Convenience – all categories in enum order
ALL_CATEGORIES: List[ToxicityCategory] = list(ToxicityCategory)


# ---------------------------------------------------------------------------
# Helper API -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def get_category_by_name(name: str) -> Optional[ToxicityCategory]:
    """Return enum value matching *name* (case-insensitive).

    Accepts a handful of common synonyms/variations – e.g. "selfharm", "clean".
    Returns *None* if no match found.
    """

    if not isinstance(name, str):
        return None

    normalised = name.strip().upper()
    try:
        return ToxicityCategory[normalised]
    except KeyError:
        synonyms = {
            "SELFHARM": ToxicityCategory.SELF_HARM,
            "SELF_HARM": ToxicityCategory.SELF_HARM,
            "NONTOXIC": ToxicityCategory.NON_TOXIC,
            "NON_TOXIC": ToxicityCategory.NON_TOXIC,
            "CLEAN": ToxicityCategory.NON_TOXIC,
        }
        return synonyms.get(normalised)


def is_valid_category(category: Union[str, ToxicityCategory]) -> bool:
    """Return *True* if *category* is a recognised toxicity category."""

    if isinstance(category, ToxicityCategory):
        return category in ToxicityCategory
    if isinstance(category, str):
        return get_category_by_name(category) is not None
    return False


def get_all_categories() -> List[ToxicityCategory]:
    """Return list of **all** categories."""

    return ALL_CATEGORIES.copy()


def get_toxic_categories() -> List[ToxicityCategory]:
    """Return list of toxic-only categories (i.e. excluding NON_TOXIC)."""

    return TOXIC_CATEGORIES.copy()


def get_category_description(category: Union[str, ToxicityCategory]) -> Optional[str]:
    """Return human-readable description for *category*.

    Accepts both enum instances and string names.
    Returns *None* if *category* unrecognised.
    """

    if isinstance(category, str):
        category = get_category_by_name(category)
    if isinstance(category, ToxicityCategory):
        return CATEGORY_DESCRIPTIONS.get(category)
    return None 