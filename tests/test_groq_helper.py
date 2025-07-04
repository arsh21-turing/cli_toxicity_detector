import sys, os, json, shutil, tempfile, importlib
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

# Ensure project root in path *before* importing local modules
ROOT_DIR = Path(__file__).resolve().parent.parent
if ROOT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, ROOT_DIR.as_posix())

import pytest

from groq_cache import GroqCache
from model_loader import _groq_second_opinion, ToxicityCategory


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure each test starts without lingering GROQ_API_KEY."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    yield
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def _make_fake_groq(monkeypatch, prob=0.1):
    """Inject a stub groq module that returns deterministic scores."""

    class _FakeCompletionMessage:
        def __init__(self, payload):
            self.content = payload

    class _FakeChoice:
        def __init__(self, payload):
            self.message = _FakeCompletionMessage(payload)

    class _FakeCompletion:
        def __init__(self, payload):
            self.choices = [_FakeChoice(payload)]

    class _FakeCompletions:
        def __init__(self, payload):
            self._payload = payload

        def create(self, **_kw):  # type: ignore
            return _FakeCompletion(self._payload)

    class _FakeChat:
        def __init__(self, payload):
            self.completions = _FakeCompletions(payload)

    class _FakeClient:
        def __init__(self, *a, **k):
            self.chat = _FakeChat("{\"insult\": 0.2}")

    fake_module = ModuleType("groq")
    fake_module.Client = _FakeClient  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "groq", fake_module)


def test_groq_with_key(monkeypatch):
    """When GROQ_API_KEY is set and groq lib present, helper returns prob map."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    _make_fake_groq(monkeypatch)

    probs = _groq_second_opinion("some text")
    assert isinstance(probs, dict)
    assert all(isinstance(v, float) for v in probs.values())


def test_groq_without_key(monkeypatch):
    """No API key → helper returns None and does not error."""
    _make_fake_groq(monkeypatch)
    assert _groq_second_opinion("hello") is None


def test_groq_importerror(monkeypatch):
    """Missing groq module should be handled gracefully."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    # Ensure 'groq' import fails
    monkeypatch.setitem(sys.modules, "groq", None)
    assert _groq_second_opinion("hi") is None


# ---------------------------------------------------------------------------
# Extended cache lifecycle tests -------------------------------------------
# ---------------------------------------------------------------------------

def test_cache_roundtrip_and_cli_clear(monkeypatch):
    """First call → live, second → cache, clear flag wipes disk, third → live again."""

    # Use a temporary dir for cache so we don't touch user files
    tmpdir = tempfile.mkdtemp()
    monkeypatch.setenv("GROQ_API_KEY", "dummy-key")

    import model_loader as ml
    ml._groq_cache = GroqCache(tmpdir)

    # Create fake groq module
    fake_module = ModuleType("groq")
    fake_client = MagicMock()
    fake_module.Client = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "groq", fake_module)

    fake_json = json.dumps({"insult": 0.2})
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=MagicMock(content=fake_json))]
    fake_client.chat.completions.create.return_value = fake_completion

    text = "cache life-cycle sentence"

    # First call -> live
    ml._groq_cache.clear()
    _groq_second_opinion(text)
    assert fake_client.chat.completions.create.call_count == 1

    # Second call -> cache
    _groq_second_opinion(text)
    assert fake_client.chat.completions.create.call_count == 1  # unchanged

    # Clear cache programmatically -------------------------------------
    ml._groq_cache.clear()
    assert ml._groq_cache.get_cache_size() == 0

    # Third call -> live again
    result3 = _groq_second_opinion(text)
    assert fake_client.chat.completions.create.call_count == 2

    shutil.rmtree(tmpdir) 