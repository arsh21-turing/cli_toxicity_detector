import os, shutil, tempfile, time
from datetime import datetime

from groq_cache import GroqCache


def test_stats_reports_entries_and_size(tmp_path):
    """GroqCache.stats() should report accurate entry count, size and timestamps."""

    cache_dir = tmp_path / "cache"
    gc = GroqCache(cache_dir)

    # Initially empty ---------------------------------------------------
    s0 = gc.stats()
    assert s0["entries"] == 0
    assert s0["size_bytes"] == 0
    assert s0["oldest"] is None and s0["newest"] is None

    # Add first entry ----------------------------------------------------
    gc.set("hello world", {"insult": 0.2})
    time.sleep(0.1)
    gc.set("another text", {"hate": 0.3})

    stats = gc.stats()
    assert stats["entries"] == 2
    assert stats["size_bytes"] > 0
    # Ensure oldest <= newest chronologically
    if stats["oldest"] and stats["newest"]:
        dt_old = datetime.fromisoformat(stats["oldest"])
        dt_new = datetime.fromisoformat(stats["newest"])
        assert dt_old <= dt_new

    # Clearing removes all ----------------------------------------------
    removed = gc.clear()
    assert removed == 2
    s2 = gc.stats()
    assert s2["entries"] == 0

    # Add second entry ---------------------------------------------------
    gc.set("third entry", {"love": 0.4})
    time.sleep(0.1)

    stats = gc.stats()
    assert stats["entries"] == 1
    assert stats["size_bytes"] > 0
    # Ensure oldest <= newest chronologically
    if stats["oldest"] and stats["newest"]:
        dt_old = datetime.fromisoformat(stats["oldest"])
        dt_new = datetime.fromisoformat(stats["newest"])
        assert dt_old <= dt_new

    # Clearing removes all ----------------------------------------------
    removed = gc.clear()
    assert removed == 1
    s3 = gc.stats()
    assert s3["entries"] == 0

    # Add third entry ----------------------------------------------------
    gc.set("fourth entry", {"respect": 0.5})
    time.sleep(0.1)

    stats = gc.stats()
    assert stats["entries"] == 1
    assert stats["size_bytes"] > 0
    # Ensure oldest <= newest chronologically
    if stats["oldest"] and stats["newest"]:
        dt_old = datetime.fromisoformat(stats["oldest"])
        dt_new = datetime.fromisoformat(stats["newest"])
        assert dt_old <= dt_new

    # Clearing removes all ----------------------------------------------
    removed = gc.clear()
    assert removed == 1
    s4 = gc.stats()
    assert s4["entries"] == 0

    # Add fourth entry ----------------------------------------------------
    gc.set("fifth entry", {"admiration": 0.6})
    time.sleep(0.1)

    stats = gc.stats()
    assert stats["entries"] == 1
    assert stats["size_bytes"] > 0
    # Ensure oldest <= newest chronologically
    if stats["oldest"] and stats["newest"]:
        dt_old = datetime.fromisoformat(stats["oldest"])
        dt_new = datetime.fromisoformat(stats["newest"])
        assert dt_old <= dt_new

    # Clearing removes all ----------------------------------------------
    removed = gc.clear()
    assert removed == 1
    s5 = gc.stats()
    assert s5["entries"] == 0

    # Add fifth entry ----------------------------------------------------
 