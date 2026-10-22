"""Settings Vega applies when it starts the Ollama server."""

from __future__ import annotations

from vega import server


def test_prompt_cache_is_capped_for_a_16gb_machine() -> None:
    # llama.cpp's server defaults to an 8 GB RAM prompt cache; it filled during a looping task
    assert server.SERVER_DEFAULTS["LLAMA_ARG_CACHE_RAM"] == "1024"
