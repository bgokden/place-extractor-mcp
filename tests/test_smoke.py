"""Smoke tests for the MCP wiring.

These do NOT download the model — they verify the server, tools, and output
shape are wired correctly. Model inference is covered by a manual/integration
run (see README) since it requires downloading the weights.
"""

import asyncio

from place_extractor_mcp.server import extract_places, list_place_types, mcp


def test_server_name():
    assert mcp.name == "place-extractor"


def test_tools_registered():
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert {"extract_places", "list_place_types"} <= names


def test_list_place_types():
    types = {t["type"] for t in list_place_types()}
    assert types == {"CITY", "COUNTRY", "REGION", "ENTITY"}


def test_empty_text_shortcircuits():
    # Empty input must not attempt to load the model.
    out = extract_places("   ")
    assert out["places"] == []
    assert "text" in out
