"""Place Extractor MCP server.

Exposes multilingual place/location extraction as an MCP tool, backed by the
mDeBERTa token-classification models published at https://huggingface.co/Berk.

The heavy model dependencies (transformers / optimum / onnxruntime) are imported
lazily on first use, so the server process starts instantly and only loads the
model when a tool is actually called.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from mcp.server.fastmcp import FastMCP

# Default to the ONNX-optimized multilingual model; override with an env var.
DEFAULT_MODEL = os.environ.get(
    "PLACE_EXTRACTOR_MODEL",
    "Berk/multilingual-place-extractor-mdeberta-13lang-onnx-v7",
)

mcp = FastMCP("place-extractor")


@lru_cache(maxsize=1)
def _pipeline():
    """Build and cache the token-classification pipeline.

    Tries the ONNX runtime via `optimum` first (fast, low-memory); falls back to
    a plain 🤗 Transformers pipeline if optimum/onnxruntime is unavailable.
    """
    from transformers import AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    try:
        from optimum.onnxruntime import ORTModelForTokenClassification

        model = ORTModelForTokenClassification.from_pretrained(DEFAULT_MODEL)
        return pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )
    except Exception:
        # Fall back to letting transformers resolve the model directly.
        return pipeline(
            "token-classification",
            model=DEFAULT_MODEL,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )


@mcp.tool()
def extract_places(text: str, language: str | None = None) -> dict[str, Any]:
    """Extract place/location entities from text in 13 languages.

    Detects cities, countries, regions, and named entities (landmarks / points of
    interest) across Latin and non-Latin scripts using a multilingual mDeBERTa
    token-classification model.

    Args:
        text: The input text to analyze (any of the supported languages).
        language: Optional ISO language hint for your own bookkeeping. The model
            is multilingual and auto-detects, so this does not change results.

    Returns:
        A dict with the original text, the (optional) language hint, and a list of
        `places`, each: {text, type, score, start, end} where `type` is one of
        CITY, COUNTRY, REGION, ENTITY.
    """
    if not text or not text.strip():
        return {"text": text, "language": language, "places": []}

    results = _pipeline()(text)
    places = [
        {
            "text": r.get("word"),
            "type": r.get("entity_group"),
            "score": round(float(r.get("score", 0.0)), 4),
            "start": int(r["start"]) if r.get("start") is not None else None,
            "end": int(r["end"]) if r.get("end") is not None else None,
        }
        for r in results
    ]
    return {"text": text, "language": language, "places": places}


@mcp.tool()
def list_place_types() -> list[dict[str, str]]:
    """List the place entity types this server can extract."""
    return [
        {"type": "CITY", "description": "Cities and towns"},
        {"type": "COUNTRY", "description": "Countries and nations"},
        {"type": "REGION", "description": "Regions, states, provinces, areas"},
        {"type": "ENTITY", "description": "Landmarks and points of interest"},
    ]


def main() -> None:
    """Run the MCP server over stdio (the transport Claude Desktop expects)."""
    mcp.run()


if __name__ == "__main__":
    main()
