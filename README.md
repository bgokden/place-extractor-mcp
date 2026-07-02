# place-extractor-mcp

[![CI](https://github.com/bgokden/place-extractor-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/bgokden/place-extractor-mcp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![MCP](https://img.shields.io/badge/MCP-server-8A2BE2)

An [MCP](https://modelcontextprotocol.io) server that gives any MCP client
(Claude Desktop, Claude Code, Cursor, …) a **multilingual place-extraction**
tool: pull cities, countries, regions, and landmarks out of text in **13
languages**, across Latin and non-Latin scripts.

It wraps the open-source **mDeBERTa** token-classification models published at
[huggingface.co/Berk](https://huggingface.co/Berk), using the ONNX-optimized
variant for fast, low-memory inference.

## Tools

| Tool | Description |
|------|-------------|
| `extract_places(text, language?)` | Returns detected places as `{text, type, score, start, end}`. `type` ∈ `CITY`, `COUNTRY`, `REGION`, `ENTITY`. |
| `list_place_types()` | Lists the entity types the model can extract. |

### Example

> **Prompt:** "Extract the places from: *Ik reis van Amsterdam via Berlijn naar İstanbul.*"

```json
{
  "text": "Ik reis van Amsterdam via Berlijn naar İstanbul.",
  "places": [
    {"text": "Amsterdam", "type": "CITY", "score": 0.99, "start": 12, "end": 21},
    {"text": "Berlijn",   "type": "CITY", "score": 0.99, "start": 26, "end": 33},
    {"text": "İstanbul",  "type": "CITY", "score": 0.99, "start": 40, "end": 48}
  ]
}
```

## Install

```bash
# with uv (recommended)
uvx place-extractor-mcp
# or with pip
pip install place-extractor-mcp
```

Or from source:

```bash
git clone https://github.com/bgokden/place-extractor-mcp
cd place-extractor-mcp
pip install -e .
```

## Use with Claude Desktop

Add to `claude_desktop_config.json`
(macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "place-extractor": {
      "command": "uvx",
      "args": ["place-extractor-mcp"]
    }
  }
}
```

(If installed with `pip`, use `"command": "place-extractor-mcp"` and drop `args`.)

Restart Claude Desktop and ask it to extract places from any text.

## Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `PLACE_EXTRACTOR_MODEL` | `Berk/multilingual-place-extractor-mdeberta-13lang-onnx-v7` | Swap in another compatible token-classification model / version. |

The first call downloads the model (cached afterward by 🤗 Hub). The heavy
dependencies are imported lazily, so the server starts instantly.

## Entity types

- **CITY** — cities and towns
- **COUNTRY** — countries and nations
- **REGION** — regions, states, provinces, areas
- **ENTITY** — landmarks and points of interest

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT © [Berk Gökden](https://berkgokden.com) · models © the same author at
[huggingface.co/Berk](https://huggingface.co/Berk)
