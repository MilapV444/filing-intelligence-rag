"""Single configuration point for the project.

NFR-8 requires model identifiers to live in exactly one place, so that swapping a
model is one edit rather than a grep. NFR-4 requires credentials to come only from
the environment, so the key is never read from a file and never returned by any
function that gets logged or serialised.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
# The PDFs themselves live where the human put them. The manifest stays in
# corpus/ under version control; the documents are gitignored and never
# redistributed, only recorded with the public URL they came from.
DOCUMENTS_DIR = ROOT / "sample_pdf"
INDEX_DIR = ROOT / "index"
NOTES_DIR = ROOT / "notes"

# --- Models ------------------------------------------------------------------
# DECISION-39162611: high-frequency, short-output, structurally simple calls run
# on Haiku 4.5; the reasoning the demo is judged on runs on Opus 5. This split is
# the lever that makes the NFR-1 cost ceiling reachable. It is a starting point,
# not a measured result -- AC-8 measures real per-run spend at T-8 and retunes.

ROUTER_MODEL = "claude-haiku-4-5"
REFLECTION_MODEL = "claude-haiku-4-5"
VISION_MODEL = "claude-haiku-4-5"
SPECIALIST_MODEL = "claude-opus-5"
SYNTHESIS_MODEL = "claude-opus-5"

# USD per million tokens, for the run cost report required by FR-18 and the
# ceiling enforced by NFR-1. Published Anthropic API rates.
MODEL_RATES_USD_PER_MTOK = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
}

# --- Run limits --------------------------------------------------------------
# NFR-1: a single analysis query stays under this. FR-9: the retrieval loop halts
# on sufficiency, on the iteration ceiling, or on this spend ceiling.

RUN_COST_CEILING_USD = 0.25
MAX_RETRIEVAL_ITERATIONS = 4
MAX_SPECIALISTS_PER_QUERY = 2  # FR-13

# --- Embedding ---------------------------------------------------------------
# DECISION-c49d609a: the Anthropic API exposes no embeddings endpoint, so
# embeddings are produced locally and nothing leaves the machine at embed time.

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
ENV_FILE = ROOT / ".env"


class MissingCredentialError(RuntimeError):
    """Raised when the API key is absent. Never carries the key's value."""


def _from_env_file(name: str) -> str:
    """Read one key from a local .env file, if there is one.

    A shell variable only lives in the window that set it, which makes it easy to
    lose and impossible for a separately-spawned process to see. `.env` is
    gitignored and never committed, so it holds the key for this machine without
    it reaching the repository.

    Deliberately minimal: KEY=VALUE per line, # comments, optional surrounding
    quotes. No dependency, and nothing here executes the file's contents.
    """
    try:
        text = ENV_FILE.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == name:
            return value.strip().strip("'\"").strip()
    return ""


def api_key() -> str:
    """Return the Anthropic API key.

    The real environment wins over `.env`, so an exported key overrides the file
    and a rotated one takes effect without editing anything. Resolved on every
    call rather than cached at import, so a key set after import is picked up.
    The error names the variable but never any part of a value.
    """
    key = os.environ.get(API_KEY_ENV_VAR, "").strip() or _from_env_file(API_KEY_ENV_VAR)
    if not key:
        raise MissingCredentialError(
            f"{API_KEY_ENV_VAR} is not set. Either export it in the shell you run "
            f"from, or put a single line {API_KEY_ENV_VAR}=<key> in {ENV_FILE.name} "
            "at the project root, which is gitignored. It is never written to any "
            "output or committed."
        )
    return key


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of one API call in USD, from reported token usage."""
    try:
        rate = MODEL_RATES_USD_PER_MTOK[model]
    except KeyError:
        raise KeyError(
            f"no published rate recorded for {model!r}; add it to "
            "MODEL_RATES_USD_PER_MTOK or run cost reporting is wrong"
        ) from None
    return (input_tokens * rate["input"] + output_tokens * rate["output"]) / 1_000_000
