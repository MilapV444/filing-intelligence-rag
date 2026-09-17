"""Anthropic client wrapper with cost accounting.

Everything that calls a model goes through here, for one reason: NFR-1 caps a
run at USD 0.25 and FR-18 makes the note report what it spent. A call site that
talks to the SDK directly is spend nobody counted.

The client is injectable. Every test in this project runs against a recorded
stand-in rather than the live API, so the suite costs nothing and stays
deterministic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import config


class Truncated(RuntimeError):
    """The model ran out of output tokens before finishing its reply."""


class BudgetExceeded(RuntimeError):
    """Raised before a call that would breach the run's spend ceiling.

    Raised *before*, not after: discovering the overrun afterwards means the
    money is already gone, which is not a ceiling.
    """


@dataclass(frozen=True)
class Usage:
    model: str
    input_tokens: int
    output_tokens: int
    purpose: str = ""

    @property
    def cost_usd(self) -> float:
        return config.cost_usd(self.model, self.input_tokens, self.output_tokens)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "purpose": self.purpose,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class Budget:
    """Cumulative spend for one run (NFR-1)."""

    ceiling_usd: float = config.RUN_COST_CEILING_USD
    usages: list[Usage] = field(default_factory=list)

    @property
    def spent_usd(self) -> float:
        return sum(u.cost_usd for u in self.usages)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.ceiling_usd - self.spent_usd)

    @property
    def calls(self) -> int:
        return len(self.usages)

    def record(self, usage: Usage) -> None:
        self.usages.append(usage)

    def check(self, purpose: str = "") -> None:
        if self.spent_usd >= self.ceiling_usd:
            raise BudgetExceeded(
                f"run has spent ${self.spent_usd:.4f} of its ${self.ceiling_usd:.2f} "
                f"ceiling; refusing to start {purpose or 'another call'}"
            )

    def check_projected(self, model: str, prompt_chars: int, purpose: str = "") -> None:
        """Refuse a call whose likely cost would breach the ceiling.

        Checking only what has already been spent is not a ceiling: the first
        live run sat at $0.228 of $0.25, started one more Opus call, and finished
        at $0.4302. A ceiling has to price the call it is about to authorise.
        """
        self.check(purpose)
        projected = config.cost_usd(
            model,
            int(prompt_chars / config.CHARS_PER_TOKEN),
            config.ASSUMED_OUTPUT_TOKENS,
        )
        if self.spent_usd + projected > self.ceiling_usd:
            raise BudgetExceeded(
                f"{purpose or 'this call'} is projected to cost about "
                f"${projected:.4f} on {model}, which would take the run past its "
                f"${self.ceiling_usd:.2f} ceiling (spent ${self.spent_usd:.4f}). "
                "Refusing before spending rather than reporting the overrun after."
            )

    def to_dict(self) -> dict:
        return {
            "ceiling_usd": self.ceiling_usd,
            "spent_usd": round(self.spent_usd, 6),
            "calls": self.calls,
            "by_call": [u.to_dict() for u in self.usages],
        }


class LLM:
    """Thin wrapper over the Messages API. One place to count tokens."""

    def __init__(self, client=None, budget: Budget | None = None):
        self._client = client
        self.budget = budget if budget is not None else Budget()

    def client(self):
        """Construct the SDK client lazily.

        Lazily so that importing this module -- which the offline ingest path
        does transitively -- never demands a credential.

        The key is passed explicitly rather than left to the SDK. The SDK
        resolves credentials from the process environment only, so a key living
        in .env satisfied config.api_key() and was invisible to the client: the
        first live call this project ever made failed with "could not resolve
        authentication method" over a correctly configured .env. Our resolution
        chain is the authority, so whatever it returns is what the client gets.
        """
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=config.api_key())
        return self._client

    # Non-streaming default. 4000 was too low: a specialist reading a full
    # evidence block ran past it and returned JSON cut off mid-string, which
    # surfaced as a parse error rather than as "the answer did not fit".
    DEFAULT_MAX_TOKENS = 16000

    def _send(self, *, model: str, system: str, prompt: str, purpose: str,
              max_tokens: int, output_config: dict | None = None):
        self.budget.check_projected(model, len(system) + len(prompt), purpose)
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if output_config:
            kwargs["output_config"] = output_config

        response = self.client().messages.create(**kwargs)
        if getattr(response, "stop_reason", None) == "max_tokens":
            # Say what actually happened. Truncated JSON otherwise reads as a
            # model that cannot follow a schema.
            raise Truncated(
                f"{purpose}: the response hit the {max_tokens} token limit and was "
                "cut off. Raise max_tokens or reduce the evidence sent."
            )
        usage = Usage(
            model=model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            purpose=purpose,
        )
        self.budget.record(usage)
        return response, usage

    @staticmethod
    def _first_text(response) -> str:
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""

    def text(self, *, model: str, system: str, prompt: str, purpose: str,
             max_tokens: int | None = None) -> tuple[str, Usage]:
        max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        response, usage = self._send(
            model=model, system=system, prompt=prompt, purpose=purpose, max_tokens=max_tokens
        )
        return self._first_text(response), usage

    def json(self, *, model: str, system: str, prompt: str, schema: dict, purpose: str,
             max_tokens: int | None = None) -> tuple[dict, Usage]:
        """Ask for a response constrained to `schema`.

        Structured output is used rather than "reply with JSON" in the prompt
        because the loop branches on these values; a stray prose preamble would
        turn a parse failure into a silent behaviour change.
        """
        max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        response, usage = self._send(
            model=model, system=system, prompt=prompt, purpose=purpose,
            max_tokens=max_tokens,
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = self._first_text(response)
        try:
            return json.loads(text), usage
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{purpose}: model returned text that is not valid JSON ({exc})"
            ) from None
