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
        """
        if self._client is None:
            import anthropic

            # No api_key argument: the SDK resolves the environment itself.
            # config.api_key() is called first so a missing key fails with our
            # message, which names the .env option, rather than the SDK's.
            config.api_key()
            self._client = anthropic.Anthropic()
        return self._client

    def _send(self, *, model: str, system: str, prompt: str, purpose: str,
              max_tokens: int, output_config: dict | None = None):
        self.budget.check(purpose)
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if output_config:
            kwargs["output_config"] = output_config

        response = self.client().messages.create(**kwargs)
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
             max_tokens: int = 4000) -> tuple[str, Usage]:
        response, usage = self._send(
            model=model, system=system, prompt=prompt, purpose=purpose, max_tokens=max_tokens
        )
        return self._first_text(response), usage

    def json(self, *, model: str, system: str, prompt: str, schema: dict, purpose: str,
             max_tokens: int = 4000) -> tuple[dict, Usage]:
        """Ask for a response constrained to `schema`.

        Structured output is used rather than "reply with JSON" in the prompt
        because the loop branches on these values; a stray prose preamble would
        turn a parse failure into a silent behaviour change.
        """
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
