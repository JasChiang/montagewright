"""What the run has spent, against one ceiling.

A single global cap in dollars, and no per-stage quotas. Quotas make a budget
into a behaviour modifier: a stage told it has little left starts choosing
cheaper answers, and cheaper answers to editorial questions are worse ones.
Running out of money is a reason to stop with the best cut so far, never a
reason to do the next step badly.

Pricing lives in a table rather than in the arithmetic, because rates change
and a hard-coded rate is a silent error the day they do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# USD per million tokens. Thinking tokens bill at the output rate.
PRICING: dict[str, dict[str, float]] = {
    "gemini-3.6-flash": {
        "input": 1.50,
        "cached_input": 0.15,
        "output": 7.50,
    },
    "gemini-3.5-flash": {
        "input": 1.50,
        "cached_input": 0.15,
        "output": 9.00,
    },
}


class BudgetSpent(RuntimeError):
    """The cap is reached. Deliver what exists; do not degrade to continue."""


@dataclass
class Ledger:
    cap_usd: float
    model_id: str = "gemini-3.6-flash"
    entries: list[dict[str, float | str]] = field(default_factory=list)

    @property
    def spent_usd(self) -> float:
        return sum(float(entry["usd"]) for entry in self.entries)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)

    def record(
        self,
        stage: str,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        rates = PRICING.get(self.model_id, PRICING["gemini-3.6-flash"])
        fresh = max(0, input_tokens - cached_tokens)
        usd = (
            fresh * rates["input"]
            + cached_tokens * rates["cached_input"]
            + output_tokens * rates["output"]
        ) / 1_000_000
        self.entries.append(
            {
                "stage": stage,
                "input": input_tokens,
                "cached": cached_tokens,
                "output": output_tokens,
                "usd": round(usd, 6),
            }
        )
        return usd

    def check(self) -> None:
        """Call before dispatching, so the cap stops work rather than paying for it."""

        if self.spent_usd >= self.cap_usd:
            raise BudgetSpent(
                f"spent ${self.spent_usd:.4f} of ${self.cap_usd:.2f}; "
                "delivering the best cut reached so far"
            )

    def summary(self) -> dict[str, object]:
        by_stage: dict[str, float] = {}
        for entry in self.entries:
            by_stage[str(entry["stage"])] = round(
                by_stage.get(str(entry["stage"]), 0.0) + float(entry["usd"]), 6
            )
        return {
            "cap_usd": self.cap_usd,
            "spent_usd": round(self.spent_usd, 6),
            "remaining_usd": round(self.remaining_usd, 6),
            "calls": len(self.entries),
            "by_stage": by_stage,
        }
