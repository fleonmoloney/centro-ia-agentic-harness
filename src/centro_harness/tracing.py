"""Trazabilidad: cada llamada al LLM deja un evento con latencia y tokens.

Los tokens son *reales* cuando el backend los informa (`last_usage`, como hacen
`OpenAICompatLLM` y `TransformersLLM`) y *estimados* (1 token ≈ 4 caracteres) cuando no
(`ScriptedLLM`). Cada evento dice cuál de las dos es, y el resumen informa la fuente: así una
cifra estimada no se confunde con una medida.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from centro_harness.budget import Budget
from centro_harness.llm import LLM

T = TypeVar("T", bound=BaseModel)


def estimate_tokens(text: str) -> int:
    """Estimación gruesa: ~4 caracteres por token, mínimo 1."""
    return max(1, len(text) // 4)


@dataclass
class TraceEvent:
    label: str
    kind: str  # "generate" | "structured" | "structured_error"
    tokens_in: int
    tokens_out: int
    latency_ms: float
    error: str | None = None
    tokens_real: bool = False


@dataclass
class Tracer:
    events: list[TraceEvent] = field(default_factory=list)

    def record(self, event: TraceEvent) -> None:
        self.events.append(event)

    @property
    def calls(self) -> int:
        return len(self.events)

    @property
    def errors(self) -> int:
        return sum(1 for e in self.events if e.error)

    @property
    def tokens_in(self) -> int:
        return sum(e.tokens_in for e in self.events)

    @property
    def tokens_out(self) -> int:
        return sum(e.tokens_out for e in self.events)

    @property
    def latency_ms(self) -> float:
        return sum(e.latency_ms for e in self.events)

    @property
    def tokens_source(self) -> str:
        """'provider' si todos los eventos traen conteo real, 'estimated' si ninguno, 'mixed'."""
        if not self.events:
            return "none"
        real = sum(e.tokens_real for e in self.events)
        if real == len(self.events):
            return "provider"
        return "estimated" if real == 0 else "mixed"

    def count(self, kind: str) -> int:
        return sum(1 for e in self.events if e.kind == kind)

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "errors": self.errors,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_source": self.tokens_source,
            "latency_ms": round(self.latency_ms, 3),
        }

    def to_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for event in self.events:
                fh.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")


class TracedLLM:
    """Envuelve cualquier `LLM`: registra un evento por llamada (también las fallidas) y,
    si se le pasa un `Budget`, corta antes de la llamada que lo superaría."""

    def __init__(self, inner: LLM, tracer: Tracer, budget: Budget | None = None) -> None:
        self.inner = inner
        self.tracer = tracer
        self.budget = budget

    def _guard(self) -> None:
        if self.budget is not None:
            self.budget.check(
                calls=self.tracer.calls, tokens=self.tracer.tokens_in + self.tracer.tokens_out
            )

    def _tokens(self, prompt: str, output: str) -> tuple[int, int, bool]:
        usage = getattr(self.inner, "last_usage", None)
        if usage is not None:
            return int(usage[0]), int(usage[1]), True
        return estimate_tokens(prompt), (estimate_tokens(output) if output else 0), False

    def generate(self, prompt: str, *, label: str = "") -> str:
        self._guard()
        start = time.perf_counter()
        out = self.inner.generate(prompt, label=label)
        tokens_in, tokens_out, real = self._tokens(prompt, out)
        self.tracer.record(
            TraceEvent(
                label or "generate",
                "generate",
                tokens_in,
                tokens_out,
                (time.perf_counter() - start) * 1000,
                tokens_real=real,
            )
        )
        return out

    def structured(self, prompt: str, schema: type[T], *, label: str = "") -> T:
        self._guard()
        start = time.perf_counter()
        try:
            out = self.inner.structured(prompt, schema, label=label)
        except Exception as exc:
            tokens_in, tokens_out, real = self._tokens(prompt, "")
            self.tracer.record(
                TraceEvent(
                    label or "structured",
                    "structured_error",
                    tokens_in,
                    tokens_out,
                    (time.perf_counter() - start) * 1000,
                    error=type(exc).__name__,
                    tokens_real=real,
                )
            )
            raise
        tokens_in, tokens_out, real = self._tokens(prompt, out.model_dump_json())
        self.tracer.record(
            TraceEvent(
                label or "structured",
                "structured",
                tokens_in,
                tokens_out,
                (time.perf_counter() - start) * 1000,
                tokens_real=real,
            )
        )
        return out
