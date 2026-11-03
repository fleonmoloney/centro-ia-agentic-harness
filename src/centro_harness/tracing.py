"""Trazabilidad: cada llamada al LLM deja un evento con latencia y tokens estimados.

Los tokens se estiman con una heurística (1 token ≈ 4 caracteres). Sirve para comparar
configuraciones entre sí; no reemplaza el conteo real de tu proveedor.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

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

    def count(self, kind: str) -> int:
        return sum(1 for e in self.events if e.kind == kind)

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "errors": self.errors,
            "tokens_in_est": self.tokens_in,
            "tokens_out_est": self.tokens_out,
            "latency_ms": round(self.latency_ms, 3),
        }

    def to_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for event in self.events:
                fh.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")


class TracedLLM:
    """Envuelve cualquier `LLM` y registra un evento por llamada (también las fallidas)."""

    def __init__(self, inner: LLM, tracer: Tracer) -> None:
        self.inner = inner
        self.tracer = tracer

    def generate(self, prompt: str, *, label: str = "") -> str:
        start = time.perf_counter()
        out = self.inner.generate(prompt, label=label)
        self.tracer.record(
            TraceEvent(
                label or "generate",
                "generate",
                estimate_tokens(prompt),
                estimate_tokens(out),
                (time.perf_counter() - start) * 1000,
            )
        )
        return out

    def structured(self, prompt: str, schema: type[T], *, label: str = "") -> T:
        start = time.perf_counter()
        try:
            out = self.inner.structured(prompt, schema, label=label)
        except Exception as exc:
            self.tracer.record(
                TraceEvent(
                    label or "structured",
                    "structured_error",
                    estimate_tokens(prompt),
                    0,
                    (time.perf_counter() - start) * 1000,
                    error=type(exc).__name__,
                )
            )
            raise
        self.tracer.record(
            TraceEvent(
                label or "structured",
                "structured",
                estimate_tokens(prompt),
                estimate_tokens(out.model_dump_json()),
                (time.perf_counter() - start) * 1000,
            )
        )
        return out
