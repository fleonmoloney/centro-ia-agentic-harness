"""Supervisión humana (HITL) con pausa real del grafo y bitácora de auditoría.

Política (default-deny):
* herramienta desconocida  -> bloqueada,
* riesgo bajo              -> se ejecuta y se registra,
* riesgo alto              -> el grafo se PAUSA (`interrupt`) hasta que una persona decida;
                              la aprobación debe traer el nombre del revisor, si no se rechaza.

Es el equivalente conceptual al `AskPermissionRequirement` de BeeAI, implementado con
LangGraph (checkpointer + `interrupt` + `Command(resume=...)`). No usa BeeAI.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field


class Risk(str, Enum):
    LOW = "low"
    HIGH = "high"


class ToolSpec(BaseModel):
    name: str
    risk: Risk
    description: str = ""


Decision = Literal[
    "auto_approved", "approved", "rejected", "rejected_no_reviewer", "denied_unknown_tool"
]


class ToolCallRecord(BaseModel):
    """Entrada de auditoría: qué se pidió, qué se decidió, quién y cuándo."""

    tool: str
    arguments: dict
    decision: Decision
    reviewer: str | None = None
    result: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, Callable[..., object]]] = {}

    def register(self, spec: ToolSpec, fn: Callable[..., object]) -> None:
        self._tools[spec.name] = (spec, fn)

    def spec(self, name: str) -> ToolSpec | None:
        entry = self._tools.get(name)
        return entry[0] if entry else None

    def call(self, name: str, arguments: dict) -> object:
        return self._tools[name][1](**arguments)


class _GateState(TypedDict, total=False):
    tool: str
    arguments: dict
    decision: str
    reviewer: str | None
    result: str | None
    record: dict


@dataclass
class GateOutcome:
    status: Literal["pending_approval", "completed", "blocked"]
    record: ToolCallRecord | None = None
    pending: dict | None = None


def _redact(arguments: dict, keys: frozenset[str]) -> dict:
    return {k: ("***" if k in keys else v) for k, v in arguments.items()}


class HITLGate:
    """Puerta de aprobación para llamadas a herramientas."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        redact_keys: tuple[str, ...] = ("citizen_id",),
    ) -> None:
        self.registry = registry
        self.audit: list[ToolCallRecord] = []
        self._redact_keys = frozenset(redact_keys)
        self._graph = self._build()

    # ------------------------------------------------------------------ grafo
    def _build(self):
        registry = self.registry
        redact_keys = self._redact_keys

        def gate(state: _GateState) -> dict:
            spec = registry.spec(state["tool"])
            if spec is None:
                return {"decision": "denied_unknown_tool", "reviewer": None}
            if spec.risk is Risk.LOW:
                return {"decision": "auto_approved", "reviewer": None}
            answer = interrupt(
                {
                    "tool": spec.name,
                    "risk": spec.risk.value,
                    "description": spec.description,
                    "arguments": _redact(state["arguments"], redact_keys),
                }
            )
            reviewer = (answer or {}).get("reviewer") or None
            if not (answer or {}).get("approved"):
                return {"decision": "rejected", "reviewer": reviewer}
            if not reviewer:
                return {"decision": "rejected_no_reviewer", "reviewer": None}
            return {"decision": "approved", "reviewer": reviewer}

        def route(state: _GateState) -> str:
            return "execute" if state["decision"] in ("auto_approved", "approved") else "record"

        def execute(state: _GateState) -> dict:
            try:
                result = str(registry.call(state["tool"], state["arguments"]))
            except Exception as exc:  # la herramienta falló: se audita, no se oculta
                result = f"error: {type(exc).__name__}: {exc}"
            return {"result": result}

        def record(state: _GateState) -> dict:
            rec = ToolCallRecord(
                tool=state["tool"],
                arguments=_redact(state["arguments"], redact_keys),
                decision=state["decision"],  # type: ignore[arg-type]
                reviewer=state.get("reviewer"),
                result=state.get("result"),
            )
            return {"record": rec.model_dump(mode="json")}

        graph = StateGraph(_GateState)
        graph.add_node("gate", gate)
        graph.add_node("execute", execute)
        graph.add_node("record", record)
        graph.add_edge(START, "gate")
        graph.add_conditional_edges("gate", route, {"execute": "execute", "record": "record"})
        graph.add_edge("execute", "record")
        graph.add_edge("record", END)
        return graph.compile(checkpointer=MemorySaver())

    # ------------------------------------------------------------------ API
    def _outcome(self, result: dict) -> GateOutcome:
        interrupts = result.get("__interrupt__")
        if interrupts:
            return GateOutcome("pending_approval", pending=interrupts[0].value)
        record = ToolCallRecord.model_validate(result["record"])
        self.audit.append(record)
        ok = record.decision in ("auto_approved", "approved")
        return GateOutcome("completed" if ok else "blocked", record=record)

    def submit(self, tool: str, arguments: dict, *, thread_id: str) -> GateOutcome:
        """Pide ejecutar una herramienta. Si requiere aprobación, queda pausada."""
        config = {"configurable": {"thread_id": thread_id}}
        return self._outcome(self._graph.invoke({"tool": tool, "arguments": arguments}, config))

    def resume(
        self, *, thread_id: str, approved: bool, reviewer: str | None = None
    ) -> GateOutcome:
        """Entrega la decisión humana a una llamada pausada."""
        config = {"configurable": {"thread_id": thread_id}}
        command = Command(resume={"approved": approved, "reviewer": reviewer})
        return self._outcome(self._graph.invoke(command, config))

    def export_audit(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for rec in self.audit:
                fh.write(json.dumps(rec.model_dump(mode="json"), ensure_ascii=False) + "\n")


__all__ = [
    "GateOutcome",
    "HITLGate",
    "Risk",
    "ToolCallRecord",
    "ToolRegistry",
    "ToolSpec",
]
