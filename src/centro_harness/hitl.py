"""Supervisión humana (HITL) con pausa real del grafo y bitácora de auditoría.

Política (default-deny):
* herramienta desconocida  -> bloqueada,
* riesgo bajo              -> se ejecuta y se registra,
* riesgo alto              -> el grafo se PAUSA (`interrupt`) hasta que una persona decida;
                              la aprobación debe traer el nombre del revisor (y, si se configuró
                              una lista, el revisor debe estar autorizado), si no se rechaza.

Integridad de la aprobación (v0.3): cada solicitud recibe un `request_id`. `resume` exige ese
identificador, así que una decisión humana no puede aplicarse a una solicitud distinta de la que
la persona vio. Un segundo `submit` sobre un hilo pausado se rechaza, y un `resume` sobre un
hilo que no está pausado también (no duplica ejecución ni auditoría).

Es el equivalente conceptual al `AskPermissionRequirement` de BeeAI, implementado con
LangGraph (checkpointer + `interrupt` + `Command(resume=...)`). No usa BeeAI.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from centro_harness.schemas import is_valid_ci, mask_ci


class Risk(str, Enum):
    LOW = "low"
    HIGH = "high"


class ToolSpec(BaseModel):
    name: str
    risk: Risk
    description: str = ""


Decision = Literal[
    "auto_approved",
    "approved",
    "rejected",
    "rejected_no_reviewer",
    "rejected_unauthorized",
    "denied_unknown_tool",
]

_EXECUTED = ("auto_approved", "approved")


# --------------------------------------------------------------------------- errores
class HITLError(RuntimeError):
    """Base de los errores de uso de la puerta."""


class ThreadBusyError(HITLError):
    """Ese hilo ya tiene una solicitud pausada esperando decisión."""


class NoPendingRequestError(HITLError):
    """No hay una solicitud pausada en ese hilo (nunca existió o ya se resolvió)."""


class RequestMismatchError(HITLError):
    """La decisión apunta a una solicitud distinta de la que está pausada."""


# --------------------------------------------------------------------------- datos personales
# Candidatas a cédula: x.xxx.xxx-x (con o sin millones) o 7-8 dígitos seguidos. Solo se enmascara
# si el dígito verificador coincide, para no tocar otros números. Puede enmascarar de más algún
# número de 7-8 dígitos que valide por casualidad: se prefiere errar hacia ocultar.
_CI_CANDIDATE = re.compile(r"(?<!\d)\d{1,2}(?:\.\d{3}){2}-\d(?!\d)|(?<!\d)\d{7,8}(?!\d)")


def scrub_text(text: str) -> str:
    """Enmascara las cédulas válidas que aparezcan dentro de un texto libre."""

    def _replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        return mask_ci(raw) if is_valid_ci(raw) else raw

    return _CI_CANDIDATE.sub(_replace, text)


def redact(value: Any, keys: frozenset[str]) -> Any:
    """Copia `value` enmascarando, a cualquier profundidad, las claves sensibles.

    Además pasa por `scrub_text` cualquier texto, para atrapar cédulas escritas en valores que
    no están bajo una clave conocida.
    """
    if isinstance(value, dict):
        return {k: ("***" if k in keys else redact(v, keys)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v, keys) for v in value]
    if isinstance(value, str):
        return scrub_text(value)
    return value


# ------------------------------------------------------------------ registro de herramientas
class ToolCallRecord(BaseModel):
    """Entrada de auditoría: qué se pidió, qué se decidió, quién y cuándo."""

    request_id: str
    tool: str
    arguments: dict
    decision: Decision
    reviewer: str | None = None
    result: str | None = None
    error: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, Callable[..., object]]] = {}

    def register(self, spec: ToolSpec, fn: Callable[..., object]) -> None:
        self._tools[spec.name] = (spec, fn)

    def spec(self, name: str) -> ToolSpec | None:
        entry = self._tools.get(name)
        return entry[0] if entry else None

    def specs(self) -> list[ToolSpec]:
        return [spec for spec, _ in self._tools.values()]

    def call(self, name: str, arguments: dict) -> object:
        return self._tools[name][1](**arguments)


class _GateState(TypedDict, total=False):
    request_id: str
    tool: str
    arguments: dict
    decision: str
    reviewer: str | None
    result: str | None
    error: str | None
    record: dict | None


@dataclass
class GateOutcome:
    status: Literal["pending_approval", "completed", "blocked", "failed"]
    record: ToolCallRecord | None = None
    pending: dict | None = None
    request_id: str | None = None


def sqlite_checkpointer(path: str | Path):
    """Checkpointer SQLite: las solicitudes pausadas sobreviven a un reinicio.

    Requiere el extra `persist` (`pip install '.[persist]'`). El archivo contiene los argumentos
    de las llamadas pausadas EN CLARO (hacen falta para ejecutarlas al aprobar): protegé el
    archivo como protegerías los datos de origen.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise RuntimeError("Falta el extra: pip install '.[persist]'") from exc
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(target), check_same_thread=False))


class HITLGate:
    """Puerta de aprobación para llamadas a herramientas."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        redact_keys: tuple[str, ...] = ("citizen_id",),
        authorized_reviewers: set[str] | frozenset[str] | None = None,
        checkpointer: Any | None = None,
        audit_path: str | Path | None = None,
        scrub_results: bool = True,
    ) -> None:
        self.registry = registry
        self.audit: list[ToolCallRecord] = []
        self._redact_keys = frozenset(redact_keys)
        self._authorized = frozenset(authorized_reviewers) if authorized_reviewers else None
        self._audit_path = Path(audit_path) if audit_path else None
        self._scrub_results = scrub_results
        self._graph = self._build(checkpointer or MemorySaver())

    # ------------------------------------------------------------------ grafo
    def _build(self, checkpointer: Any):
        registry = self.registry
        redact_keys = self._redact_keys
        authorized = self._authorized
        scrub_results = self._scrub_results

        def clean(text: str | None) -> str | None:
            if text is None or not scrub_results:
                return text
            return scrub_text(text)

        def gate(state: _GateState) -> dict:
            spec = registry.spec(state["tool"])
            if spec is None:
                return {"decision": "denied_unknown_tool", "reviewer": None}
            if spec.risk is Risk.LOW:
                return {"decision": "auto_approved", "reviewer": None}
            answer = (
                interrupt(
                    {
                        "request_id": state["request_id"],
                        "tool": spec.name,
                        "risk": spec.risk.value,
                        "description": spec.description,
                        "arguments": redact(state["arguments"], redact_keys),
                    }
                )
                or {}
            )
            reviewer = answer.get("reviewer") or None
            if not answer.get("approved"):
                return {"decision": "rejected", "reviewer": reviewer}
            if not reviewer:
                return {"decision": "rejected_no_reviewer", "reviewer": None}
            if authorized is not None and reviewer not in authorized:
                return {"decision": "rejected_unauthorized", "reviewer": reviewer}
            return {"decision": "approved", "reviewer": reviewer}

        def route(state: _GateState) -> str:
            return "execute" if state["decision"] in _EXECUTED else "record"

        def execute(state: _GateState) -> dict:
            try:
                return {"result": clean(str(registry.call(state["tool"], state["arguments"])))}
            except Exception as exc:  # la herramienta falló: se audita como falla, no se oculta
                return {"error": clean(f"{type(exc).__name__}: {exc}")}

        def record(state: _GateState) -> dict:
            rec = ToolCallRecord(
                request_id=state["request_id"],
                tool=state["tool"],
                arguments=redact(state["arguments"], redact_keys),
                decision=state["decision"],  # type: ignore[arg-type]
                reviewer=state.get("reviewer"),
                result=state.get("result"),
                error=state.get("error"),
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
        return graph.compile(checkpointer=checkpointer)

    # ------------------------------------------------------------------ API
    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _snapshot(self, thread_id: str):
        return self._graph.get_state(self._config(thread_id))

    def pending(self, thread_id: str) -> dict | None:
        """Lo que la persona debe revisar en ese hilo, o None si no hay nada pausado.

        Sirve para volver a mostrar la solicitud tras un reinicio (con checkpointer persistente).
        """
        snapshot = self._snapshot(thread_id)
        if not snapshot.next:
            return None
        for task in snapshot.tasks:
            for item in getattr(task, "interrupts", ()):
                return item.value
        return None

    def _outcome(self, result: dict, request_id: str) -> GateOutcome:
        interrupts = result.get("__interrupt__")
        if interrupts:
            return GateOutcome(
                "pending_approval", pending=interrupts[0].value, request_id=request_id
            )
        record = ToolCallRecord.model_validate(result["record"])
        self._append_audit(record)
        if record.decision not in _EXECUTED:
            status = "blocked"
        else:
            status = "failed" if record.error else "completed"
        return GateOutcome(status, record=record, request_id=request_id)  # type: ignore[arg-type]

    def _append_audit(self, record: ToolCallRecord) -> None:
        self.audit.append(record)
        if self._audit_path is not None:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self._audit_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")

    def submit(self, tool: str, arguments: dict, *, thread_id: str) -> GateOutcome:
        """Pide ejecutar una herramienta. Si requiere aprobación, queda pausada.

        Lanza `ThreadBusyError` si el hilo ya tiene una solicitud esperando decisión.
        """
        if self._snapshot(thread_id).next:
            raise ThreadBusyError(
                f"El hilo {thread_id!r} tiene una solicitud pausada; resolvela antes de enviar otra"
            )
        request_id = uuid.uuid4().hex
        # Se reinician todos los campos: un hilo reutilizado no debe heredar resultados viejos.
        initial: _GateState = {
            "request_id": request_id,
            "tool": tool,
            "arguments": arguments,
            "decision": "",
            "reviewer": None,
            "result": None,
            "error": None,
            "record": None,
        }
        return self._outcome(self._graph.invoke(initial, self._config(thread_id)), request_id)

    def resume(
        self,
        *,
        thread_id: str,
        request_id: str,
        approved: bool,
        reviewer: str | None = None,
    ) -> GateOutcome:
        """Entrega la decisión humana a una solicitud pausada.

        `request_id` debe ser el de la solicitud que la persona revisó (viene en `pending`).
        """
        snapshot = self._snapshot(thread_id)
        if not snapshot.next:
            raise NoPendingRequestError(f"No hay una solicitud pausada en el hilo {thread_id!r}")
        if snapshot.values.get("request_id") != request_id:
            raise RequestMismatchError(
                f"La solicitud pausada en {thread_id!r} no es {request_id!r}; no se ejecutó nada"
            )
        command = Command(resume={"approved": approved, "reviewer": reviewer})
        return self._outcome(self._graph.invoke(command, self._config(thread_id)), request_id)

    def export_audit(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for rec in self.audit:
                fh.write(json.dumps(rec.model_dump(mode="json"), ensure_ascii=False) + "\n")


__all__ = [
    "GateOutcome",
    "HITLError",
    "HITLGate",
    "NoPendingRequestError",
    "RequestMismatchError",
    "Risk",
    "ThreadBusyError",
    "ToolCallRecord",
    "ToolRegistry",
    "ToolSpec",
    "redact",
    "scrub_text",
    "sqlite_checkpointer",
]
