"""BeeAI: herramientas que pasan por la puerta HITL.

`gated_tools(gate, thread_id)` devuelve una `Tool` de BeeAI por cada herramienta del registro.
Cualquier agente de BeeAI que las use obtiene la política del harness: riesgo bajo se ejecuta y
se audita; riesgo alto queda PENDIENTE hasta que una persona llame a `gate.resume(...)`.

Límite: el esquema de entrada es genérico (`arguments: dict`), porque `ToolSpec` no declara los
argumentos de cada herramienta. Un modelo débil puede confundirse con eso.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from centro_harness.hitl import HITLGate
from centro_harness.integrations.common import describe_outcome

try:
    from beeai_framework.context import RunContext
    from beeai_framework.emitter import Emitter
    from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions
except ImportError as exc:  # pragma: no cover - depende del extra
    raise ImportError("Falta BeeAI: pip install -e '.[beeai]'") from exc


class GateInput(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict, description="Argumentos")


class GatedBeeTool(Tool[GateInput, ToolRunOptions, StringToolOutput]):
    name = "gated"  # se reemplaza por instancia; la clase base exige que existan
    description = "Herramienta con supervisión humana"

    def __init__(self, gate: HITLGate, tool_name: str, thread_id: str) -> None:
        super().__init__()
        self._gate = gate
        self._thread_id = thread_id
        self.name = tool_name
        spec = gate.registry.spec(tool_name)
        self.description = (spec.description if spec else "") or f"Herramienta {tool_name}"

    @property
    def input_schema(self) -> type[GateInput]:
        return GateInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "gated", self.name], creator=self)

    async def _run(
        self, input: GateInput, options: ToolRunOptions | None, context: RunContext
    ) -> StringToolOutput:
        outcome = self._gate.submit(self.name, input.arguments, thread_id=self._thread_id)
        return StringToolOutput(describe_outcome(outcome, self._gate.registry))


def gated_tools(gate: HITLGate, *, thread_id: str) -> list[GatedBeeTool]:
    return [GatedBeeTool(gate, spec.name, thread_id) for spec in gate.registry.specs()]
