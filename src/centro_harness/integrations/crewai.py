"""CrewAI: herramientas por la puerta HITL y modelo del harness.

* `gated_crew_tools(gate, thread_id)` → una `BaseTool` de CrewAI por herramienta del registro.
* `HarnessCrewLLM(llm)` → `BaseLLM` de CrewAI que delega en un LLM del harness. Si se pasa un
  `TracedLLM`, el presupuesto (llamadas/tokens/segundos) y las trazas alcanzan también al equipo.
  Si el presupuesto se agota, la excepción `BudgetExceeded` corta la ejecución del equipo.

Límites: CrewAI formatea los prompts a su manera (ReAct en texto); con modelos débiles el formato
falla con frecuencia. El esquema de las herramientas es genérico (`arguments: dict`).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from centro_harness.hitl import HITLGate
from centro_harness.integrations.common import describe_outcome

try:
    from crewai import BaseLLM
    from crewai.tools import BaseTool
except ImportError as exc:  # pragma: no cover - depende del extra
    raise ImportError("Falta CrewAI: pip install -e '.[crewai]'") from exc


class GateInput(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict, description="Argumentos")


def _flatten(messages: str | list[dict[str, Any]]) -> str:
    if isinstance(messages, str):
        return messages
    return "\n\n".join(f"[{m.get('role', '?')}]\n{m.get('content', '')}" for m in messages)


class HarnessCrewLLM(BaseLLM):
    """Adaptador: CrewAI llama a `call`, el harness responde con su LLM (con presupuesto)."""

    def __init__(self, llm: Any, model: str = "harness") -> None:
        super().__init__(model=model)
        object.__setattr__(self, "_harness_llm", llm)

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None) -> str:  # noqa: E501
        return self._harness_llm.generate(_flatten(messages), label="crewai")

    def supports_function_calling(self) -> bool:
        return False  # el equipo usa el formato ReAct en texto

    def get_context_window_size(self) -> int:
        return 8192


def _make_tool(gate: HITLGate, name: str, description: str, thread_id: str) -> BaseTool:
    class _Gated(BaseTool):
        args_schema: type[BaseModel] = GateInput

        def _run(self, arguments: dict[str, Any] | None = None, **kwargs: Any) -> str:
            args = arguments if arguments is not None else kwargs
            outcome = gate.submit(name, args, thread_id=thread_id)
            return describe_outcome(outcome, gate.registry)

    return _Gated(name=name, description=description)


def gated_crew_tools(gate: HITLGate, *, thread_id: str) -> list[BaseTool]:
    return [
        _make_tool(gate, spec.name, spec.description or f"Herramienta {spec.name}", thread_id)
        for spec in gate.registry.specs()
    ]
