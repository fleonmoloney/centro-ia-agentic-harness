"""AG2 (>= 1.x): agentes cuyas herramientas pasan por la puerta HITL.

AG2 1.x es una reescritura: ya no existe la API clásica de AutoGen (`ConversableAgent`). Esto
usa `ag2.Agent`. `gated_ag2_tools(gate, thread_id)` crea una herramienta por cada entrada del
registro; `build_agent` arma un agente con ellas. El modelo lo pone el llamador (`config=`):
un `OpenAIConfig(...)` apuntando a cualquier servidor compatible, o `ag2.testing.TestConfig`
en las pruebas.

Límite: el modelo es el cliente propio de AG2, no el `TracedLLM` del harness. Por eso el
presupuesto de llamadas/tokens del harness NO alcanza a estos agentes; lo único que el harness
garantiza acá es la puerta sobre las herramientas.
"""

from __future__ import annotations

from typing import Any

from centro_harness.hitl import HITLGate
from centro_harness.integrations.common import describe_outcome

try:
    from ag2 import Agent
    from ag2.tools.final.function_tool import FunctionTool
except ImportError as exc:  # pragma: no cover - depende del extra
    raise ImportError("Falta AG2: pip install -e '.[ag2]'") from exc


def gated_ag2_tools(gate: HITLGate, *, thread_id: str) -> list[FunctionTool]:
    tools = []
    for spec in gate.registry.specs():

        def make(name: str):
            def _call(arguments: dict[str, Any] | None = None) -> str:
                outcome = gate.submit(name, arguments or {}, thread_id=thread_id)
                return describe_outcome(outcome, gate.registry)

            return _call

        fn = make(spec.name)
        fn.__name__ = spec.name
        fn.__doc__ = spec.description or f"Herramienta {spec.name}"
        tools.append(FunctionTool.ensure_tool(fn))
    return tools


def build_agent(
    gate: HITLGate, *, name: str, prompt: str, config: Any, thread_id: str
) -> Agent:
    return Agent(name, prompt, config=config, tools=gated_ag2_tools(gate, thread_id=thread_id))
