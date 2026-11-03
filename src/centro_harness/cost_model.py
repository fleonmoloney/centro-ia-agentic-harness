"""Economía de la inferencia: modelo analítico de asignación de herramientas.

Compara dos formas de armar un pipeline de tareas con K herramientas:

* agent-centric: el agente recibe TODAS las herramientas; cada turno el prompt las incluye
  todas y el modelo gasta una salida extra eligiendo cuál usar.
* task-centric: cada tarea declara la herramienta que necesita; el prompt incluye solo esa y
  no hay paso de selección.

IMPORTANTE: es un modelo con supuestos explícitos (ver `assumptions`), calculado con
`estimate_tokens` (~4 caracteres/token). No mide un proveedor real ni la calidad de las
respuestas; sirve para dimensionar la palanca y decidir si vale la pena medirla en serio.
"""

from __future__ import annotations

from dataclasses import dataclass

from centro_harness.tracing import estimate_tokens


@dataclass(frozen=True)
class ToolDoc:
    name: str
    description: str


@dataclass(frozen=True)
class TaskSpec:
    description: str
    tool: str  # nombre de la herramienta que la tarea realmente necesita


@dataclass(frozen=True)
class CostComparison:
    agent_centric_tokens: int
    task_centric_tokens: int
    n_tools: int
    n_tasks: int
    assumptions: str

    @property
    def savings_pct(self) -> float:
        if self.agent_centric_tokens == 0:
            return 0.0
        saved = self.agent_centric_tokens - self.task_centric_tokens
        return round(100 * saved / self.agent_centric_tokens, 1)


def compare_tool_assignment(
    tools: list[ToolDoc],
    tasks: list[TaskSpec],
    *,
    system_prompt_tokens: int = 200,
    selection_output_tokens: int = 30,
) -> CostComparison:
    """Tokens totales (entrada + salida de selección) de ambos enfoques para `tasks`."""
    by_name = {t.name: t for t in tools}
    missing = {task.tool for task in tasks} - by_name.keys()
    if missing:
        raise ValueError(f"Tareas con herramientas no declaradas: {sorted(missing)}")

    all_tools_tokens = sum(estimate_tokens(f"{t.name}: {t.description}") for t in tools)

    agent_total = 0
    task_total = 0
    for task in tasks:
        task_tokens = estimate_tokens(task.description)
        agent_total += (
            system_prompt_tokens + task_tokens + all_tools_tokens + selection_output_tokens
        )
        own = by_name[task.tool]
        task_total += (
            system_prompt_tokens
            + task_tokens
            + estimate_tokens(f"{own.name}: {own.description}")
        )

    return CostComparison(
        agent_centric_tokens=agent_total,
        task_centric_tokens=task_total,
        n_tools=len(tools),
        n_tasks=len(tasks),
        assumptions=(
            f"prompt de sistema={system_prompt_tokens} tok; selección de herramienta="
            f"{selection_output_tokens} tok de salida por turno en agent-centric; "
            "1 turno por tarea; sin reintentos por mala selección (lo favorece al agent-centric)."
        ),
    )
