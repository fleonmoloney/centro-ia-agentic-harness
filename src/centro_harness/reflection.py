"""Patrón de reflexión (generar → evaluar → refinar) como grafo cíclico de LangGraph.

Qué es real: el ciclo, el ruteo por umbral, el tope de iteraciones (control de costo), la
salida estructurada del evaluador con reintento y la traza por llamada.
Qué depende del LLM que le pases: la calidad del texto y de la autocrítica.

Cortes ordenados (v0.3): si se agota el presupuesto (`TracedLLM` con `Budget`) o el evaluador no
logra producir una puntuación válida, el grafo termina con el mejor borrador disponible y un
`status` explícito (`budget_exceeded` / `evaluator_failed`) en lugar de lanzar una excepción.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from centro_harness.budget import BudgetExceeded
from centro_harness.llm import LLM
from centro_harness.schemas import StructuredOutputError, structured_with_retry


class Evaluation(BaseModel):
    """Autocrítica estructurada del evaluador."""

    score: float = Field(ge=0.0, le=1.0, description="Calidad de la respuesta, de 0 a 1")
    missing: str = Field(default="", description="Qué información falta")
    superfluous: str = Field(default="", description="Qué información sobra")


class ReflectionState(TypedDict, total=False):
    task: str
    draft: str
    critique: dict
    score: float
    iteration: int
    history: Annotated[list[dict], operator.add]
    halt: Literal["budget", "evaluator"]
    halt_detail: str
    status: Literal["accepted", "max_iterations", "budget_exceeded", "evaluator_failed"]


def route_by_score(
    state: ReflectionState, *, threshold: float, max_iterations: int
) -> Literal["accept", "refine", "stop"]:
    """Decide el siguiente paso. Función pura: se prueba sin LLM ni grafo."""
    if state.get("score", 0.0) >= threshold:
        return "accept"
    if state.get("iteration", 0) >= max_iterations:
        return "stop"
    return "refine"


def build_reflection_graph(llm: LLM, *, threshold: float = 0.85, max_iterations: int = 3):
    """Compila el grafo. `max_iterations` acota el costo: nunca hay más de N generaciones."""
    if max_iterations < 1:
        raise ValueError("max_iterations debe ser >= 1")

    def generate(state: ReflectionState) -> dict:
        iteration = state.get("iteration", 0) + 1
        prompt = f"Tarea: {state['task']}"
        critique = state.get("critique")
        if critique:
            prompt += (
                f"\n\nBorrador previo:\n{state.get('draft', '')}\n"
                f"Falta: {critique.get('missing') or '-'}\n"
                f"Sobra: {critique.get('superfluous') or '-'}\n"
                "Escribí una versión mejorada que corrija lo anterior."
            )
        try:
            draft = llm.generate(prompt, label="generate")
        except BudgetExceeded as exc:
            return {"halt": "budget", "halt_detail": str(exc)}
        return {"draft": draft, "iteration": iteration}

    def evaluate(state: ReflectionState) -> dict:
        prompt = (
            f"Tarea: {state['task']}\n\nRespuesta a evaluar:\n{state['draft']}\n\n"
            "Puntuá la respuesta de 0 a 1 e indicá qué falta y qué sobra."
        )
        try:
            ev = structured_with_retry(llm, prompt, Evaluation, label="evaluate")
        except BudgetExceeded as exc:
            return {"halt": "budget", "halt_detail": str(exc)}
        except StructuredOutputError as exc:
            return {"halt": "evaluator", "halt_detail": str(exc)}
        return {
            "score": ev.score,
            "critique": ev.model_dump(),
            "history": [{"iteration": state["iteration"], "score": ev.score}],
        }

    def accept(state: ReflectionState) -> dict:
        return {"status": "accepted"}

    def stop(state: ReflectionState) -> dict:
        return {"status": "max_iterations"}

    def halt(state: ReflectionState) -> dict:
        reason = state.get("halt")
        return {"status": "budget_exceeded" if reason == "budget" else "evaluator_failed"}

    def after_generate(state: ReflectionState) -> str:
        return "halt" if state.get("halt") else "evaluate"

    def after_evaluate(state: ReflectionState) -> str:
        if state.get("halt"):
            return "halt"
        return route_by_score(state, threshold=threshold, max_iterations=max_iterations)

    graph = StateGraph(ReflectionState)
    graph.add_node("generate", generate)
    graph.add_node("evaluate", evaluate)
    graph.add_node("accept", accept)
    graph.add_node("stop", stop)
    graph.add_node("halt", halt)
    graph.add_edge(START, "generate")
    graph.add_conditional_edges(
        "generate", after_generate, {"evaluate": "evaluate", "halt": "halt"}
    )
    graph.add_conditional_edges(
        "evaluate",
        after_evaluate,
        {"accept": "accept", "refine": "generate", "stop": "stop", "halt": "halt"},
    )
    graph.add_edge("accept", END)
    graph.add_edge("stop", END)
    graph.add_edge("halt", END)
    return graph.compile()


def run_reflection(
    llm: LLM, task: str, *, threshold: float = 0.85, max_iterations: int = 3
) -> ReflectionState:
    """Atajo: compila y ejecuta el grafo para una tarea."""
    app = build_reflection_graph(llm, threshold=threshold, max_iterations=max_iterations)
    return app.invoke(
        {"task": task, "iteration": 0, "history": []},
        config={"recursion_limit": max_iterations * 3 + 5},
    )
