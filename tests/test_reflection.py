import pytest

from centro_harness.llm import ScriptedLLM
from centro_harness.reflection import build_reflection_graph, route_by_score, run_reflection


def _llm(scores):
    return ScriptedLLM(
        texts=["borrador 1", "borrador 2", "borrador 3", "borrador 4"],
        structured=[{"score": s, "missing": "ejemplos", "superfluous": ""} for s in scores],
    )


def test_router_accepts_at_threshold():
    state = {"score": 0.85, "iteration": 1}
    assert route_by_score(state, threshold=0.85, max_iterations=3) == "accept"


def test_router_refines_below_threshold():
    state = {"score": 0.6, "iteration": 1}
    assert route_by_score(state, threshold=0.85, max_iterations=3) == "refine"


def test_router_stops_at_iteration_cap():
    state = {"score": 0.1, "iteration": 3}
    assert route_by_score(state, threshold=0.85, max_iterations=3) == "stop"


def test_graph_converges_and_records_history():
    out = run_reflection(_llm([0.5, 0.7, 0.9]), "explicar", max_iterations=3)
    assert out["status"] == "accepted"
    assert out["iteration"] == 3
    assert [h["score"] for h in out["history"]] == [0.5, 0.7, 0.9]
    assert out["draft"] == "borrador 3"


def test_graph_respects_cost_cap():
    llm = _llm([0.2])
    out = run_reflection(llm, "explicar", max_iterations=2)
    assert out["status"] == "max_iterations"
    assert out["iteration"] == 2
    generate_calls = [p for p in llm.prompts if p.startswith("Tarea:") and "evaluar" not in p]
    assert len(generate_calls) >= 2


def test_feedback_reaches_next_generation_prompt():
    llm = _llm([0.3, 0.95])
    run_reflection(llm, "explicar", max_iterations=3)
    refine_prompts = [p for p in llm.prompts if "Borrador previo" in p]
    assert len(refine_prompts) == 1
    assert "ejemplos" in refine_prompts[0]


def test_invalid_iteration_cap():
    with pytest.raises(ValueError):
        build_reflection_graph(_llm([0.5]), max_iterations=0)
