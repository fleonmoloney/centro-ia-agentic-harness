import json

import pytest

from centro_harness.cost_model import TaskSpec, ToolDoc, compare_tool_assignment
from centro_harness.llm import ScriptedLLM
from centro_harness.schemas import InstitutionalTicket, structured_with_retry
from centro_harness.tracing import TracedLLM, Tracer, estimate_tokens


def _tools(n):
    return [ToolDoc(f"t{i}", "descripción de la herramienta " * 4) for i in range(n)]


def test_task_centric_is_cheaper_with_many_tools():
    tasks = [TaskSpec("consultar", f"t{i % 6}") for i in range(4)]
    cmp = compare_tool_assignment(_tools(6), tasks)
    assert cmp.task_centric_tokens < cmp.agent_centric_tokens
    assert cmp.savings_pct > 0


def test_savings_grow_with_number_of_tools():
    tasks = [TaskSpec("consultar", "t0")]
    small = compare_tool_assignment(_tools(2), tasks).savings_pct
    large = compare_tool_assignment(_tools(10), tasks).savings_pct
    assert large > small


def test_single_tool_still_saves_selection_step():
    cmp = compare_tool_assignment(_tools(1), [TaskSpec("consultar", "t0")])
    assert cmp.agent_centric_tokens - cmp.task_centric_tokens == 30


def test_undeclared_tool_is_an_error():
    with pytest.raises(ValueError):
        compare_tool_assignment(_tools(1), [TaskSpec("x", "inexistente")])


def test_estimate_tokens_minimum_one():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 40) == 10


def test_tracer_counts_failed_attempts_too():
    tracer = Tracer()
    bad = {
        "citizen_id": "1.234.567-8",
        "issue_category": "otro",
        "urgency_level": "baja",
        "recommended_action": "Archivar",
    }
    good = {**bad, "citizen_id": "1.234.567-2"}
    llm = TracedLLM(ScriptedLLM(structured=[bad, good]), tracer)
    structured_with_retry(llm, "extraer", InstitutionalTicket)
    assert tracer.calls == 2
    assert tracer.errors == 1
    assert tracer.count("structured_error") == 1
    assert tracer.tokens_in > 0


def test_tracer_exports_jsonl(tmp_path):
    tracer = Tracer()
    llm = TracedLLM(ScriptedLLM(texts=["hola"]), tracer)
    llm.generate("saludar", label="saludo")
    path = tmp_path / "trace.jsonl"
    tracer.to_jsonl(path)
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["label"] == "saludo"
    assert row["kind"] == "generate"
