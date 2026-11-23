import json

import pytest

from centro_harness.budget import Budget, BudgetExceeded
from centro_harness.jsonutil import extract_json
from centro_harness.llm import ScriptedLLM
from centro_harness.reflection import run_reflection
from centro_harness.schemas import (
    InstitutionalTicket,
    StructuredOutputError,
    structured_with_retry,
)
from centro_harness.tracing import TracedLLM, Tracer

VALID = {
    "citizen_id": "1.234.567-2",
    "issue_category": "salud",
    "urgency_level": "alta",
    "recommended_action": "Derivar",
}


# ------------------------------------------------------------------ extract_json
def test_extract_json_plain_fenced_and_surrounded():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Claro:\n```json\n{"a": {"b": 2}}\n```\nfin') == {"a": {"b": 2}}
    assert extract_json('Respuesta: {"a": 1} espero que sirva') == {"a": 1}
    assert extract_json('basura { no json } y luego {"ok": true}') == {"ok": True}


@pytest.mark.parametrize("text", ["", "sin objeto", "[1, 2, 3]", "{roto"])
def test_extract_json_rejects_text_without_an_object(text):
    with pytest.raises(ValueError):
        extract_json(text)


# ------------------------------------------------------------------ reparación
class _BrokenThenValid(ScriptedLLM):
    def __init__(self, failures):
        super().__init__(structured=[VALID])
        self.failures = failures
        self.calls = 0

    def structured(self, prompt, schema, *, label=""):
        self.calls += 1
        if self.calls <= self.failures:
            raise json.JSONDecodeError("Expecting value", "texto libre", 0)
        return super().structured(prompt, schema, label=label)


def test_json_errors_are_repaired_not_fatal():
    llm = _BrokenThenValid(failures=2)
    assert structured_with_retry(llm, "x", InstitutionalTicket).citizen_id == "1.234.567-2"
    assert llm.calls == 3


def test_json_errors_still_respect_the_attempt_limit():
    llm = _BrokenThenValid(failures=99)
    with pytest.raises(StructuredOutputError) as info:
        structured_with_retry(llm, "x", InstitutionalTicket, max_attempts=2)
    assert info.value.attempts == 2
    assert "JSONDecodeError" in info.value.last_error


# ------------------------------------------------------------------ presupuesto
def test_budget_blocks_the_call_that_would_exceed_max_calls():
    inner = ScriptedLLM(texts=["a"])
    llm = TracedLLM(inner, Tracer(), Budget(max_calls=2))
    llm.generate("1")
    llm.generate("2")
    with pytest.raises(BudgetExceeded) as info:
        llm.generate("3")
    assert info.value.reason == "max_calls"
    assert len(inner.prompts) == 2  # la tercera llamada nunca llegó al modelo


def test_budget_max_tokens_uses_what_was_already_spent():
    llm = TracedLLM(ScriptedLLM(texts=["x" * 400]), Tracer(), Budget(max_tokens=100))
    llm.generate("p" * 400)  # ~100 de entrada + ~100 de salida
    with pytest.raises(BudgetExceeded) as info:
        llm.generate("otra")
    assert info.value.reason == "max_tokens"


def test_budget_max_seconds():
    llm = TracedLLM(ScriptedLLM(texts=["a"]), Tracer(), Budget(max_seconds=0.0))
    with pytest.raises(BudgetExceeded) as info:
        llm.generate("x")
    assert info.value.reason == "max_seconds"


def test_no_budget_means_no_limit():
    llm = TracedLLM(ScriptedLLM(texts=["a"]), Tracer())
    for _ in range(50):
        llm.generate("x")


def test_scripted_tokens_are_labelled_as_estimated():
    tracer = Tracer()
    TracedLLM(ScriptedLLM(texts=["a"]), tracer).generate("x")
    assert tracer.tokens_source == "estimated"
    assert tracer.summary()["tokens_source"] == "estimated"


# ------------------------------------------------------------------ reflexión: cortes ordenados
def test_reflection_stops_cleanly_when_the_budget_runs_out():
    inner = ScriptedLLM(
        texts=["borrador 1", "borrador 2", "borrador 3"],
        structured=[{"score": 0.3, "missing": "mas", "superfluous": ""}],
    )
    llm = TracedLLM(inner, Tracer(), Budget(max_calls=3))
    out = run_reflection(llm, "explicar", max_iterations=5)
    assert out["status"] == "budget_exceeded"
    assert out["iteration"] == 2
    assert out["draft"] == "borrador 2"  # se conserva el mejor borrador disponible
    assert "max_calls" in out["halt_detail"]


def test_reflection_reports_a_failing_evaluator_instead_of_crashing():
    llm = ScriptedLLM(
        texts=["borrador 1"],
        structured=[{"score": 5.0}],  # fuera de rango: nunca valida
    )
    out = run_reflection(llm, "explicar", max_iterations=3)
    assert out["status"] == "evaluator_failed"
    assert out["draft"] == "borrador 1"
