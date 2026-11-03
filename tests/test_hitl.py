import json

import pytest

from centro_harness.hitl import HITLGate, Risk, ToolRegistry, ToolSpec


@pytest.fixture
def gate():
    executed = []
    registry = ToolRegistry()
    registry.register(ToolSpec(name="buscar", risk=Risk.LOW), lambda **kw: "encontrado")

    def borrar(**kw):
        executed.append(kw)
        return "borrado"

    registry.register(
        ToolSpec(name="borrar", risk=Risk.HIGH, description="Borra un registro"), borrar
    )

    def rota(**kw):
        raise RuntimeError("falló")

    registry.register(ToolSpec(name="rota", risk=Risk.LOW), rota)
    g = HITLGate(registry)
    g.executed = executed
    return g


def test_low_risk_runs_without_pause(gate):
    out = gate.submit("buscar", {}, thread_id="t1")
    assert out.status == "completed"
    assert out.record.decision == "auto_approved"
    assert out.record.result == "encontrado"


def test_high_risk_pauses_and_does_not_execute(gate):
    out = gate.submit("borrar", {"id": 7}, thread_id="t2")
    assert out.status == "pending_approval"
    assert out.pending["tool"] == "borrar"
    assert gate.executed == []
    assert gate.audit == []


def test_approval_resumes_and_executes_once(gate):
    gate.submit("borrar", {"id": 7}, thread_id="t3")
    out = gate.resume(thread_id="t3", approved=True, reviewer="ana")
    assert out.status == "completed"
    assert out.record.reviewer == "ana"
    assert gate.executed == [{"id": 7}]
    assert len(gate.audit) == 1


def test_rejection_blocks_execution(gate):
    gate.submit("borrar", {"id": 7}, thread_id="t4")
    out = gate.resume(thread_id="t4", approved=False, reviewer="ana")
    assert out.status == "blocked"
    assert out.record.decision == "rejected"
    assert gate.executed == []


def test_approval_without_reviewer_is_rejected(gate):
    gate.submit("borrar", {"id": 7}, thread_id="t5")
    out = gate.resume(thread_id="t5", approved=True, reviewer=None)
    assert out.record.decision == "rejected_no_reviewer"
    assert gate.executed == []


def test_unknown_tool_is_denied(gate):
    out = gate.submit("formatear", {}, thread_id="t6")
    assert out.status == "blocked"
    assert out.record.decision == "denied_unknown_tool"


def test_tool_error_is_audited_not_hidden(gate):
    out = gate.submit("rota", {}, thread_id="t7")
    assert out.status == "completed"
    assert out.record.result.startswith("error: RuntimeError")


def test_citizen_id_is_redacted_in_prompt_and_audit(gate):
    pending = gate.submit("borrar", {"citizen_id": "1.234.567-2"}, thread_id="t8").pending
    assert pending["arguments"]["citizen_id"] == "***"
    out = gate.resume(thread_id="t8", approved=True, reviewer="ana")
    assert out.record.arguments["citizen_id"] == "***"
    # la herramienta sí recibió el valor real
    assert gate.executed == [{"citizen_id": "1.234.567-2"}]


def test_independent_threads_do_not_interfere(gate):
    gate.submit("borrar", {"id": 1}, thread_id="a")
    gate.submit("borrar", {"id": 2}, thread_id="b")
    gate.resume(thread_id="b", approved=True, reviewer="ana")
    assert gate.executed == [{"id": 2}]


def test_export_audit_writes_jsonl(gate, tmp_path):
    gate.submit("buscar", {}, thread_id="t9")
    target = tmp_path / "audit" / "log.jsonl"
    gate.export_audit(target)
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["tool"] == "buscar"
    assert rows[0]["decision"] == "auto_approved"
