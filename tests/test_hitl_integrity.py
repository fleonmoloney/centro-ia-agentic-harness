"""Integridad de la puerta HITL: cada prueba nace de una falla reproducida en la v0.2.0."""

import json

import pytest

from centro_harness.hitl import (
    HITLGate,
    NoPendingRequestError,
    RequestMismatchError,
    Risk,
    ThreadBusyError,
    ToolRegistry,
    ToolSpec,
    redact,
    scrub_text,
)

CI = "1.234.567-2"  # cédula con verificador válido


def _gate(**kwargs):
    executed = []
    registry = ToolRegistry()

    def borrar(**kw):
        executed.append(kw)
        return f"borrado {kw.get('id')}"

    registry.register(ToolSpec(name="borrar", risk=Risk.HIGH, description="Borra"), borrar)
    registry.register(
        ToolSpec(name="ficha", risk=Risk.LOW),
        lambda **kw: f"ficha de {kw['datos']['citizen_id']}: sin deudas",
    )
    registry.register(ToolSpec(name="rota", risk=Risk.LOW), lambda **kw: 1 / 0)
    g = HITLGate(registry, **kwargs)
    g.executed = executed
    return g


# ------------------------------------------------------------------ 1. redacción
def test_redact_is_recursive():
    data = {"datos": {"citizen_id": CI, "otro": [{"citizen_id": CI}]}}
    out = redact(data, frozenset({"citizen_id"}))
    assert out == {"datos": {"citizen_id": "***", "otro": [{"citizen_id": "***"}]}}


def test_scrub_masks_valid_ci_in_free_text_and_leaves_other_numbers():
    text = f"ficha de {CI} y pedido 1.234.567-8 y total 12.345"
    scrubbed = scrub_text(text)
    assert CI not in scrubbed
    assert "*.***.567-2" in scrubbed
    assert "1.234.567-8" in scrubbed  # no es una cédula válida: no se toca
    assert "12.345" in scrubbed


def test_audit_masks_nested_arguments_and_tool_result():
    g = _gate()
    out = g.submit("ficha", {"datos": {"citizen_id": CI}}, thread_id="r1")
    assert out.record.arguments == {"datos": {"citizen_id": "***"}}
    assert CI not in (out.record.result or "")
    assert "sin deudas" in out.record.result


# ----------------------------------------------------------- 2. aprobación ligada a la solicitud
def test_second_submit_on_a_pending_thread_is_refused_and_keeps_the_first():
    g = _gate()
    first = g.submit("borrar", {"id": "A"}, thread_id="dup")
    with pytest.raises(ThreadBusyError):
        g.submit("borrar", {"id": "B"}, thread_id="dup")
    g.resume(thread_id="dup", request_id=first.request_id, approved=True, reviewer="ana")
    assert g.executed == [{"id": "A"}]


def test_resume_with_a_wrong_request_id_does_not_execute():
    g = _gate()
    g.submit("borrar", {"id": "A"}, thread_id="x")
    with pytest.raises(RequestMismatchError):
        g.resume(thread_id="x", request_id="otra-solicitud", approved=True, reviewer="ana")
    assert g.executed == []


def test_pending_payload_carries_the_request_id():
    g = _gate()
    out = g.submit("borrar", {"id": "A"}, thread_id="p")
    assert out.request_id
    assert out.pending["request_id"] == out.request_id


# ------------------------------------------------------------- 3. resume duplicado / inexistente
def test_double_resume_does_not_duplicate_audit_or_execution():
    g = _gate()
    out = g.submit("borrar", {"id": 1}, thread_id="d")
    g.resume(thread_id="d", request_id=out.request_id, approved=True, reviewer="ana")
    with pytest.raises(NoPendingRequestError):
        g.resume(thread_id="d", request_id=out.request_id, approved=True, reviewer="ana")
    assert len(g.executed) == 1
    assert len(g.audit) == 1


def test_resume_on_unknown_thread_is_a_clear_error():
    with pytest.raises(NoPendingRequestError):
        _gate().resume(thread_id="nunca", request_id="x", approved=True, reviewer="ana")


def test_reusing_a_finished_thread_does_not_leak_the_previous_result():
    g = _gate()
    first = g.submit("borrar", {"id": 1}, thread_id="reuse")
    g.resume(thread_id="reuse", request_id=first.request_id, approved=True, reviewer="ana")
    second = g.submit("formatear", {}, thread_id="reuse")  # herramienta desconocida
    assert second.record.decision == "denied_unknown_tool"
    assert second.record.result is None
    assert second.record.reviewer is None


# ------------------------------------------------------------------ 4. herramienta que falla
def test_tool_failure_is_reported_as_failed_not_completed():
    out = _gate().submit("rota", {}, thread_id="f")
    assert out.status == "failed"
    assert out.record.decision == "auto_approved"
    assert out.record.error.startswith("ZeroDivisionError")
    assert out.record.result is None


# ------------------------------------------------------------------ revisores autorizados
def test_reviewer_outside_the_allowlist_is_rejected():
    g = _gate(authorized_reviewers={"ana"})
    out = g.submit("borrar", {"id": 1}, thread_id="a1")
    done = g.resume(thread_id="a1", request_id=out.request_id, approved=True, reviewer="mallory")
    assert done.status == "blocked"
    assert done.record.decision == "rejected_unauthorized"
    assert g.executed == []


def test_reviewer_inside_the_allowlist_is_accepted():
    g = _gate(authorized_reviewers={"ana"})
    out = g.submit("borrar", {"id": 1}, thread_id="a2")
    done = g.resume(thread_id="a2", request_id=out.request_id, approved=True, reviewer="ana")
    assert done.status == "completed"


# ------------------------------------------------------------------ persistencia
def test_audit_path_appends_each_decision(tmp_path):
    path = tmp_path / "audit.jsonl"
    g = _gate(audit_path=path)
    g.submit("ficha", {"datos": {"citizen_id": CI}}, thread_id="w1")
    g.submit("formatear", {}, thread_id="w2")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["decision"] for r in rows] == ["auto_approved", "denied_unknown_tool"]
    assert CI not in path.read_text(encoding="utf-8")


def test_pending_request_survives_a_restart_with_sqlite(tmp_path):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    from centro_harness.hitl import sqlite_checkpointer

    db = tmp_path / "hitl.sqlite"
    g1 = _gate(checkpointer=sqlite_checkpointer(db))
    out = g1.submit("borrar", {"id": 9}, thread_id="persist")
    assert out.status == "pending_approval"
    del g1  # "reinicio": se pierde todo el estado en memoria

    g2 = _gate(checkpointer=sqlite_checkpointer(db))
    pending = g2.pending("persist")
    assert pending and pending["request_id"] == out.request_id
    done = g2.resume(thread_id="persist", request_id=out.request_id, approved=True, reviewer="ana")
    assert done.status == "completed"
    assert g2.executed == [{"id": 9}]
