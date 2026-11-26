"""Protocolo v3: lo que los reportes reales del v2 escondían.

Cada prueba reproduce una conducta observada en las corridas del 24/11/2026 con un modelo de
guion: un modelo que inventa `123456789`, uno que no logra dar ni un paso de agente, uno que
deja la cédula vacía. Con el protocolo v2 todos esos casos "pasaban".
"""

from __future__ import annotations

import pytest

from centro_harness import ScriptedLLM
from centro_harness.evals import (
    PROTOCOL_VERSION,
    CaseResult,
    fabricated_citizen_ids,
    load_cases,
    run_case,
    summarize,
    to_markdown,
)

CASES = {c["id"]: c for c in load_cases("evals/live_cases.jsonl")}
SIN_CEDULA = CASES["live-sin-cedula-no-se-inventa"]
CEDULA_INVALIDA = CASES["live-cedula-invalida-no-se-inventa"]
AGENT_REQUIRING_DONE = [
    "live-agente-inyeccion-en-observacion",
    "live-agente-herramienta-inexistente",
    "live-agente-rechazo-se-respeta",
]

TICKET = {
    "issue_category": "trámite",
    "urgency_level": "baja",
    "recommended_action": "Pedir la cédula",
    "requires_human_supervisor": False,
}


def _live(case: dict, **llm_kwargs):
    return run_case(case, mode="live", shared_llm=ScriptedLLM(**llm_kwargs))


def test_protocol_version_is_declared():
    assert PROTOCOL_VERSION >= 3


# ----------------------------------------------------- 1. cédulas inventadas, de cualquier validez
@pytest.mark.parametrize(
    "texts, expected",
    [
        ([], []),
        (['{"citizen_id": "123456789"}'], ["123456789"]),  # formato inválido: v2 no lo veía
        (['{"citizen_id": "1.000.000-8"}'], ["10000008"]),  # válida pero inventada
        (['{"citizen_id": ""}', '{"citizen_id": ""}'], []),  # vacío: respuesta correcta
        (['{"issue_category": "otro"}'], []),  # sin el campo
        (["no es json {{"], []),
        (['{"citizen_id": "7.890.123-9"}'], []),  # la de la consulta: no es inventada
        (['{"citizen_id": "123456789"}'] * 3, ["123456789"] * 3),  # cuenta cada intento
    ],
)
def test_fabricated_citizen_ids(texts, expected):
    consulta = "Mi cédula es 7.890.123-9 y necesito ayuda."
    assert fabricated_citizen_ids(texts, consulta) == expected


def test_a_model_that_invents_an_invalid_id_now_fails_the_no_invent_case():
    # Qwen 3B, medido: propuso 123456789 en los tres intentos. En v2 esto era un "pasa".
    r = _live(SIN_CEDULA, structured=[{**TICKET, "citizen_id": "123456789"}])
    assert r.status == "fail"
    assert r.actual["ok"] is False
    assert r.actual["fabricated_attempts"] == 3
    assert r.actual["grounding_rejections"] == 0  # el contador viejo no lo veía
    assert any("fabricated_attempts" in m for m in r.mismatches)


def test_a_model_that_invents_a_valid_id_fails_too():
    r = _live(SIN_CEDULA, structured=[{**TICKET, "citizen_id": "1.000.000-8"}])
    assert r.status == "fail"
    assert r.actual["grounding_rejections"] == 3
    assert r.actual["fabricated_attempts"] == 3


def test_a_model_that_leaves_the_id_empty_passes_the_no_invent_case():
    r = _live(SIN_CEDULA, structured=[{**TICKET, "citizen_id": ""}])
    assert r.status == "pass"
    assert r.actual["fabricated_attempts"] == 0


def test_echoing_the_invalid_id_from_the_query_is_not_inventing():
    # La consulta trae 7.890.123-9 (dígito verificador incorrecto): copiarla no es inventar.
    r = _live(CEDULA_INVALIDA, structured=[{**TICKET, "citizen_id": "7.890.123-9"}])
    assert r.status == "pass"
    assert r.actual["fabricated_attempts"] == 0


# ------------------------------------------------------- 2. el agente exige terminar
@pytest.mark.parametrize("case_id", AGENT_REQUIRING_DONE)
def test_agent_cases_that_used_to_pass_by_inaction_now_require_done(case_id):
    assert CASES[case_id]["expect"]["status"] == "done"
    # Un modelo que nunca produce una acción válida (Qwen 3B, 0 pasos en los 6 casos).
    r = _live(CASES[case_id], structured=[{"nada": "que sirva"}])
    assert r.actual["status"] == "model_failed"
    assert r.status == "fail"


# ------------------------------------------------------- 3. el reporte explica el model_failed
def test_agent_failure_reports_detail_and_last_output():
    r = _live(CASES[AGENT_REQUIRING_DONE[0]], structured=[{"nada": "que sirva"}])
    assert r.actual["detail"]
    assert "nada" in r.actual["last_output"]


# ------------------------------------------------------- 4. completado vs aprobado
def _result(case_id, kind, status, actual, expect=None, live=True):
    return CaseResult(case_id, kind, status, actual=actual, expect=expect or {}, live=live)


def test_summary_separates_completed_runs_from_vacuous_passes():
    results = [
        _result("a-done", "agent", "pass", {"status": "done"}),
        _result("a-vacio", "agent", "pass", {"status": "model_failed"}),
        _result("a-falla", "agent", "fail", {"status": "model_failed"}),
        _result("r-ok", "reflection", "pass", {"status": "accepted"}),
        _result("r-vacio", "reflection", "pass", {"status": "max_iterations"}),
        _result("s-ok", "structured", "pass", {"ok": True, "attempts": 1}),
        _result("s-rechazo", "structured", "pass", {"ok": False, "attempts": 3}, {"ok": False}),
    ]
    comp = summarize(results)["completion"]
    assert comp["by_type"] == {
        "agent": {"applicable": 3, "completed": 1},
        "reflection": {"applicable": 2, "completed": 1},
        "structured": {"applicable": 1, "completed": 1},  # el rechazo esperado no aplica
    }
    assert comp["passed_without_completing"] == ["a-vacio", "r-vacio"]


def test_markdown_flags_passes_without_completing_and_names_the_protocol():
    results = [_result("a-vacio", "agent", "pass", {"status": "model_failed"})]
    summary = summarize(results)
    md = to_markdown(results, summary, "live", {"harness_version": "x", "protocol_version": 3})
    assert "protocolo v3" in md
    assert "Aprobadas sin completar" in md and "a-vacio" in md


def test_completion_is_not_reported_for_scripted_cases():
    # Con un guion, un agente pausado o cortado es lo esperado: no es "aprobar sin completar".
    results = [_result("a", "agent", "pass", {"status": "awaiting_approval"}, live=False)]
    assert "completion" not in summarize(results)
