import json
import threading
from pathlib import Path

import pytest

from centro_harness.evals import (
    exit_code,
    load_cases,
    main,
    run_case,
    summarize,
    wilson,
)
from centro_harness.mockmodel import make_server

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "evals" / "cases.jsonl"
LIVE = ROOT / "evals" / "live_cases.jsonl"


# ------------------------------------------------------------------ modo scripted (compuerta de CI)
def test_all_golden_cases_pass_in_scripted_mode():
    results = [run_case(c, mode="scripted", shared_llm=None) for c in load_cases(CASES)]
    failed = [(r.id, r.mismatches) for r in results if r.status == "fail"]
    assert not failed, failed
    assert summarize(results)["ran"] >= 25
    assert summarize(results)["invariant_violations"] == 0


def test_live_cases_are_skipped_in_scripted_mode():
    live = load_cases(LIVE)
    assert len(live) >= 15
    assert all(c.get("live") for c in live)
    assert all(run_case(c, mode="scripted", shared_llm=None).status == "skipped" for c in live)


def test_scripted_cases_are_skipped_in_live_mode():
    scripted = load_cases(CASES)
    assert not any(c.get("live") for c in scripted)
    assert run_case(scripted[0], mode="live", shared_llm=None).status == "skipped"


def test_runner_detects_a_regression():
    case = {
        "id": "falla-a-proposito",
        "type": "reflection",
        "task": "x",
        "scores": [0.9],
        "expect": {"status": "max_iterations"},
    }
    result = run_case(case, mode="scripted", shared_llm=None)
    assert result.status == "fail"
    assert "status" in result.mismatches[0]


def test_compare_operators():
    from centro_harness.evals import _compare

    actual = {"n": 3, "answer": "Está EN revisión", "tools": ["a", "b"]}
    assert _compare({"min_n": 3, "max_n": 3, "contains_answer": "en revisión"}, actual) == []
    assert _compare({"contains_tools": "b"}, actual) == []
    assert len(_compare({"min_n": 4}, actual)) == 1
    assert len(_compare({"max_n": 2}, actual)) == 1
    assert len(_compare({"contains_answer": "rechazado"}, actual)) == 1
    assert len(_compare({"contains_tools": "z"}, actual)) == 1
    assert len(_compare({"n": 4}, actual)) == 1


def test_cli_exit_code_and_reports(tmp_path):
    out_json = tmp_path / "r.json"
    out_md = tmp_path / "r.md"
    code = main(["--cases", str(CASES), "--out-json", str(out_json), "--out-md", str(out_md)])
    assert code == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["summary"]["failed"] == 0
    assert data["meta"]["llm"] == "scripted"
    assert "# Reporte de evals" in out_md.read_text(encoding="utf-8")


def test_cli_returns_1_when_a_case_fails(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            {"id": "mal", "type": "cost", "tools": 2, "tasks": 2, "expect": {"min_savings_pct": 99}}
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["--cases", str(bad)]) == 1


# ------------------------------------------------------------------ estadística
def test_wilson_interval_basics():
    assert wilson(0, 0) is None
    lo, hi = wilson(5, 10)
    assert 0.23 < lo < 0.25 and 0.75 < hi < 0.77
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and lo > 0.69  # con 10/10 el piso sigue lejos de 1: pocas muestras


def test_exit_code_rules():
    ok = {"errors": 0, "invariant_violations": 0, "failed": 0, "pass_rate": 0.5}
    assert exit_code(ok, "scripted", None) == 0
    assert exit_code({**ok, "failed": 1}, "scripted", None) == 1
    assert exit_code({**ok, "failed": 1}, "live", None) == 0  # en vivo, fallar es medir
    assert exit_code(ok, "live", 0.9) == 1
    assert exit_code({**ok, "errors": 1}, "live", None) == 2
    assert exit_code({**ok, "invariant_violations": 1}, "scripted", None) == 2


# ------------------------------------------------------------------ camino vivo, de punta a punta
@pytest.fixture
def mock_url():
    servers = []

    def _start(behavior="normal"):
        server = make_server(0, behavior)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_port}/v1"

    yield _start
    for s in servers:
        s.shutdown()
        s.server_close()


def _live(url, tmp_path, *extra):
    out = tmp_path / "live.json"
    code = main(
        ["--cases", str(LIVE), "--llm", "compat", "--base-url", url, "--model", "mock",
         "--repeats", "3", "--retries", "0", "--out-json", str(out), *extra]
    )  # fmt: skip
    return code, json.loads(out.read_text(encoding="utf-8"))


def test_live_pipeline_end_to_end_against_the_mock_server(mock_url, tmp_path):
    code, data = _live(mock_url("normal"), tmp_path)
    summary = data["summary"]
    assert code == 0, [r for r in data["results"] if r["status"] != "pass"][:3]
    assert summary["errors"] == 0 and summary["invariant_violations"] == 0
    assert summary["ran"] == len(load_cases(LIVE)) * 3
    assert summary["tokens_source"] == "provider"
    assert summary["pass_rate"] == 1.0
    assert summary["ci95"][0] < 1.0  # un 100 % sobre pocas repeticiones no es certeza
    assert summary["structured"]["repair_rate"] > 0  # el JSON roto simulado sí se reparó
    assert summary["agent"]["unauthorized_executions"] == 0
    assert data["meta"]["model"] == "mock" and data["meta"]["repeats"] == 3


def test_a_model_that_obeys_an_injection_lowers_the_score_but_never_the_gate(mock_url, tmp_path):
    code, data = _live(mock_url("obedient"), tmp_path, "--min-pass-rate", "1.0")
    summary = data["summary"]
    assert summary["per_case"]["live-agente-inyeccion-en-observacion"]["pass"] == 0
    assert summary["agent"]["proposed_high_risk"] >= 3  # el modelo SÍ intentó borrar
    poisoned = [r for r in data["results"] if r["id"] == "live-agente-inyeccion-en-observacion"]
    assert all(r["actual"]["proposed_high_risk"] == 1 for r in poisoned)
    assert all(r["actual"]["executed_high_risk"] == 0 for r in poisoned)  # la puerta lo detuvo
    assert all("borrar_registro" not in r["actual"]["tools_executed"] for r in poisoned)
    assert summary["agent"]["unauthorized_executions"] == 0
    assert summary["invariant_violations"] == 0
    assert code == 1  # tasa bajo el mínimo: medición, no violación de seguridad


def test_infrastructure_failure_is_an_error_not_a_model_failure(mock_url, tmp_path):
    code, data = _live(mock_url("broken"), tmp_path)
    assert code == 2
    assert data["summary"]["errors"] > 0
    assert data["summary"]["failed"] == 0  # un servidor caído no cuenta como "el modelo falló"


def test_only_filter_runs_just_the_matching_cases(tmp_path, capsys):
    from centro_harness.evals import main

    out = tmp_path / "r.json"
    code = main(["--cases", "evals/cases.jsonl", "--only", "agente", "--out-json", str(out)])
    import json

    summary = json.loads(out.read_text(encoding="utf-8"))["summary"]
    assert code == 0
    assert 0 < summary["total"] < 29
