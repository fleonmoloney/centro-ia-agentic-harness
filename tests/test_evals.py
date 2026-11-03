import json
from pathlib import Path

from centro_harness.evals import load_cases, main, run_case, summarize

CASES = Path(__file__).resolve().parent.parent / "evals" / "cases.jsonl"


def test_all_golden_cases_pass_in_scripted_mode():
    results = [run_case(c, mode="scripted", shared_llm=None) for c in load_cases(CASES)]
    failed = [(r.id, r.mismatches) for r in results if r.status == "fail"]
    assert not failed, failed
    assert summarize(results)["ran"] >= 15


def test_live_cases_are_skipped_without_a_real_llm():
    live = [c for c in load_cases(CASES) if c.get("live")]
    assert live, "debería haber al menos un caso live"
    assert all(run_case(c, mode="scripted", shared_llm=None).status == "skipped" for c in live)


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


def test_cli_exit_code_and_reports(tmp_path):
    out_json = tmp_path / "r.json"
    out_md = tmp_path / "r.md"
    code = main(["--cases", str(CASES), "--out-json", str(out_json), "--out-md", str(out_md)])
    assert code == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["summary"]["failed"] == 0
    assert "# Reporte de evals" in out_md.read_text(encoding="utf-8")


def test_cli_returns_1_when_a_case_fails(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            {
                "id": "mal",
                "type": "cost",
                "tools": 2,
                "tasks": 2,
                "expect": {"min_savings_pct": 99},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["--cases", str(bad)]) == 1
