import json
import threading

import pytest

from centro_harness.compare import compare, main
from centro_harness.evals import main as evals_main
from centro_harness.mockmodel import make_server


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cmp")
    paths = []
    for behavior, label in (("normal", "mock-a"), ("obedient", "mock-b")):
        server = make_server(0, behavior)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        out = tmp / f"{label}.json"
        evals_main(
            ["--cases", "evals/live_cases.jsonl", "--llm", "compat", "--model", "mock",
             "--base-url", f"http://127.0.0.1:{server.server_port}/v1", "--repeats", "2",
             "--retries", "0", "--label", label, "--out-json", str(out)]
        )  # fmt: skip
        server.shutdown()
        server.server_close()
        paths.append(str(out))
    return paths


def test_table_has_one_row_per_model_and_wilson_intervals(reports):
    text = compare(reports)
    assert "mock-a" in text and "mock-b" in text
    assert text.count("\n| mock-") == 2
    assert "Wilson" in text
    assert "(0." in text  # intervalo dentro de la celda


def test_overlapping_intervals_are_flagged_as_no_evidence(reports):
    text = compare(reports)
    assert "NO hay evidencia de diferencia" in text  # 0.9x vs 1.0 con pocas muestras


def test_scripted_reports_are_rejected(tmp_path):
    out = tmp_path / "s.json"
    evals_main(["--cases", "evals/cases.jsonl", "--out-json", str(out)])
    assert main([str(out)]) == 2


def test_mismatched_cases_are_called_out(reports, tmp_path):
    data = json.loads(open(reports[0], encoding="utf-8").read())
    data["summary"]["per_case"].pop(next(iter(data["summary"]["per_case"])))
    altered = tmp_path / "alt.json"
    altered.write_text(json.dumps(data), encoding="utf-8")
    assert "NO cubren los mismos casos" in compare([reports[0], str(altered)])
