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


def test_reports_with_different_case_sets_are_not_contrasted(reports, tmp_path):
    """Un 0/6 de los dos casos difíciles no se contrasta con una suite de 19 casos."""
    data = json.loads(open(reports[0], encoding="utf-8").read())
    data["summary"]["per_case"].pop(next(iter(data["summary"]["per_case"])))
    data["summary"]["pass_rate"] = 0.0  # fuerza una diferencia
    altered = tmp_path / "alt2.json"
    altered.write_text(json.dumps(data), encoding="utf-8")
    text = compare([reports[1], str(altered)])
    assert "no comparables (casos distintos)" in text
    assert "intervalos no se solapan" not in text


def test_different_protocols_or_ticket_variants_are_not_contrasted(reports, tmp_path):
    for key, value, expected in (
        ("protocol_version", 99, "protocolo distinto"),
        ("ticket_schema", "experimental", "variante de esquema distinto"),
    ):
        data = json.loads(open(reports[0], encoding="utf-8").read())
        data["meta"][key] = value
        data["summary"]["pass_rate"] = 0.0
        altered = tmp_path / f"{key}.json"
        altered.write_text(json.dumps(data), encoding="utf-8")
        text = compare([reports[1], str(altered)])
        assert expected in text


def test_a_v5_neutral_report_is_comparable_with_v6_but_a_v5_standard_one_is_not(reports, tmp_path):
    from centro_harness.compare import _why_not_comparable

    def meta_of(path, **meta):
        data = json.loads(open(path, encoding="utf-8").read())
        data["meta"].update(meta)
        return data

    v6 = meta_of(reports[0], protocol_version=6, ticket_schema="neutral")
    v5_neutral = meta_of(reports[1], protocol_version=5, ticket_schema="neutral")
    v5_standard = meta_of(reports[1], protocol_version=5, ticket_schema="standard")
    assert _why_not_comparable(v6, v5_neutral) is None
    assert _why_not_comparable(v6, v5_standard) == "protocolo distinto"
