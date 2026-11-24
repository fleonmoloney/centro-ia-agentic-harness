"""Protocolo v2: prompt con ejemplo, anclaje de la cédula y truncamiento como fallo."""

import json
import threading

import pytest

from centro_harness.evals import TRUNCATION_FAILS_FROM, main, run_case
from centro_harness.llm import (
    TruncatedResponse,
    example_skeleton,
    schema_prompt,
    structured_from_text,
)
from centro_harness.mockmodel import make_server
from centro_harness.schemas import (
    InstitutionalTicket,
    StructuredOutputError,
    ci_grounding_check,
    structured_with_retry,
)

VALID = {
    "citizen_id": "1.234.567-2", "issue_category": "salud", "urgency_level": "alta",
    "recommended_action": "Derivar a ASSE",
}  # fmt: skip


# ------------------------------------------------------------------ estilo de prompt
def test_skeleton_has_the_schema_keys_and_enum_options():
    skeleton = example_skeleton(InstitutionalTicket)
    assert set(skeleton) == set(InstitutionalTicket.model_fields)
    assert skeleton["issue_category"] == "salud | empleo | trámite | otro"
    assert skeleton["requires_human_supervisor"] is False


def test_example_prompt_shows_the_shape_not_the_json_schema():
    v1 = schema_prompt("Consulta", InstitutionalTicket, "schema")
    v2 = schema_prompt("Consulta", InstitutionalTicket, "example")
    assert '"properties"' in v1 and '"properties"' not in v2
    assert "Respondé SOLO" in v2  # el servidor simulado y los casos dependen de este marcador
    with pytest.raises(ValueError):
        schema_prompt("x", InstitutionalTicket, "otro")


def test_model_that_echoes_the_schema_fails_v1_but_not_v2():
    """Reproduce el modo de falla medido con Qwen2.5-3B: devolvió el esquema, no una instancia.

    El 'modelo' es un guion que imita ese comportamiento; prueba el cambio de prompt, no
    la capacidad de ningún modelo real.
    """

    def generate(prompt):
        if '"properties"' in prompt:
            return json.dumps(InstitutionalTicket.model_json_schema(), ensure_ascii=False)
        return json.dumps(VALID, ensure_ascii=False)

    with pytest.raises(ValueError):
        structured_from_text(generate, "x", InstitutionalTicket, style="schema")
    assert structured_from_text(generate, "x", InstitutionalTicket, style="example").citizen_id


# ------------------------------------------------------------------ anclaje de la cédula
class Inventor:
    """Siempre devuelve una cédula con verificador válido, figure o no en la consulta."""

    def structured(self, prompt, schema, *, label=""):
        return schema.model_validate({**VALID, "citizen_id": "1.000.000-8"})


def test_grounding_rejects_a_valid_but_invented_id():
    check = ci_grounding_check("Hola, necesito ayuda pero no tengo la cédula.")
    with pytest.raises(StructuredOutputError) as err:
        structured_with_retry(Inventor(), "x", InstitutionalTicket, check=check)
    assert "no figura" in err.value.last_error
    assert check.rejections == 3


def test_grounding_accepts_the_id_in_any_dotted_or_plain_format():
    for text in ("Mi cédula es 1.234.567-2.", "CI 12345672", "ci: 1 234 567-2"):
        check = ci_grounding_check(text)
        assert check(InstitutionalTicket.model_validate(VALID)) is None, text


def test_without_the_check_the_invented_id_passes_which_is_the_original_problem():
    out = structured_with_retry(Inventor(), "x", InstitutionalTicket)
    assert out.citizen_id == "1.000.000-8"


# ------------------------------------------------------------------ respuestas vacías por tokens
def _case():
    return {"id": "t", "type": "structured", "schema": "InstitutionalTicket", "live": True,
            "prompt": "x", "expect": {"ok": True}}  # fmt: skip


class Truncated:
    def __init__(self, max_tokens):
        self.max_tokens = max_tokens

    def structured(self, prompt, schema, *, label=""):
        raise TruncatedResponse("vacío", self.max_tokens)

    def generate(self, prompt, *, label=""):
        raise TruncatedResponse("vacío", self.max_tokens)


def test_truncation_at_a_high_cap_counts_as_a_model_failure():
    res = run_case(_case(), mode="live", shared_llm=Truncated(TRUNCATION_FAILS_FROM))
    assert res.status == "fail" and "agotó" in res.detail


def test_truncation_at_a_low_cap_is_a_configuration_error_not_a_model_failure():
    res = run_case(_case(), mode="live", shared_llm=Truncated(512))
    assert res.status == "error"


# ------------------------------------------------------------------ de punta a punta
def test_cli_runs_the_whole_live_suite_with_prompt_style_example(tmp_path):
    server = make_server(0, "normal")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    out = tmp_path / "r.json"
    try:
        code = main(
            ["--cases", "evals/live_cases.jsonl", "--llm", "compat", "--model", "mock",
             "--base-url", f"http://127.0.0.1:{server.server_port}/v1", "--retries", "0",
             "--prompt-style", "example", "--out-json", str(out)]
        )  # fmt: skip
    finally:
        server.shutdown()
        server.server_close()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["meta"]["prompt_style"] == "example"
    assert code == 0, [r for r in data["results"] if r["status"] != "pass"][:3]
