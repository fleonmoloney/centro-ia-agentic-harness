"""Variante `neutral`: cambia lo que lee el modelo, no lo que valida el arnés."""

import json

import pytest
from pydantic import ValidationError

from centro_harness import evals
from centro_harness.llm import example_skeleton, schema_prompt
from centro_harness.schemas import InstitutionalTicket, InstitutionalTicketNeutral


@pytest.fixture(autouse=True)
def _restore_variant():
    yield
    evals._set_ticket_variant(evals.DEFAULT_TICKET_VARIANT)


def test_neutral_description_drops_the_valid_check_digit_hint():
    standard = schema_prompt("T", InstitutionalTicket, "example")
    neutral = schema_prompt("T", InstitutionalTicketNeutral, "example")
    assert "verificador válido" in standard
    assert "verificador válido" not in neutral
    assert "tal como figura en la consulta" in neutral


def test_neutral_keeps_the_same_skeleton_keys():
    assert list(example_skeleton(InstitutionalTicketNeutral)) == list(
        example_skeleton(InstitutionalTicket)
    )


@pytest.mark.parametrize("cls", [InstitutionalTicket, InstitutionalTicketNeutral])
def test_both_variants_validate_identically(cls):
    ok = dict(citizen_id="1.234.567-2", issue_category="salud", urgency_level="baja",
              recommended_action="derivar")
    assert cls.model_validate(ok).citizen_id == "1.234.567-2"
    for bad in ("1.234.567-3", "", "0123456789"):
        with pytest.raises(ValidationError):
            cls.model_validate({**ok, "citizen_id": bad})


def test_variant_is_selected_only_for_the_ticket_and_defaults_to_neutral():
    assert evals.DEFAULT_TICKET_VARIANT == "neutral"
    assert evals._schema_for("InstitutionalTicket") is InstitutionalTicketNeutral
    evals._set_ticket_variant("standard")
    assert evals._schema_for("InstitutionalTicket") is InstitutionalTicket
    assert evals._schema_for("MedicalProtocol") is evals.SCHEMAS["MedicalProtocol"]
    with pytest.raises(ValueError):
        evals._set_ticket_variant("otra")


def test_cli_records_the_variant_in_the_report(tmp_path):
    out = tmp_path / "r.json"
    evals.main(["--llm", "scripted", "--out-json", str(out)])
    assert json.loads(out.read_text(encoding="utf-8"))["meta"]["ticket_schema"] == "neutral"
    evals.main(["--llm", "scripted", "--ticket-schema", "standard", "--out-json", str(out)])
    assert json.loads(out.read_text(encoding="utf-8"))["meta"]["ticket_schema"] == "standard"
