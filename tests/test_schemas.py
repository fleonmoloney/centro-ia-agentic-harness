import pytest
from pydantic import ValidationError

from centro_harness.llm import ScriptedLLM
from centro_harness.schemas import (
    InstitutionalTicket,
    MedicalProtocol,
    StructuredOutputError,
    ci_check_digit,
    format_ci,
    is_valid_ci,
    mask_ci,
    structured_with_retry,
)

VALID = {
    "citizen_id": "1.234.567-2",
    "issue_category": "empleo",
    "urgency_level": "media",
    "recommended_action": "Orientar a DINAE",
}


def test_ci_check_digit_known_value():
    assert ci_check_digit("1234567") == 2


@pytest.mark.parametrize("raw", ["1.234.567-2", "12345672", "1234567-2"])
def test_valid_ci_formats(raw):
    assert is_valid_ci(raw)


@pytest.mark.parametrize("raw", ["1.234.567-8", "", "abc", "123", "123456789012"])
def test_invalid_ci(raw):
    assert not is_valid_ci(raw)


def test_ci_is_normalized_and_masked():
    assert format_ci("12345672") == "1.234.567-2"
    assert mask_ci("1.234.567-2") == "*.***.567-2"
    assert mask_ci("") == "***"


def test_ticket_rejects_bad_ci():
    with pytest.raises(ValidationError):
        InstitutionalTicket(**{**VALID, "citizen_id": "1.234.567-8"})


def test_ticket_normalizes_ci():
    assert InstitutionalTicket(**{**VALID, "citizen_id": "12345672"}).citizen_id == "1.234.567-2"


@pytest.mark.parametrize(
    ("category", "urgency", "expected"),
    [
        ("salud", "alta", True),
        ("trámite", "crítica", True),
        ("salud", "media", False),
        ("empleo", "alta", False),
    ],
)
def test_ticket_escalation_rule(category, urgency, expected):
    ticket = InstitutionalTicket(**{**VALID, "issue_category": category, "urgency_level": urgency})
    assert ticket.requires_human_supervisor is expected


def test_medical_protocol_low_confidence_requires_review():
    p = MedicalProtocol(
        protocol_id="P-1",
        patient_age_range="18-65",
        contraindications=[],
        confidence_level="medium",
    )
    assert p.requires_clinician_review is True


def test_retry_repairs_after_validation_error():
    bad = {**VALID, "citizen_id": "1.234.567-8"}
    llm = ScriptedLLM(structured=[bad, VALID])
    ticket = structured_with_retry(llm, "extraer", InstitutionalTicket)
    assert ticket.citizen_id == "1.234.567-2"
    # el segundo prompt le muestra el error al modelo
    assert "no validó" in llm.prompts[1]
    assert "citizen_id" in llm.prompts[1]


def test_retry_is_bounded():
    llm = ScriptedLLM(structured=[{**VALID, "citizen_id": "1.234.567-8"}])
    with pytest.raises(StructuredOutputError) as info:
        structured_with_retry(llm, "extraer", InstitutionalTicket, max_attempts=2)
    assert info.value.attempts == 2
    assert len(llm.prompts) == 2
