"""Protocolo v4: el prompt del paso del agente mostraba una forma imposible de validar.

Con el estilo `example`, el relleno genérico daba `tool`, `arguments` y `final_answer` como
"<texto>". Los modelos reales que copiaban la forma fallaban en el validador (`model_failed`).
"""

import json

import pytest
from pydantic import ValidationError

from centro_harness.agent import AgentAction
from centro_harness.evals import PROTOCOL_VERSION
from centro_harness.llm import example_skeleton, schema_prompt
from centro_harness.schemas import InstitutionalTicket, MedicalProtocol


def test_protocol_is_v5():
    assert PROTOCOL_VERSION == 5


def test_agent_skeleton_shows_arguments_as_object_and_alternatives():
    shape = example_skeleton(AgentAction)
    assert shape["arguments"] == {}
    assert "null" in shape["tool"] and "null" in shape["final_answer"]


def test_agent_skeleton_has_no_pipe_alternatives():
    """Un modelo de 7B copió `buscar_normativa | null` como nombre de herramienta."""
    shape = example_skeleton(AgentAction)
    assert all("|" not in v for v in shape.values() if isinstance(v, str))


def test_agent_example_prompt_explains_one_or_the_other():
    prompt = schema_prompt("Tarea", AgentAction, "example")
    assert "null si llamás una herramienta" in prompt
    assert "null si das la respuesta final" in prompt


def test_copying_the_old_skeleton_is_invalid_but_the_documented_actions_are_valid():
    with pytest.raises(ValidationError):  # lo que hacía un modelo al copiar el relleno viejo
        AgentAction.model_validate(
            {"reasoning": "x", "tool": "t", "arguments": "<texto>", "final_answer": "<texto>"}
        )
    AgentAction.model_validate({"tool": "t", "arguments": {"id": "1"}, "final_answer": None})
    AgentAction.model_validate({"tool": None, "arguments": {}, "final_answer": "listo"})


def test_other_schemas_keep_the_exact_v3_skeleton():
    """structured y reflection no cambian: sus números de v3 siguen valiendo."""
    assert json.dumps(example_skeleton(InstitutionalTicket), ensure_ascii=False) == (
        '{"citizen_id": "<texto>", "issue_category": "salud | empleo | trámite | otro", '
        '"urgency_level": "baja | media | alta | crítica", "recommended_action": "<texto>", '
        '"requires_human_supervisor": false}'
    )
    assert json.dumps(example_skeleton(MedicalProtocol), ensure_ascii=False) == (
        '{"protocol_id": "<texto>", "patient_age_range": "<texto>", "contraindications": [], '
        '"confidence_level": "high | medium | low", "requires_clinician_review": false}'
    )
