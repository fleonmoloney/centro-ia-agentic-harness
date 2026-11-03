"""Salidas estructuradas: esquemas Pydantic + bucle de reparación ante datos inválidos.

Regla del harness: el output de un agente que va a un sistema institucional se valida contra un
`BaseModel`. Si no valida, se le devuelve el error al modelo y se reintenta (acotado).
"""

from __future__ import annotations

import re
from typing import Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from centro_harness.llm import LLM

T = TypeVar("T", bound=BaseModel)

_CI_WEIGHTS = (2, 9, 8, 7, 6, 3, 4)


# --------------------------------------------------------------------------- cédula uruguaya
def ci_check_digit(body: str) -> int:
    """Dígito verificador de la cédula uruguaya para el cuerpo numérico (hasta 7 dígitos)."""
    padded = body.zfill(7)
    total = sum(int(d) * w for d, w in zip(padded, _CI_WEIGHTS, strict=True))
    return (10 - total % 10) % 10


def is_valid_ci(raw: str) -> bool:
    """True si `raw` tiene 7 u 8 dígitos y el verificador coincide. Ignora puntos y guiones."""
    digits = re.sub(r"\D", "", raw or "")
    if not 7 <= len(digits) <= 8:
        return False
    return ci_check_digit(digits[:-1]) == int(digits[-1])


def format_ci(raw: str) -> str:
    """Formato canónico x.xxx.xxx-d (asume una cédula válida)."""
    digits = re.sub(r"\D", "", raw)
    body, check = digits[:-1], digits[-1]
    return f"{int(body):,}".replace(",", ".") + f"-{check}"


def mask_ci(raw: str) -> str:
    """Enmascara una cédula para logs: deja los últimos 3 dígitos del cuerpo y el verificador."""
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) < 4:
        return "***"
    return f"*.***.{digits[-4:-1]}-{digits[-1]}"


# --------------------------------------------------------------------------- esquemas
class InstitutionalTicket(BaseModel):
    """Derivación de una consulta ciudadana en un organismo público."""

    citizen_id: str = Field(description="Cédula de identidad uruguaya, con verificador válido")
    issue_category: Literal["salud", "empleo", "trámite", "otro"]
    urgency_level: Literal["baja", "media", "alta", "crítica"]
    recommended_action: str = Field(min_length=3)
    requires_human_supervisor: bool = False

    @field_validator("citizen_id")
    @classmethod
    def _valid_ci(cls, value: str) -> str:
        if not is_valid_ci(value):
            raise ValueError("cédula inválida: formato o dígito verificador incorrecto")
        return format_ci(value)

    @model_validator(mode="after")
    def _escalate_when_risky(self) -> InstitutionalTicket:
        # Regla de negocio no negociable: lo crítico, o lo urgente en salud, lo ve una persona.
        risky = self.urgency_level == "crítica" or (
            self.issue_category == "salud" and self.urgency_level == "alta"
        )
        if risky:
            self.requires_human_supervisor = True
        return self


class MedicalProtocol(BaseModel):
    """Protocolo clínico extraído por un agente; nunca se aplica sin revisión si hay duda."""

    protocol_id: str
    patient_age_range: str
    contraindications: list[str]
    confidence_level: Literal["high", "medium", "low"]
    requires_clinician_review: bool = False

    @model_validator(mode="after")
    def _review_unless_high(self) -> MedicalProtocol:
        if self.confidence_level != "high":
            self.requires_clinician_review = True
        return self


# --------------------------------------------------------------------------- reparación
class StructuredOutputError(RuntimeError):
    """El modelo no produjo una salida válida dentro del límite de intentos."""

    def __init__(self, attempts: int, last_error: str) -> None:
        super().__init__(f"Sin salida válida tras {attempts} intentos. Último error: {last_error}")
        self.attempts = attempts
        self.last_error = last_error


def structured_with_retry(
    llm: LLM,
    prompt: str,
    schema: type[T],
    *,
    max_attempts: int = 3,
    label: str = "structured",
) -> T:
    """Pide una salida estructurada; si no valida, reintenta mostrando el error al modelo."""
    current = prompt
    last_error = ""
    for _ in range(max_attempts):
        try:
            return llm.structured(current, schema, label=label)
        except ValidationError as exc:
            last_error = "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
            )
            current = (
                f"{prompt}\n\nTu respuesta anterior no validó contra el esquema: {last_error}\n"
                "Corregí solo eso y devolvé el objeto completo."
            )
    raise StructuredOutputError(max_attempts, last_error)
