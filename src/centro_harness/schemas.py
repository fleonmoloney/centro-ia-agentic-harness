"""Salidas estructuradas: esquemas Pydantic + bucle de reparación ante datos inválidos.

Regla del harness: el output de un agente que va a un sistema institucional se valida contra un
`BaseModel`. Si no valida, se le devuelve el error al modelo y se reintenta (acotado).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from centro_harness.jsonutil import extract_json
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


class InstitutionalTicketNeutral(InstitutionalTicket):
    """Variante de medición: misma validación, descripción de `citizen_id` sin empujar a inventar.

    La descripción estándar dice "con verificador válido": a un modelo que no encuentra la cédula
    en la consulta le pide una cédula válida. Acá solo se pide la que figura en la consulta. La
    validación (formato y verificador) es idéntica; cambia únicamente lo que lee el modelo.
    """

    citizen_id: str = Field(description="Cédula tal como figura en la consulta; vacío si no hay")


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


class GroundingError(ValueError):
    """Un dato obligatorio no figura en el texto de origen (posible invención del modelo)."""


_CI_IN_TEXT = re.compile(r"\d{1,2}[.\s]?\d{3}[.\s]?\d{3}\s?-?\s?\d")


def ci_grounding_check(source_text: str) -> Callable[[BaseModel], str | None]:
    """Verificación de anclaje: la cédula extraída debe figurar en el texto de la consulta.

    Validar el dígito verificador no distingue una cédula real de una inventada que casualmente
    pasa el cálculo (medido: un modelo inventó `1.000.000-8` cuando la consulta no traía cédula).
    Devuelve una función `ticket -> mensaje de error | None`. Cuenta los rechazos en
    `check.rejections` para poder medir cuántas veces el modelo intentó inventar.
    """
    found = {re.sub(r"\D", "", m) for m in _CI_IN_TEXT.findall(source_text or "")}

    def check(ticket: BaseModel) -> str | None:
        digits = re.sub(r"\D", "", getattr(ticket, "citizen_id", "") or "")
        if digits in found:
            return None
        check.rejections += 1
        return (
            "citizen_id no figura en el texto de la consulta. Si la consulta no trae una "
            "cédula, no la inventes: dejá citizen_id vacío."
        )

    check.rejections = 0
    return check


def _describe_error(exc: ValueError) -> str:
    if isinstance(exc, GroundingError):
        return str(exc)
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
    return f"{type(exc).__name__}: {exc}"[:300]


def structured_with_retry(
    llm: LLM,
    prompt: str,
    schema: type[T],
    *,
    max_attempts: int = 3,
    label: str = "structured",
    check: Callable[[T], str | None] | None = None,
) -> T:
    """Pide una salida estructurada; si no valida, reintenta mostrando el error al modelo.

    Se repara todo `ValueError` (esquema inválido, JSON roto, texto sin objeto). Los fallos de
    infraestructura (red, presupuesto agotado) NO se reintentan aquí: se propagan.
    """
    current = prompt
    last_error = ""
    for _ in range(max_attempts):
        try:
            obj = llm.structured(current, schema, label=label)
            problem = check(obj) if check else None
            if problem:
                raise GroundingError(problem)
            return obj
        except ValueError as exc:  # ValidationError y JSONDecodeError son ValueError
            last_error = _describe_error(exc)
            current = (
                f"{prompt}\n\nTu respuesta anterior no validó contra el esquema: {last_error}\n"
                "Corregí solo eso y devolvé el objeto completo."
            )
    raise StructuredOutputError(max_attempts, last_error)


__all__ = [
    "InstitutionalTicket",
    "InstitutionalTicketNeutral",
    "MedicalProtocol",
    "GroundingError",
    "StructuredOutputError",
    "ci_grounding_check",
    "extract_json",
    "format_ci",
    "is_valid_ci",
    "mask_ci",
    "structured_with_retry",
]
