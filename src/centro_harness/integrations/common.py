"""Piezas comunes: cómo se le cuenta al framework lo que decidió la puerta."""

from __future__ import annotations

from centro_harness.hitl import GateOutcome, ToolRegistry


def describe_outcome(outcome: GateOutcome, registry: ToolRegistry) -> str:
    """Texto que ve el modelo tras llamar a una herramienta por la puerta."""
    rec = outcome.record
    if outcome.status == "pending_approval":
        return (
            "PENDIENTE DE APROBACIÓN HUMANA: la acción NO se ejecutó. "
            f"request_id={outcome.request_id}. No la repitas; informá que espera revisión."
        )
    assert rec is not None
    if outcome.status == "completed":
        return rec.result or ""
    if outcome.status == "failed":
        return f"ERROR de la herramienta: {rec.error}"
    if rec.decision == "denied_unknown_tool":
        names = ", ".join(spec.name for spec in registry.specs())
        return f"BLOQUEADO: la herramienta no existe. Disponibles: {names}"
    return f"BLOQUEADO ({rec.decision}): la acción no se ejecutó"
