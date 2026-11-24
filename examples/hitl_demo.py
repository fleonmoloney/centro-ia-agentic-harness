"""Demo de la puerta HITL: qué pasa con una herramienta de riesgo alto. No usa ningún modelo.

Uso:  python examples/hitl_demo.py

Muestra, en orden: una herramienta de riesgo bajo (se ejecuta sola), una de riesgo alto (queda
pausada), un revisor no autorizado (rechazado), una aprobación con otro `request_id` (rechazada) y
la aprobación correcta (se ejecuta). Al final imprime la auditoría, con la cédula enmascarada.
"""

from __future__ import annotations

from centro_harness import (
    HITLGate,
    RequestMismatchError,
    Risk,
    ToolRegistry,
    ToolSpec,
)


def main() -> list[str]:
    """Corre la demo e imprime cada paso. Devuelve las decisiones de la auditoría, en orden."""
    registro_borrado: list[int] = []

    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="consultar", risk=Risk.LOW, description="Consulta un trámite"),
        lambda **kw: f"Trámite {kw['numero']}: en revisión",
    )
    registry.register(
        ToolSpec(name="borrar_registro", risk=Risk.HIGH, description="Borra un registro"),
        lambda **kw: registro_borrado.append(kw["id"]) or "borrado",
    )
    gate = HITLGate(registry, authorized_reviewers={"supervisora"})

    print("1) Riesgo bajo: se ejecuta sin pedir nada")
    salida = gate.submit("consultar", {"numero": 4521}, thread_id="caso-0")
    print(f"   estado={salida.status}  resultado={salida.record.result!r}\n")

    print("2) Riesgo alto: queda pausada, NO se ejecuta")
    pedido = gate.submit(
        "borrar_registro", {"id": 7, "citizen_id": "1.234.567-2"}, thread_id="caso-1"
    )
    print(f"   estado={pedido.status}")
    print(f"   la persona debe revisar: {pedido.pending}")
    print(f"   registros borrados hasta ahora: {registro_borrado}\n")

    print("3) Un revisor que no está autorizado")
    rechazo = gate.resume(
        thread_id="caso-1", request_id=pedido.request_id, approved=True, reviewer="intruso"
    )
    print(f"   decisión={rechazo.record.decision}  borrados={registro_borrado}\n")

    # Tras un rechazo la solicitud queda cerrada: se vuelve a pedir para seguir con la demo.
    pedido = gate.submit("borrar_registro", {"id": 7}, thread_id="caso-2")

    print("4) Aprobar con un request_id que no es el que se revisó")
    try:
        gate.resume(
            thread_id="caso-2", request_id="otro-pedido", approved=True, reviewer="supervisora"
        )
    except RequestMismatchError as exc:
        print(f"   RequestMismatchError: {exc}\n")

    print("5) La persona autorizada aprueba el pedido correcto")
    ok = gate.resume(
        thread_id="caso-2", request_id=pedido.request_id, approved=True, reviewer="supervisora"
    )
    print(f"   decisión={ok.record.decision}  revisó={ok.record.reviewer}")
    print(f"   borrados={registro_borrado}\n")

    print("Auditoría (la cédula aparece enmascarada):")
    for rec in gate.audit:
        print(f"   {rec.tool:<16} {rec.decision:<22} reviewer={rec.reviewer}  args={rec.arguments}")
    return [rec.decision for rec in gate.audit]


if __name__ == "__main__":
    main()
