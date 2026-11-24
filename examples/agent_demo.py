"""Demo del agente con pausa y reanudación. No usa ningún modelo: las respuestas son un guion.

Uso:  python examples/agent_demo.py            # la persona aprueba
      python examples/agent_demo.py --reject   # la persona rechaza

El guion (`ScriptedLLM`) reemplaza al modelo: prueba la lógica de control del harness, no mide la
calidad de ningún modelo. Para usar un modelo real, cambiá `ScriptedLLM(...)` por
`build_llm(...)` (ver docs/providers.md); el resto del código no cambia.

Recorrido: el agente consulta el trámite (riesgo bajo, se ejecuta), pide notificar al ciudadano
(riesgo alto, se pausa), la corrida se guarda como JSON, se restaura como si hubiera habido un
reinicio, y recién ahí la persona decide.
"""

from __future__ import annotations

import argparse

from centro_harness import (
    AgentRun,
    AgentRunner,
    Budget,
    HITLGate,
    Risk,
    ScriptedLLM,
    ToolRegistry,
    ToolSpec,
    TracedLLM,
    Tracer,
)

TAREA = "Notificá al ciudadano que su trámite 4521 fue aprobado"


def main(argv: list[str] | None = None) -> AgentRun:
    """Corre la demo e imprime cada paso. Devuelve la corrida final."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reject", action="store_true", help="la persona rechaza la acción")
    args = parser.parse_args(argv)

    enviados: list[str] = []
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="consultar", risk=Risk.LOW, description="Consulta el estado de un trámite"),
        lambda **kw: f"Trámite {kw['numero']}: aprobado",
    )
    registry.register(
        ToolSpec(name="notificar", risk=Risk.HIGH, description="Envía un aviso al ciudadano"),
        lambda **kw: enviados.append(kw["mensaje"]) or "aviso enviado",
    )
    gate = HITLGate(registry, authorized_reviewers={"supervisora"})

    modelo = ScriptedLLM(
        structured=[
            {"reasoning": "primero verifico", "tool": "consultar", "arguments": {"numero": 4521}},
            {
                "reasoning": "está aprobado, aviso",
                "tool": "notificar",
                "arguments": {"mensaje": "Su trámite 4521 fue aprobado"},
            },
            # El guion es fijo: lo que dice el "modelo" al final lo elegimos nosotros. Un modelo
            # real lee la observación "BLOQUEADO" y redacta su propia respuesta.
            {
                "final_answer": (
                    "La supervisora rechazó el aviso: el ciudadano NO fue notificado."
                    if args.reject
                    else "Trámite verificado y ciudadano notificado."
                )
            },
        ]
    )
    tracer = Tracer()
    llm = TracedLLM(modelo, tracer, Budget(max_calls=20, max_seconds=120))
    runner = AgentRunner(llm, gate, max_steps=6)

    print(f"Tarea: {TAREA}\n")
    run = runner.start(TAREA)
    print(f"1) Estado tras start(): {run.status}")
    print(f"   Ya ejecutado: {run.executed_tools}")
    print(f"   Esperando decisión sobre: {run.pending['tool']} {run.pending['arguments']}")

    guardado = run.to_json()
    print(f"\n2) Corrida guardada como JSON ({len(guardado)} caracteres); enviados: {enviados}")

    decision = "rechaza" if args.reject else "aprueba"
    print(f"\n3) Se restaura el JSON y la persona {decision}")
    run = runner.resume(
        AgentRun.from_json(guardado), approved=not args.reject, reviewer="supervisora"
    )
    print(f"   Estado final: {run.status}")
    print(f"   Respuesta: {run.answer}")
    print(f"   Herramientas ejecutadas: {run.executed_tools}")
    print(f"   Avisos realmente enviados: {enviados}")
    print(f"   Llamadas al modelo: {tracer.calls}")
    return run


if __name__ == "__main__":
    main()
