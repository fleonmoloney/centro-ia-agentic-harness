"""Agente con un modelo REAL y aprobación humana de verdad (input por consola).

Uso (servidor OpenAI-compatible: OpenRouter, Ollama, vLLM, llama.cpp...):

    export HARNESS_API_KEY=...                       # no hace falta en Ollama / local
    python examples/real_model_demo.py --base-url https://openrouter.ai/api/v1 \\
        --model <id>:free --max-tokens 4096
    python examples/real_model_demo.py --framework crewai --base-url ... --model ...

Sin servidor (prueba del cableado, NO es un modelo):
    python -m centro_harness.mockmodel --port 8000     # en otra terminal
    python examples/real_model_demo.py --base-url http://localhost:8000/v1 --model mock

Con GPU en Colab/Kaggle: `--llm transformers --model Qwen/Qwen2.5-3B-Instruct --quantization 4bit`.
En Colab/Kaggle, `!python ...` no tiene entrada de teclado: usá `--approve yes` / `--approve no`,
o `%run examples/real_model_demo.py ...` para que `input()` funcione en el notebook.

Qué hace: el modelo propone acciones, la puerta HITL decide. Si pide `enviar_notificacion`
(riesgo alto), el script se detiene, te muestra qué iba a ejecutar (con la cédula enmascarada) y
espera tu decisión. Usá solo datos sintéticos si el proveedor es externo (ver docs/providers.md).

`--framework`:
  harness  (por defecto) el lazo de agente del harness: pausa, reanudación, `Budget`, trazas.
  crewai   un agente de CrewAI cuyo modelo es el LLM del harness (el `Budget` también lo corta)
           y cuyas herramientas pasan por la puerta. CrewAI no se detiene a esperar: termina
           diciendo "pendiente" y la aprobación se resuelve después, aquí.
  Para AG2 ver examples/ag2_real_model.py (necesita otro entorno: ver ese archivo).
"""

from __future__ import annotations

import argparse
import json
import sys

from centro_harness import (
    AgentRunner,
    Budget,
    BudgetExceeded,
    HITLGate,
    LLMError,
    Risk,
    ToolRegistry,
    ToolSpec,
    TracedLLM,
    Tracer,
    build_llm,
)

TAREA = "Enviá una notificación al ciudadano 1.234.567-2 avisando que su trámite 4521 fue aprobado"
REVISORA = "supervisora"


def make_gate() -> tuple[HITLGate, list[dict]]:
    """Dos herramientas: una de lectura (riesgo bajo) y un envío (riesgo alto)."""
    enviados: list[dict] = []
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="consultar_estado_tramite",
            risk=Risk.LOW,
            description="Consulta el estado de un trámite. Argumentos: numero (entero)",
        ),
        lambda **kw: f"Trámite {kw.get('numero', '?')}: aprobado",
    )
    registry.register(
        ToolSpec(
            name="enviar_notificacion",
            risk=Risk.HIGH,
            description="Envía un aviso a un ciudadano. Argumentos: destino (cédula), texto",
        ),
        lambda **kw: enviados.append(kw) or "aviso enviado",
    )
    return HITLGate(registry, authorized_reviewers={REVISORA}), enviados


def build_traced_llm(args: argparse.Namespace) -> tuple[TracedLLM, Tracer]:
    options = {"max_tokens": args.max_tokens, "prompt_style": args.prompt_style}
    if args.llm == "compat":
        options.update(base_url=args.base_url, model=args.model, min_interval=args.min_interval)
        if args.extra_body:
            options["extra_body"] = args.extra_body
        if args.system_prompt:
            options["system_prompt"] = args.system_prompt
    else:
        options.update(model=args.model)
        if args.quantization:
            options["quantization"] = args.quantization
    tracer = Tracer()
    budget = Budget(max_calls=args.max_calls, max_seconds=args.max_seconds)
    return TracedLLM(build_llm(args.llm, **options), tracer, budget), tracer


def decide(args: argparse.Namespace, pending: dict) -> bool:
    """Muestra lo que se iba a ejecutar y devuelve la decisión de la persona."""
    print("\n>>> PIDE APROBACIÓN")
    print(f"    herramienta: {pending['tool']}   riesgo: {pending['risk']}")
    print(f"    argumentos (enmascarados): {pending['arguments']}")
    if args.approve != "ask":
        print(f"    decisión automática (--approve {args.approve})")
        return args.approve == "yes"
    try:
        return input("    ¿Aprobás? [s/N]: ").strip().lower() in ("s", "si", "sí", "y", "yes")
    except EOFError:
        print("    (sin entrada: se rechaza)")
        return False


def run_harness(args, llm, gate, enviados):
    runner = AgentRunner(llm, gate, max_steps=args.max_steps)
    run = runner.start(args.task)
    while run.status == "awaiting_approval":
        run = runner.resume(run, approved=decide(args, run.pending), reviewer=REVISORA)
    print(f"\nEstado final: {run.status}" + (f"  ({run.detail})" if run.detail else ""))
    print(f"Respuesta del modelo: {run.answer}")
    print(f"Herramientas ejecutadas: {run.executed_tools}")
    return run.status == "done"


def run_crewai(args, llm, gate, enviados):
    try:
        from crewai import Agent, Crew, Task

        from centro_harness.integrations.crewai import HarnessCrewLLM, gated_crew_tools
    except ImportError:
        sys.exit("Falta CrewAI: pip install -e '.[crewai]'")
    thread = "crew-1"
    agent = Agent(
        role="Gestor de trámites",
        goal="Resolver el pedido usando solo las herramientas disponibles",
        backstory="Funcionario de un organismo público",
        llm=HarnessCrewLLM(llm),
        tools=gated_crew_tools(gate, thread_id=thread),
        allow_delegation=False,
        verbose=False,
        max_iter=args.max_steps,
    )
    task = Task(description=args.task, expected_output="Una respuesta breve", agent=agent)
    result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
    print(f"\nRespuesta del equipo de CrewAI: {result}")
    pending = gate.pending(thread)
    if pending:
        ok = decide(args, pending)
        gate.resume(
            thread_id=thread, request_id=pending["request_id"], approved=ok, reviewer=REVISORA
        )
        print(
            "(El equipo ya había terminado: la acción se resolvió después, "
            "sin que el modelo la vea.)"
        )
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--framework", choices=["harness", "crewai"], default="harness")
    p.add_argument("--llm", choices=["compat", "transformers"], default="compat")
    p.add_argument("--base-url", help="servidor OpenAI-compatible (o HARNESS_BASE_URL)")
    p.add_argument("--model", help="id del modelo (o HARNESS_MODEL)")
    p.add_argument("--quantization", choices=["4bit", "8bit", "fp16"], help="solo transformers")
    p.add_argument("--max-tokens", type=int, default=2048, help="más con modelos de razonamiento")
    p.add_argument("--prompt-style", choices=["schema", "example"], default="example")
    p.add_argument("--min-interval", type=float, default=0.0, help="segundos entre llamadas")
    p.add_argument(
        "--extra-body", type=json.loads,
        help='JSON extra por petición, p. ej. \'{"reasoning_effort": "low"}\'',
    )
    p.add_argument("--system-prompt", help="mensaje de sistema fijo (solo compat)")
    p.add_argument("--max-calls", type=int, default=20)
    p.add_argument("--max-seconds", type=float, default=300)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--approve", choices=["ask", "yes", "no"], default="ask")
    p.add_argument("--task", default=TAREA)
    args = p.parse_args(argv)

    gate, enviados = make_gate()
    llm, tracer = build_traced_llm(args)
    print(f"Tarea: {args.task}\nFramework: {args.framework}\n")
    try:
        ok = {"harness": run_harness, "crewai": run_crewai}[args.framework](
            args, llm, gate, enviados
        )
    except BudgetExceeded as exc:
        print(f"\nCortado por el presupuesto del harness: {exc}")
        ok = False
    except LLMError as exc:
        print(f"\nFalló la infraestructura del modelo (no es un error del modelo): {exc}")
        return 2
    print(f"Avisos realmente enviados: {enviados}")
    print(f"Trazas: {tracer.summary()}")
    print("Auditoría:", [(r.tool, r.decision) for r in gate.audit])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
