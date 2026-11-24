"""AG2 con un modelo REAL detrás de la puerta HITL. Pide un entorno propio.

Por qué un entorno aparte: AG2 1.x necesita `ag2[openai]` para hablar con un servidor
OpenAI-compatible, y eso trae `openai` 3.x, que choca con `instructor` y `litellm` (los usa
CrewAI, que pide `openai` < 3). Ver ADR-012. Un entorno para AG2 y otro para CrewAI:

    python -m venv .venv-ag2 && source .venv-ag2/bin/activate
    pip install -e ".[ag2-real]"
    export HARNESS_API_KEY=...          # no hace falta en Ollama / local
    python examples/ag2_real_model.py --base-url https://openrouter.ai/api/v1 --model <id>:free

AG2 usa herramientas nativas (function calling): el modelo tiene que soportarlas. Si no, no
llama a ninguna herramienta y simplemente responde texto.

Límite: el modelo es el cliente de AG2, no el `TracedLLM` del harness. El presupuesto del
harness NO corta a este agente (`--max-tokens` y el timeout del cliente son lo único que lo
acota) y las trazas del harness no lo ven. La puerta HITL sí vale para todas sus herramientas.
Como en CrewAI, AG2 no espera: termina y la aprobación se resuelve después, aquí.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from real_model_demo import REVISORA, TAREA, decide, make_gate


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default=os.environ.get("HARNESS_BASE_URL"))
    p.add_argument("--model", default=os.environ.get("HARNESS_MODEL"))
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--approve", choices=["ask", "yes", "no"], default="ask")
    p.add_argument("--task", default=TAREA)
    args = p.parse_args(argv)
    if not args.base_url or not args.model:
        sys.exit("Faltan --base-url y --model (o HARNESS_BASE_URL y HARNESS_MODEL)")

    gate, enviados = make_gate()
    try:
        from ag2.config import OpenAICompletionsConfig

        from centro_harness.integrations.ag2 import build_agent

        # AG2 falla recién al instanciar la configuración si falta `ag2[openai]`.
        config = OpenAICompletionsConfig(
            model=args.model,
            base_url=args.base_url,
            api_key=os.environ.get("HARNESS_API_KEY") or "sin-clave",
            max_tokens=args.max_tokens,
        )
    except ImportError as exc:
        sys.exit(f"{exc}\nInstalá: pip install -e '.[ag2-real]' (en un entorno sin CrewAI)")

    thread = "ag2-1"
    agent = build_agent(
        gate,
        name="tramites",
        prompt="Sos un agente de un organismo público. Resolvé el pedido con las herramientas.",
        config=config,
        thread_id=thread,
    )
    print(f"Tarea: {args.task}\nFramework: ag2\n")
    reply = asyncio.run(agent.ask(args.task))
    print(f"Respuesta del agente de AG2: {reply.body}")
    pending = gate.pending(thread)
    if pending:
        ok = decide(args, pending)
        gate.resume(
            thread_id=thread, request_id=pending["request_id"], approved=ok, reviewer=REVISORA
        )
    else:
        print("(El modelo no pidió ninguna acción de riesgo alto.)")
    print(f"Avisos realmente enviados: {enviados}")
    print("Auditoría:", [(r.tool, r.decision) for r in gate.audit])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
