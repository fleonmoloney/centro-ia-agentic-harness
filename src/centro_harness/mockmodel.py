"""Servidor OpenAI-compatible de juguete: un "modelo" hecho con reglas, SIN ningún modelo detrás.

Para qué sirve: probar de punta a punta el camino de evaluación con modelo real (cliente HTTP,
reparación de JSON, lazo de agente, reportes, códigos de salida) en una máquina sin GPU, antes de
gastar tiempo de Colab/Kaggle. Es una herramienta de verificación del harness.

Qué NO es: una medición. Sus respuestas salen de palabras clave, está afinado para los casos de
`evals/live_cases.jsonl` y los "tokens" que informa son estimaciones por caracteres. Un reporte
generado contra este servidor no dice NADA sobre ningún modelo de lenguaje.

Comportamientos (`--behavior`):
  normal    responde bien; cada 9.ª petición devuelve JSON roto para ejercitar la reparación.
  obedient  igual, pero obedece instrucciones inyectadas en las observaciones (pide borrar).
  broken    todas las peticiones fallan con HTTP 500 (para probar el manejo de infraestructura).

Uso: python -m centro_harness.mockmodel --port 8000
     python -m centro_harness.evals --cases evals/live_cases.jsonl --llm compat \\
         --base-url http://localhost:8000/v1 --model mock --repeats 3
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

_CI = re.compile(r"(?<!\d)(?:\d{1,2}(?:\.\d{3}){2}-\d|\d{7,8})(?!\d)")


def _action(**fields) -> str:
    return json.dumps(fields, ensure_ascii=False)


class MockModel:
    def __init__(self, behavior: str = "normal") -> None:
        if behavior not in ("normal", "obedient", "broken"):
            raise ValueError(f"behavior desconocido: {behavior}")
        self.behavior = behavior
        self.requests = 0

    # ------------------------------------------------------------------ reglas
    def reply(self, prompt: str) -> str:
        self.requests += 1
        if "Elegí la próxima acción." in prompt:
            return self._agent(prompt)
        if '"score"' in prompt and "Puntuá" in prompt:
            return json.dumps({"score": 0.9, "missing": "", "superfluous": ""})
        if '"citizen_id"' in prompt:
            return self._ticket(prompt)
        if '"protocol_id"' in prompt:
            found = re.search(r"P-\d+", prompt)
            return json.dumps(
                {
                    "protocol_id": found.group(0) if found else "P-000",
                    "patient_age_range": "18-65",
                    "contraindications": ["embarazo", "insuficiencia renal"],
                    "confidence_level": "low",
                },
                ensure_ascii=False,
            )
        return "Respuesta simulada."

    def _ticket(self, prompt: str) -> str:
        if self.requests % 9 == 0:
            return '{"citizen_id": 1.234, "issue_category": '  # JSON roto a propósito
        consulta = prompt.split("\n\nTu respuesta anterior")[0].split("\n\nRespondé SOLO")[0]
        match = _CI.search(consulta)
        if not match:
            return "No encuentro ninguna cédula en la consulta."
        text = consulta.lower()
        if any(w in text for w in ("dolor", "respira", "ambulancia", "inconsciente")):
            category = "salud"
        elif any(w in text for w in ("paro", "empleo", "unemployment", "job")):
            category = "empleo"
        elif any(w in text for w in ("renovar", "trámite", "tramite", "jubilación")):
            category = "trámite"
        else:
            category = "otro"
        if any(w in text for w in ("dejó de respirar", "inconsciente")):
            urgency = "crítica"
        elif "no es urgente" in text:
            urgency = "baja"
        elif any(w in text for w in ("dolor", "falta de aire", "urgente")):
            urgency = "alta"
        elif category == "otro":
            urgency = "baja"
        else:
            urgency = "media"
        return json.dumps(
            {
                "citizen_id": match.group(0),
                "issue_category": category,
                "urgency_level": urgency,
                "recommended_action": "Derivar a la oficina correspondiente",
            },
            ensure_ascii=False,
        )

    def _agent(self, prompt: str) -> str:
        task_match = re.search(r"^Tarea: (.*)$", prompt, re.MULTILINE)
        task = (task_match.group(1) if task_match else "").lower()
        if "Historial:" in prompt:
            last = re.findall(r"<observacion>(.*?)</observacion>", prompt, re.DOTALL)[-1]
            already_called = "borrar_registro(" in prompt
            if (
                self.behavior == "obedient"
                and "NOTA DEL SISTEMA" in last
                and not already_called
            ):
                return _action(tool="borrar_registro", arguments={"id": 4521})
            if last.startswith("BLOQUEADO"):
                return _action(final_answer=f"No pude completar la acción. {last}")
            return _action(final_answer=f"Resultado: {last}")
        number = re.search(r"\d{2,}", task)
        if "formate" in task:
            return _action(tool="formatear_disco", arguments={})
        if "borrá" in task:
            return _action(
                tool="borrar_registro", arguments={"id": int(number.group(0)) if number else 0}
            )
        if "notificación" in task or "notificá" in task:
            return _action(
                tool="enviar_notificacion",
                arguments={"destino": "1.234.567-2", "texto": "Su trámite fue aprobado."},
            )
        if "ley" in task or "normativa" in task:
            return _action(tool="buscar_normativa", arguments={"tema": "datos personales"})
        if "trámite" in task or "estado" in task:
            return _action(
                tool="consultar_estado_tramite",
                arguments={"numero": int(number.group(0)) if number else 0},
            )
        return _action(final_answer="No sé cómo ayudar con eso.")


class _Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802 - nombre fijado por http.server
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"data": [{"id": "mock"}]})
        else:
            self._send(404, {"error": "no encontrado"})

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        model: MockModel = self.server.mock  # type: ignore[attr-defined]
        if model.behavior == "broken":
            self._send(500, {"error": "servidor simulado roto"})
            return
        prompt = body["messages"][-1]["content"]
        text = model.reply(prompt)
        self._send(
            200,
            {
                "choices": [{"message": {"role": "assistant", "content": text}}],
                # Estimación por caracteres, no un conteo de tokenizador.
                "usage": {
                    "prompt_tokens": max(1, len(prompt) // 4),
                    "completion_tokens": max(1, len(text) // 4),
                },
            },
        )

    def log_message(self, *args):  # silencio
        pass


def make_server(port: int = 0, behavior: str = "normal", host: str = "127.0.0.1") -> HTTPServer:
    server = HTTPServer((host, port), _Handler)
    server.mock = MockModel(behavior)  # type: ignore[attr-defined]
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--behavior", choices=["normal", "obedient", "broken"], default="normal")
    args = parser.parse_args(argv)
    server = make_server(args.port, args.behavior)
    print(
        f"Modelo SIMULADO ({args.behavior}) en http://127.0.0.1:{server.server_port}/v1 "
        "- no mide ningún modelo real. Ctrl+C para salir."
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
