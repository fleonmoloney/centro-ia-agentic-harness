# centro-ia-agentic-harness

Harness agéntico de referencia en Python: la capa de control que rodea a un LLM para que sea
**acotado en costo, supervisable por personas, integrable con sistemas institucionales y
evaluable con modelos reales de forma reproducible**.

Es un prototipo de arquitectura, no un producto. Lo que hace, lo que está verificado y lo que no.

## Qué incluye

| Pieza | Qué hace | Código |
|---|---|---|
| **Lazo de agente** | El modelo propone una acción estructurada, la puerta HITL decide, el resultado vuelve al modelo. Cortes explícitos: `max_steps`, acción repetida, presupuesto agotado, modelo sin acción válida. La corrida se serializa para pausarla y retomarla. | `agent.py` |
| **Supervisión humana (HITL)** | *Default-deny*: herramienta desconocida se bloquea; riesgo bajo se ejecuta y se audita; riesgo alto **pausa** hasta que una persona decide. La aprobación va ligada a un `request_id` (no se puede aprobar una cosa y ejecutar otra), exige revisor y admite lista de revisores autorizados. Pausas persistentes (SQLite) y auditoría JSONL. | `hitl.py` |
| **Presupuesto** | Tope de llamadas, tokens y segundos para toda una corrida, no solo de ciclos. | `budget.py` |
| **Reflexión acotada** | Grafo LangGraph generar → evaluar → refinar, con tope de iteraciones y cortes ordenados si el evaluador falla o se agota el presupuesto. | `reflection.py` |
| **Salidas estructuradas con reparación** | Esquemas Pydantic; si el modelo devuelve JSON roto o inválido se le muestra el error y se reintenta con límite. Validación de **cédula uruguaya** y reglas de escalamiento a humano. | `schemas.py`, `jsonutil.py` |
| **Datos personales** | Enmascarado recursivo de la cédula en argumentos, resultados y bitácora. | `hitl.py` |
| **Backends de modelo** | `OpenAICompatLLM` (vLLM, Ollama, llama.cpp o APIs; solo biblioteca estándar; tokens reales; modelos de razonamiento; límites de velocidad y cuota), `TransformersLLM` (Colab/Kaggle), `OpenAILLM`, `ScriptedLLM` (pruebas). | `llm.py` |
| **Trazas** | Un evento por llamada con latencia y tokens, marcados como `provider` o `estimated`. | `tracing.py` |
| **Evals** | Suite determinista (compuerta de CI) y suite con modelo real, con intervalos de Wilson, códigos de salida y comparador entre modelos. | `evals.py`, `compare.py` |
| **Modelo simulado** | Servidor de juguete para probar el camino de evaluación sin GPU. **No mide nada.** | `mockmodel.py` |
| **Economía de la inferencia** | Modelo analítico (no medición) de asignación de herramientas. | `cost_model.py` |

## Cómo correrlo

```bash
python -m venv .venv && source .venv/bin/activate     # en Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
python -m centro_harness.evals --cases evals/cases.jsonl
```

Con Docker: `docker build -t centro-harness . && docker run --rm centro-harness`.

### Medir un modelo real

```bash
# 1) Probar el camino completo sin modelo (ventana 1):
python -m centro_harness.mockmodel --port 8000
# 2) Correr la suite live (ventana 2). Con el modelo simulado solo prueba el harness:
python -m centro_harness.evals --cases evals/live_cases.jsonl --llm compat \
    --base-url http://localhost:8000/v1 --model mock --out-json reports/mock.json
```

- **Local con Ollama / vLLM / llama.cpp:** mismo comando, con su `--base-url` y `--model`.
- **Colab o Kaggle con GPU:** `notebooks/live_eval_colab_kaggle.ipynb` (carga modelos con
  `transformers`, fp16 o 4 bits, y compara).
- **Modelos gratuitos por API (Nemotron, Qwen, GLM, DeepSeek, Kimi...):**
  [`docs/providers.md`](docs/providers.md) tiene cuotas, costo de una pasada, flags para no
  agotar la cuota y advertencias de privacidad.
- Comparar modelos: `python -m centro_harness.compare reports/*.json`.

### Ejemplos

HITL (la aprobación exige el `request_id` de lo que la persona revisó):

```python
from centro_harness import HITLGate, Risk, ToolRegistry, ToolSpec

registry = ToolRegistry()
registry.register(ToolSpec(name="borrar_registro", risk=Risk.HIGH), lambda **kw: "borrado")
gate = HITLGate(registry, authorized_reviewers={"supervisora"})

pedido = gate.submit("borrar_registro", {"id": 7}, thread_id="caso-1")
pedido.status        # "pending_approval"; pedido.pending es lo que la persona debe revisar
gate.resume(thread_id="caso-1", request_id=pedido.request_id, approved=True, reviewer="supervisora")
```

Agente con pausa y reanudación:

```python
from centro_harness import AgentRun, AgentRunner, TracedLLM, Tracer, Budget

llm = TracedLLM(modelo, Tracer(), Budget(max_calls=20, max_seconds=120))
runner = AgentRunner(llm, gate, max_steps=6)
run = runner.start("Notificá al ciudadano que su trámite fue aprobado")
if run.status == "awaiting_approval":
    guardado = run.to_json()                       # se puede retomar tras un reinicio
    run = runner.resume(AgentRun.from_json(guardado), approved=True, reviewer="supervisora")
```

## Estado de verificación (22/11/2026)

**Verificado en este entorno** (Linux, Python 3.13):

- 131 pruebas y `ruff` en verde; 29 de 29 casos deterministas.
- **Prueba de mutación:** se rompió a propósito la puerta HITL (aprobar todo), el tope de pasos y
  el enmascarado recursivo; la suite detecta las tres (la puerta rota sale con código 2).
- Cliente HTTP, modelos de razonamiento, `Retry-After`, espaciado de peticiones, cuota agotada y
  reporte parcial: **contra servidores HTTP simulados** en un hilo (no mocks de `urllib`).
- Persistencia: una solicitud pausada sobrevive a un reinicio con SQLite, y el agente retoma
  desde la corrida serializada.
- Las fallas que motivaron la v0.3 se reprodujeron antes de corregirlas (ADR-007).

**No verificado** (no se pudo o no se hizo):

- **Ningún proveedor ni modelo real.** No hay números de calidad de ningún modelo en este repo.
  Los reportes con `harness-mockmodel` solo prueban el harness.
- `TransformersLLM`, `OpenAILLM`, Ollama, vLLM y el notebook: no se ejecutaron (sin GPU ni acceso
  a Hugging Face en el entorno de desarrollo).
- Python 3.10, 3.11 y 3.12, Windows, la imagen Docker y el CI de GitHub: solo se probó 3.13 en
  Linux. El CI de este repo es el que debe confirmar el resto.
- Formato real de los encabezados de límite de cada proveedor (el cliente los interpreta con
  mejor esfuerzo).

## Límites conocidos

- **La suite determinista no mide la calidad de ningún modelo**: usa un guion. La suite live sí,
  pero 19 casos con pocas repeticiones dan intervalos anchos; no son un *benchmark*.
- **Las observaciones de las herramientas son texto no confiable que vuelve al prompt.** El
  prompt le pide al modelo tratarlas como datos; eso es una mitigación. La garantía es la puerta:
  una acción de riesgo alto no se ejecuta sin aprobación, sea cual sea el modelo. No hay
  detección de inyección más allá de eso (ver `centro-ia-secure-mcp-gpu` para un enfoque
  de política y escáneres).
- **Sin autenticación de revisores.** El nombre llega de afuera; el harness exige que exista y,
  si se configuró, que esté en la lista. No verifica que sea verdadero: eso es del canal de
  aprobación (API/UI), que acá no existe.
- **Datos personales:** el SQLite de pausas contiene los argumentos **en claro** (hacen falta
  para ejecutar al aprobar) y el modelo recibe los argumentos que él mismo produjo. Un uso real
  exige revisar el tratamiento según la Ley 18.331 (esto no es asesoramiento legal). Los
  resultados de las herramientas se enmascaran antes de volver al modelo (`scrub_results`).
- **El enmascarado de cédulas es heurístico:** solo enmascara números que validan el dígito
  verificador y puede ocultar de más algún número de 7–8 dígitos.
- **El presupuesto de tokens puede excederse por una llamada** (se verifica antes de cada una).
- **Sin *streaming* ni llamadas nativas a herramientas:** el agente usa JSON en texto, validado.
- **Los tokens son reales solo si el backend los informa**; si no, son estimaciones.
- **`cost_model.py` es un modelo, no una medición.**
- Solo se usan `langgraph`, `langchain-core` y `pydantic`; CrewAI, BeeAI y AG2 no son
  dependencias (la conversación multiagente no está implementada).

## Estructura

```
src/centro_harness/   agent · hitl · budget · reflection · schemas · jsonutil · llm · tracing
                      evals · compare · mockmodel · cost_model
evals/cases.jsonl     29 casos deterministas (CI)
evals/live_cases.jsonl  19 casos con modelo real
tests/                131 pruebas
docs/                 architecture.md (ADR 001-011) · evals.md · providers.md
notebooks/            live_eval_colab_kaggle.ipynb
.github/workflows/    CI: ruff + pytest + evals + imagen Docker
```

## Origen

Los patrones de partida (LangGraph, BeeAI, CrewAI) se estudiaron en laboratorios de formación
de IBM Skills Network. La primera versión de este repo eran esqueletos simulados; esta versión
los reimplementa sobre LangGraph con pruebas, trazas y evals, y suma el lazo de agente, la
evaluación con modelos reales y las correcciones de integridad de la puerta HITL.

Fernando León Moloney - Licencia MIT.
