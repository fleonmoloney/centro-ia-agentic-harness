# centro-ia-agentic-harness

Harness agéntico de referencia en Python: la capa de control que rodea a un LLM para que sea
**acotado en costo, supervisable por personas, integrable con sistemas institucionales y
evaluable de forma reproducible**.

Es un prototipo de arquitectura, no un producto. Lo que hace y lo que no hace está dicho
abajo, sin adornos.

## Qué incluye (y está probado)

| Pieza | Qué hace | Código |
|---|---|---|
| **Reflexión con tope de costo** | Grafo cíclico LangGraph generar → evaluar → refinar. El evaluador devuelve una autocrítica estructurada; el ruteo es por umbral y nunca supera `max_iterations` generaciones. | `reflection.py` |
| **Supervisión humana real (HITL)** | Política *default-deny*: herramienta desconocida se bloquea; riesgo bajo se ejecuta y se audita; riesgo alto **pausa el grafo** (`interrupt`) hasta que una persona decide. Aprobar exige nombre de revisor. Bitácora exportable a JSONL y datos sensibles enmascarados. | `hitl.py` |
| **Salidas estructuradas con reparación** | Esquemas Pydantic para integrar con sistemas del Estado. Si el modelo devuelve algo inválido, se le muestra el error y se reintenta, con límite. Incluye validación de **cédula uruguaya** (dígito verificador) y reglas de escalamiento a humano. | `schemas.py` |
| **Trazabilidad** | Cada llamada al LLM (también las fallidas) deja un evento con latencia y tokens estimados; exportable a JSONL. | `tracing.py` |
| **Economía de la inferencia** | Modelo analítico que compara asignar herramientas al agente vs. a la tarea, con supuestos explícitos. | `cost_model.py` |
| **Evals** | Casos dorados en `evals/cases.jsonl`, runner con reporte Markdown/JSON y código de salida distinto de 0 si algo falla (sirve de compuerta en CI). | `evals.py` |
| **LLM intercambiable** | Protocolo mínimo; `ScriptedLLM` para pruebas deterministas y `OpenAILLM` para un modelo real. | `llm.py` |

## Cómo correrlo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # agregá ".[openai]" si vas a usar un modelo real
pytest && python -m centro_harness.evals --cases evals/cases.jsonl
```

Con Docker: `docker build -t centro-harness . && docker run --rm centro-harness`.

Ejemplo mínimo de HITL:

```python
from centro_harness import HITLGate, Risk, ToolRegistry, ToolSpec

registry = ToolRegistry()
registry.register(ToolSpec(name="borrar_registro", risk=Risk.HIGH), lambda **kw: "borrado")
gate = HITLGate(registry)

gate.submit("borrar_registro", {"id": 7}, thread_id="caso-1").status   # "pending_approval"
gate.resume(thread_id="caso-1", approved=True, reviewer="supervisora")  # ejecuta y audita
```

## Qué NO es (límites conocidos)

- **Los evals por defecto no miden la calidad de ningún modelo.** Corren con `ScriptedLLM`
  (un guion) y verifican la lógica del harness: reintentos, ruteo, tope de iteraciones, HITL,
  reglas de negocio. Para medir un modelo real hay un caso marcado `"live": true` y el modo
  `--llm openai`; ese camino **no se ejecuta en CI** (no hay clave) y hoy tiene un solo caso.
- **Los tokens son estimaciones** (~4 caracteres por token), útiles para comparar
  configuraciones entre sí, no para facturación.
- **`cost_model.py` es un modelo, no una medición.** Sus supuestos están en el código. El
  ahorro depende de cuántas herramientas haya: con 2 es modesto, con 8 es mayor.
- **El estado del HITL vive en memoria** (`MemorySaver`). Un reinicio pierde las llamadas
  pendientes. En producción hace falta un checkpointer persistente (por ejemplo Postgres) y
  un canal real de aprobación (API/UI) en lugar de llamar a `resume` desde código.
- **Sin autenticación de revisores:** el nombre del revisor es un dato que llega de afuera; el
  harness exige que exista, no que sea verdadero.
- **Un solo proveedor de LLM implementado** (OpenAI, vía LangChain). El adaptador no está
  probado contra la API real en este repo.
- **Datos personales:** se enmascara la cédula en el prompt de aprobación y en la bitácora,
  pero el estado en memoria y los argumentos de las herramientas la contienen en claro. Un uso
  real exigiría revisar el tratamiento según la Ley 18.331 de protección de datos personales
  (esto no es asesoramiento legal).

## Frameworks: qué se usa y qué no

Solo hay tres dependencias: `langgraph`, `langchain-core` y `pydantic`. CrewAI, BeeAI y
AG2/AutoGen **no son dependencias**: la versión anterior de este repo las listaba y sus
versiones fijadas eran incompatibles entre sí, así que el proyecto no se podía instalar.
Los patrones que interesaban se implementaron sobre LangGraph:

- *Permisos y supervisión humana* (el rol de `AskPermissionRequirement` en BeeAI) →
  `hitl.py`, con `interrupt` y checkpointer.
- *Herramientas a nivel de tarea* (el enfoque de CrewAI) → `cost_model.py`.
- *Conversación multiagente* (AG2/AutoGen) → no implementado.

## Estructura

```
src/centro_harness/   llm · reflection · hitl · schemas · tracing · cost_model · evals
evals/cases.jsonl     casos dorados
tests/                48 tests (pytest)
docs/architecture.md  decisiones de diseño (ADR)
.github/workflows/    CI: ruff + pytest + evals
```

## Origen

Los patrones de partida (LangGraph, BeeAI, CrewAI) se estudiaron en laboratorios de formación
de IBM Skills Network. La primera versión de este repo eran esqueletos simulados de esos
patrones; esta versión los reimplementa sobre LangGraph como un proyecto con pruebas, trazas y
evals.

Fernando León Moloney - Licencia MIT.
