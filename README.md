# centro-ia-agentic-harness

Harness agéntico de referencia en Python: la capa de control que rodea a un LLM para que sea
**acotado en costo, supervisable por personas, integrable con sistemas institucionales y
evaluable con modelos reales de forma reproducible**.

Es un prototipo de arquitectura, no un producto. Lo que hace, lo que está verificado y lo que no
está dicho abajo, sin adornos.

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
| **Integraciones** | BeeAI, AG2 y CrewAI (extras opcionales): sus herramientas pasan por la misma puerta HITL. Ver ADR-012 para el alcance exacto. | `integrations/` |
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

- 150 pruebas y `ruff` en verde (con los extras de frameworks y de SQLite instalados; sin ellos se omiten las que los necesitan); 32 de 32 casos deterministas.
- **Prueba de mutación:** se rompió a propósito la puerta HITL (aprobar todo), el tope de pasos y
  el enmascarado recursivo; la suite detecta las tres (la puerta rota sale con código 2).
- Cliente HTTP, modelos de razonamiento, `Retry-After`, espaciado de peticiones, cuota agotada y
  reporte parcial: **contra servidores HTTP simulados** en un hilo (no mocks de `urllib`).
- Integraciones: con un agente real de AG2, un equipo real de CrewAI y las herramientas de BeeAI, una acción de riesgo alto queda pendiente y nada se ejecuta sin aprobación; si el adaptador salta la puerta, las pruebas fallan. Se instalaron juntos sin conflictos (`pip check`) en Python 3.13.
- Persistencia: una solicitud pausada sobrevive a un reinicio con SQLite, y el agente retoma
  desde la corrida serializada.
- **Reportado por el autor, v0.3.0** (con capturas de pantalla del 22/11/2026): en **Windows** (cmd) la suite live contra `harness-mockmodel` dio 19/19 y 0 violaciones de invariante, así que el camino HTTP y el servidor simulado funcionan allí; en **GitHub Codespaces** `docker build` y `docker run` se completaron y mostraron el reporte de evals. Además, según el autor, las pruebas pasaron en su equipo y en Colab (sin captura). Falta anotar la versión de Python en cada entorno, y nada de esto incluye las integraciones de la v0.4. El 19/19 contra el modelo simulado prueba el harness, no un modelo.
- Las fallas que motivaron la v0.3 se reprodujeron antes de corregirlas (ADR-007).

**No verificado** (no se pudo o no se hizo):

- **Pocas mediciones reales:** dos modelos (arriba), una repetición cada uno, con el diagnóstico de
  las fallas estructurales de Qwen 3B (devuelve el esquema). Un solo proveedor por API (OpenRouter, capa gratuita), con un solo modelo. Los reportes con
  `harness-mockmodel` solo prueban el harness.
- Las integraciones se probaron **solo con modelos de guion** y con un agente por framework; no hay conversación multiagente ni delegación, ni ejecución con un modelo real. BeeAI se probó a nivel de herramienta (no un agente completo). El presupuesto del harness no alcanza a los agentes de AG2.
- `OpenAILLM`, Ollama y vLLM: no se ejecutaron. `TransformersLLM` y el notebook corrieron en
  Kaggle (reporte del autor), solo con Qwen2.5-3B en 4 bits.
- Python 3.10, 3.11 y 3.12, la suite completa de pruebas en Windows y el CI de GitHub: en el entorno de desarrollo solo se probó
  3.13 en Linux. El CI de este repo es el que debe confirmar el resto.
- Formato real de los encabezados de límite de cada proveedor (el cliente los interpreta con
  mejor esfuerzo).

## Protocolo de medición v2 (v0.5)

Tras diagnosticar las fallas de la primera medición se cambió el protocolo. **Los números de abajo
son del protocolo v1** y no son comparables con los del v2 (se conservan como línea base: `--prompt-style schema`
es el valor por defecto).

| Cambio | Motivo (medido) |
|---|---|
| `--prompt-style example`: el prompt muestra un objeto de relleno en vez del esquema JSON | Qwen2.5-3B devolvía el esquema en vez de una instancia en 9 de 9 casos estructurados |
| Anclaje de la cédula (`ci_grounding_check`, casos con `"grounded": true`): la cédula extraída debe figurar en la consulta | Nemotron inventó `1.000.000-8`, que pasa el dígito verificador |
| `live-categoria-otro` reformulado (una felicitación en vez de "¿qué horario atienden?") | El original era discutible y no debía contar contra el modelo |
| Respuesta vacía con `max_tokens` ≥ 4096 cuenta como **fallo** (con un tope menor sigue siendo error de configuración) | Excluirla favorecía al modelo (0,89 contra 0,84) |

Verificado: pruebas y casos deterministas para cada cambio, y prueba de mutación (si se desactiva
el anclaje, fallan la prueba y dos casos del CI). **No verificado con ningún modelo real:** que el
prompt con ejemplo mejore a Qwen 3B, ni cuántas veces Nemotron intenta inventar la cédula con el
anclaje activo (el reporte lo cuenta en `grounding_rejections`). Para medirlo hay que volver a
correr ambos modelos con el v2.

## Primeras mediciones reales (protocolo v1) (23/11/2026, aportadas por el autor)

| Modelo | Entorno | Éxito global (IC 95 %) | Agente | Estructurado | Reflexión | Llamadas/ejec. |
|---|---|---|---|---|---|---|
| `nvidia/nemotron-3.5-super-vl:free` | OpenRouter (capa gratuita) | 0,89 (0,67–0,97), n=18 | 6/6 | 8/10 | 2/2 | 1,5 |
| Qwen2.5-3B-Instruct, 4 bits | Kaggle | 0,26 (0,12–0,49), n=19 | 3/6 | 2/11 | 0/2 | 3,2 |

Latencia p50/p95: 8,9/33,5 s (Nemotron, API) y 37,9/58,0 s (Qwen, T4). Violaciones de la puerta
HITL: **0 en ambos** (el modelo puede equivocarse; la puerta no deja pasar nada indebido).

Cómo leerlo, sin pasarse:
- **No es una comparación directa.** Son modelos de tamaño muy distinto, servidos de forma distinta
  (API gratuita sin control de cuantización contra un modelo local en 4 bits), con una sola
  repetición por caso y decodificación voraz. Que los intervalos no se solapen dice que la
  diferencia es real en estos 19 casos, no que valga para otras tareas.
- **Nemotron: 1 ejecución sin respuesta** (`live-cedula-invalida-no-se-inventa`): el modelo de
  razonamiento agotó los 4096 tokens pensando y la respuesta quedó vacía. El harness la excluye de
  las tasas (n=18) por ser un problema de configuración; **si se la cuenta como fallo, el éxito es
  16/19 = 0,84**. Mirá las dos cifras: la decisión de excluirla favorece al modelo.
- **Nemotron: 2 fallas** (ver `reports/api_openrouter.json`):
  - `live-sin-cedula-no-se-inventa`: la consulta no traía cédula y el modelo **inventó una**
    (`1.000.000-8`) con dígito verificador válido, tras un reintento. Es el hallazgo más relevante
    para el caso de uso: validar el dígito verificador **no distingue una cédula real de una
    inventada**; el reintento con el mensaje de error incluso puede inducir a inventar una que
    pase. Pendiente: exigir que la cédula aparezca en el texto de la consulta.
  - `live-categoria-otro`: clasificó "¿en qué horario atienden?" como `trámite` en vez de `otro`.
    Es un caso **discutible** (dos personas razonables podrían discrepar), así que no debería contar
    contra el modelo; hay que reformularlo o quitarlo.
- **Qwen 3B (diagnóstico confirmado con `last_output`):** en las 9 fallas estructuradas el modelo
  **devolvió el propio esquema JSON** (`{"description": ..., "properties": ...}`) en vez de una
  instancia, y Pydantic rechazó por campos faltantes. No es "no entiende la consulta": es una
  debilidad conocida de los modelos chicos cuando el prompt incluye el esquema completo. Por lo
  tanto **el formato del prompt sí influye** en este resultado y la comparación con Nemotron
  mezcla capacidad del modelo con ese efecto. Los casos de agente (`stuck`) y reflexión
  (`clean_stop`) de Qwen siguen sin diagnosticar.
- Son 19 casos sintéticos, no un *benchmark*. Las capas gratuitas pueden entrenar con los prompts.

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
- El núcleo usa solo `langgraph`, `langchain-core` y `pydantic`. BeeAI, AG2 y CrewAI son extras
  opcionales (`pip install -e ".[frameworks]"`, Python 3.11-3.13). La conversación entre varios
  agentes no está implementada.

## Estructura

```
src/centro_harness/   agent · hitl · budget · reflection · schemas · jsonutil · llm · tracing
                      evals · compare · mockmodel · cost_model
                      integrations/ (beeai · ag2 · crewai)
evals/cases.jsonl     32 casos deterministas (CI)
evals/live_cases.jsonl  19 casos con modelo real
tests/                150 pruebas
docs/                 architecture.md (ADR 001-013) · evals.md · providers.md
notebooks/            live_eval_colab_kaggle.ipynb
.github/workflows/    CI: ruff + pytest + evals + imagen Docker + job frameworks
```

## Origen

Los patrones de partida (LangGraph, BeeAI, CrewAI) se estudiaron en laboratorios de formación
de IBM Skills Network. La primera versión de este repo eran esqueletos simulados; esta versión
los reimplementa sobre LangGraph con pruebas, trazas y evals, y suma el lazo de agente, la
evaluación con modelos reales y las correcciones de integridad de la puerta HITL.

Fernando León Moloney - Licencia MIT.
