# Decisiones de arquitectura (ADR)

## ADR-001: LangGraph en lugar de cadenas lineales
**Contexto:** el harness necesita ciclos (reflexión), ramas condicionales y pausas para
supervisión humana.
**Decisión:** usar LangGraph (`StateGraph`) para reflexión y HITL.
**Consecuencias:** el tope de iteraciones es una arista condicional explícita y testeable
(`route_by_score`). La pausa humana usa checkpointer + `interrupt`, no un `input()` ni un bucle.
**Costo:** acopla el harness a LangGraph; el estado de las pausas depende del checkpointer.

## ADR-002: Salidas estructuradas con Pydantic y reparación acotada
**Contexto:** los resultados de un agente que alimentan sistemas institucionales no pueden ser
texto libre.
**Decisión:** todo output que sale hacia un sistema se valida contra un `BaseModel`. Si no
valida, `structured_with_retry` devuelve el error al modelo y reintenta hasta `max_attempts`;
después falla con `StructuredOutputError` (no inventa un valor por defecto).
**Consecuencias:** las reglas de negocio viven en el esquema (cédula con dígito verificador,
escalamiento obligatorio a humano en casos críticos), no en el prompt, y se prueban sin LLM.
**Sin cifras de impacto:** la versión anterior afirmaba que esto "elimina el 90% de los
errores de integración". No hay medición que lo respalde, así que se retiró.

## ADR-003: HITL *default-deny* con revisor identificado
**Decisión:** herramienta no registrada → bloqueada; riesgo bajo → se ejecuta y se audita;
riesgo alto → el grafo se pausa hasta una decisión humana. Una aprobación sin nombre de revisor
se trata como rechazo (`rejected_no_reviewer`).
**Por qué:** la trazabilidad de quién autorizó qué es el valor del control. Una aprobación
anónima no se puede auditar.
**Límite:** el harness no autentica al revisor; eso es responsabilidad del canal de aprobación.

## ADR-004: LLM detrás de un protocolo, con un modelo de guion para pruebas
**Decisión:** el harness depende de `LLM` (`generate`, `structured`). `ScriptedLLM` permite
tests y evals deterministas, gratuitos y reproducibles en CI.
**Consecuencia:** los evals por defecto prueban la lógica de control, **no** la calidad del
modelo. Medir un modelo real requiere los casos `live` y una clave de API.

## ADR-005: Dependencias mínimas
**Contexto:** la versión anterior fijaba LangGraph, CrewAI, AG2 y BeeAI con versiones
incompatibles entre sí; el proyecto no se podía instalar.
**Decisión:** tres dependencias (`langgraph`, `langchain-core`, `pydantic`) con rangos, y el
proveedor de LLM como extra opcional. Los patrones de otros frameworks se implementan sobre
LangGraph en lugar de importar cada framework.
**Costo:** no hay demostración de uso de esos frameworks; solo de los patrones.

## ADR-006: El modelo de costos es analítico y declara sus supuestos
**Decisión:** `cost_model.py` calcula tokens con una heurística y supuestos explícitos
(prompt de sistema, tokens de selección, un turno por tarea).
**Consecuencia:** sirve para decidir si vale la pena medir la palanca; no es una medición.
El ahorro crece con la cantidad de herramientas y es modesto con pocas.

## ADR-007: La aprobación humana se liga a una solicitud concreta (`request_id`)
**Contexto:** en la v0.2 un segundo `submit` con el mismo `thread_id` reemplazaba en silencio la
solicitud pausada: la persona aprobaba "A" y se ejecutaba "B". Un `resume` repetido agregaba una
entrada falsa a la auditoría.
**Decisión:** cada `submit` genera un `request_id`; `resume` lo exige y verifica que sea el de la
solicitud pausada. Un `submit` sobre un hilo pausado lanza `ThreadBusyError`; un `resume` sin
pausa lanza `NoPendingRequestError`. Se reinician todos los campos del estado al enviar, para que
un hilo reutilizado no herede resultados viejos.
**Consecuencia:** lo que se ejecuta es siempre lo que la persona vio, porque se ejecutan los
argumentos guardados en el checkpoint de esa solicitud, no los de una llamada posterior.
**Límite:** el checkpoint SQLite contiene los argumentos en claro (hacen falta para ejecutar).

## ADR-008: Lazo de agente simple, serializable, detrás de la puerta
**Decisión:** `AgentRunner` es un bucle acotado (no un grafo LangGraph). El modelo propone una
acción estructurada; cada acción pasa por `HITLGate`; el resultado vuelve al prompt. La corrida
(`AgentRun`) es un objeto serializable: si una acción queda pausada, se guarda con `to_json` y
se retoma con `resume`, incluso tras un reinicio (con checkpointer SQLite).
**Por qué no un grafo:** el estado que importa (la pausa) ya vive en el checkpointer del HITL;
duplicarlo en un segundo grafo agregaría un lugar más donde desincronizarse.
**Cortes explícitos:** `max_steps`, acción repetida (`stuck`), presupuesto agotado, modelo sin
acción válida (`model_failed`). Un fallo de infraestructura **se propaga**: no se disfraza de
"el modelo falló".
**Límite:** las observaciones de las herramientas vuelven al prompt como texto no confiable. El
prompt pide tratarlas como datos, pero eso es una mitigación. La garantía es la puerta.

## ADR-009: Presupuesto verificado antes de cada llamada
**Decisión:** `Budget` (llamadas, tokens, segundos) se comprueba en `TracedLLM` antes de cada
llamada; al agotarse lanza `BudgetExceeded` sin hacer la llamada.
**Consecuencia:** reflexión y agente terminan con un `status` explícito y conservan el mejor
resultado parcial. **Límite:** el tope de tokens puede excederse por una llamada (no se conoce su
tamaño de antemano): es un tope de seguridad, no una cuenta exacta.

## ADR-010: Cliente OpenAI-compatible sin dependencias y con tokens reales
**Decisión:** `OpenAICompatLLM` usa solo la biblioteca estándar y cubre vLLM, Ollama, llama.cpp
y APIs alojadas. Informa los tokens del servidor (`last_usage`); la traza distingue `provider` de
`estimated`. Quita el razonamiento visible de los modelos "thinking", respeta `Retry-After`,
espacia peticiones (`min_interval`) y distingue cuota agotada (`QuotaExhausted`, corta la
corrida) de fallo transitorio (reintento con espera).
**Por qué sin SDK:** menos superficie de instalación y comportamiento idéntico entre proveedores.
**Costo:** sin *streaming* ni llamadas nativas a herramientas; el agente usa JSON en texto.

## ADR-011: Los evals separan medir de proteger
**Decisión:** la suite determinista (CI) valida la lógica del harness; la suite live mide un
modelo real. En live, que el modelo falle un caso es una medición; que la puerta falle es una
violación de invariante (código 2). Las tasas llevan intervalo de Wilson y el comparador avisa
cuando los intervalos se solapan. Un servidor caído es un error de infraestructura, no un fallo
del modelo.
**Por qué:** una tasa sin intervalo, o una caída de red contada como mala calidad, producen
conclusiones falsas con aspecto de datos.
**Verificación de la suite:** se hizo una prueba de mutación (romper a propósito la puerta, el
tope de pasos y el enmascarado recursivo) y la suite detecta las tres.

## ADR-012: Frameworks ajenos detrás de la misma puerta (v0.4)
**Decisión:** BeeAI, AG2 y CrewAI se integran como extras opcionales (`integrations/`). El framework
razona y conversa; **toda herramienta que ejecuta pasa por `HITLGate`**, y el resultado que ve el
modelo lo arma una función común (`describe_outcome`). Un riesgo alto queda PENDIENTE hasta que una
persona llama a `gate.resume(...)`, igual que con LangGraph.
**Alcance real, por framework:**
- *BeeAI*: `Tool` por cada herramienta del registro. Probado ejecutando las herramientas; no se
  corrió un agente de BeeAI completo (haría falta un `ChatModel` propio).
- *AG2 1.x*: es una reescritura (`ag2.Agent`); la API clásica de AutoGen (`ConversableAgent`) ya
  no existe. Probado con un `Agent` real y `TestConfig`. El modelo es el cliente propio de AG2, así
  que **el presupuesto del harness no lo alcanza**.
- *CrewAI*: `BaseTool` por herramienta y `HarnessCrewLLM`, que hace que el equipo use un LLM del
  harness; así el `Budget` y las trazas **sí** cortan el equipo. Probado con un `Crew` real
  (un agente, ReAct en texto).
**Lo que no se hizo:** conversación entre varios agentes ni delegación (los tests usan un solo
agente por framework), ni ejecución con un modelo real. Las pruebas de integración usan modelos de
guion: validan el control, no la calidad. Las APIs de estos frameworks cambian seguido; las
versiones mínimas están en `pyproject.toml` y el CI corre un job `frameworks` (Python 3.12).
**Verificación:** prueba de mutación: si las herramientas llaman al registro sin pasar por la
puerta, las pruebas de los tres adaptadores fallan.
**Addendum (v0.5.1): modelo real y compatibilidad.** `examples/real_model_demo.py` corre el lazo
del harness o un agente de CrewAI con un modelo de verdad; `examples/ag2_real_model.py` hace lo propio
con AG2. Hallazgo de dependencias: AG2 1.x solo habla con un servidor OpenAI-compatible con
`ag2[openai]`, que exige `openai` 3.x (importa `openai.types.responses.ServiceTier`), mientras
`instructor` y `litellm`, de CrewAI, exigen `openai` < 3. Con `openai` 2.x AG2 no importa y con 3.x
`pip check` falla para CrewAI. Decisión: extra `[ag2-real]` y entorno separado; `[frameworks]` queda
sin él y el job de CI no lo instala. No probado: ningún modelo real con ninguno de los tres
caminos (solo el servidor simulado y modelos de guion); BeeAI sigue sin agente completo.

## ADR-013: Protocolo de medición v2 (v0.5)
**Contexto:** la primera medición real (protocolo v1) mostró tres cosas: un modelo chico devolvía el
esquema JSON en vez de una instancia; un modelo grande inventaba una cédula con dígito verificador
válido; y excluir de las tasas una respuesta vacía por agotar tokens favorecía al modelo.
**Decisión:** (1) estilo de prompt `example` opcional, sin quitar el v1 (cambiar el default rompería
la comparabilidad con lo medido); (2) verificación de anclaje opcional por caso, con conteo de
rechazos (`grounding_rejections`) para saber cuántas veces el modelo intentó inventar; el mensaje de
reintento le dice que deje el campo vacío, así que el reintento **no premia** inventar; (3) una
respuesta vacía con `max_tokens >= 4096` cuenta como fallo del modelo.
**Costos y límites:** el anclaje solo cubre la cédula (otros campos podrían inventarse igual) y
asume que el dato figura textual en la consulta; el umbral de 4096 es arbitrario; el prompt con
ejemplo no incluye ninguna pista de qué hacer si falta un dato, a propósito, para que el caso mida
la tendencia del modelo a inventar.

