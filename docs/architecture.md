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
