# Cambios

Formato: una sección por versión, de la más nueva a la más vieja. Antes de la 1.0 la interfaz puede
cambiar entre versiones menores; lo que sí se anota siempre es cuándo cambia el **protocolo de
medición**, porque los números de un protocolo no se comparan con los de otro (ver la tabla al final).

## 0.7.1 (27/11/2026)

- `examples/real_model_demo.py`: `--timeout` (300 s por defecto; antes eran 120 s y NVIDIA Build cortaba
  con "The read operation timed out").
- El mismo ejemplo desactiva la pregunta de CrewAI de subir la traza de la ejecución a sus
  servidores (`CREWAI_TRACING_ENABLED=false`); se reactiva exportando `true`.

## 0.7.0 (27/11/2026)

- README reescrito: mediciones con el protocolo v5 (super-120B, ultra-550B, Mistral Large 4,
  Qwen2.5-7B), el experimento `neutral` (6/6 contra 0/6 con el esquema estándar, solo en un modelo),
  lista de API pública, estado de verificación y límites al día. Lo invalidado queda dicho.
- Las mediciones de los protocolos v1 a v3 pasan a `docs/historial-mediciones.md`.
- Sin cambios de código ni de protocolo: los números de la 0.6.6 y la 0.6.7 siguen valiendo.

## 0.6.7 (27/11/2026)

- El reporte en texto declara la configuración que se aparta del protocolo estándar (variante de
  esquema, `--extra-body`, system prompt, `--timeout`, estilo de prompt). Antes un reporte pegado no
  decía si se había usado la variante `neutral`.
- Notebook: explica el código de salida 2 (error de infraestructura o violación de invariante) y 3
  (corrida cortada por cuota).

## 0.6.6 (26/11/2026)

- `harness-compare` ya no contrasta con intervalos reportes que no son comparables: casos distintos,
  protocolo distinto, variante de esquema distinta o estilo de prompt distinto. Antes un 0/6 de los
  dos casos más difíciles se comparaba con una suite de 19 casos y concluía "los intervalos no se
  solapan". Avisa cuando se mezclan protocolos.
- Notebook: desplegables de proveedor y de modelo; el catálogo en vivo excluye lo que no es de chat
  (embeddings, rerank, guardas); la clave de cada proveedor se busca en el entorno, en los Secretos
  de Colab o de Kaggle, y si no, con `getpass`.

## 0.6.5 (26/11/2026) · protocolo v5

- El esqueleto del agente deja de usar `a | null` (ADR-016): Qwen2.5-7B copiaba
  `"tool": "buscar_normativa | null"`.

## 0.6.4 (26/11/2026)

- `--ticket-schema neutral`: **experimento**, no el protocolo. Quita "con verificador válido" de la
  descripción de `citizen_id` (la validación es la misma) para separar "el modelo inventa" de "el
  prompt lo empuja a inventar". Queda en `meta.ticket_schema`.

## 0.6.3 (26/11/2026)

- `--timeout` (solo `--llm compat`): segundos de espera por respuesta. Con 120 s por defecto, Mistral
  Large 4 en NVIDIA Build cortaba 5 de 19 casos.

## 0.6.2 (26/11/2026) · protocolo v4

- Corrige el esqueleto del paso del agente (ADR-015). Con `--prompt-style example` se mostraba
  `"arguments": "<texto>"` y las dos acciones a la vez, así que un modelo que copiaba la forma no
  podía producir una acción válida. En el protocolo v3, 18 de 20 fallos de agente eran
  `arguments` como texto. **Los resultados de agente del v3 no miden a los modelos.**

## 0.6.1 (25/11/2026)

- `--extra-body` y `--system-prompt` para proveedores con parámetros propios (por ejemplo
  `reasoning_effort`); quedan en el reporte.
- Notebook `live_eval_modelos_gratis.ipynb` (NVIDIA Build, Groq, OpenRouter y un modelo local en una T4).

## 0.6.0 (24/11/2026) · protocolo v3

- Los casos de agente exigen `status: done`; `fabricated_attempts` mide los intentos de inventar una
  cédula sobre los textos crudos (cuenta cualquier número que no esté en la consulta, sea o no
  válido); el reporte vuelca `detail` y `last_output` del agente y separa lo completado de lo
  aprobado por inacción (ADR-014).

## 0.5.1 (24/11/2026)

- Ejemplos ejecutables en `examples/` (HITL, agente con rechazo, modelo real, AG2 con modelo real),
  cubiertos por `tests/test_examples.py`.

## 0.5.0 (24/11/2026) · protocolo v2

- Integraciones opcionales de frameworks (BeeAI, AG2, CrewAI) con la misma puerta HITL.
- `--prompt-style example` (instancia de relleno en vez del esquema JSON completo) y chequeo de que la
  cédula aparezca en la consulta (`ci_grounding_check`).

## 0.4.0 (23/11/2026)

- Agente, supervisión humana (HITL), presupuesto, evals con modelos reales y las primeras
  integraciones de frameworks.

## Protocolos de medición

| Protocolo | Desde | Qué cambia | Comparable con |
|---|---|---|---|
| v1 | 0.4.0 | prompt con el esquema JSON completo | solo v1 |
| v2 | 0.5.0 | instancia de relleno, anclaje de la cédula | solo v2 |
| v3 | 0.6.0 | `status: done` en agente, intentos de invención, completados | structured y reflection con v4 y v5 |
| v4 | 0.6.2 | prompt del agente corregido | structured y reflection con v3 y v5 |
| v5 | 0.6.5 | esqueleto del agente sin `a \| null` | todo, con v5; structured y reflection también con v3 y v4 |

Un reporte con `meta.ticket_schema` distinto de `standard` es un experimento y no se compara con los demás.
