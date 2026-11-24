# Medir modelos reales sin pagar: guía de proveedores

El harness habla con cualquier servidor compatible con la API de OpenAI
(`/chat/completions`). Para medir un modelo basta:

```bash
# --prompt-style example = protocolo v2; sin ese flag se usa el v1 (esquema JSON completo)
export HARNESS_API_KEY=...        # no hace falta en servidores locales
python -m centro_harness.evals --cases evals/live_cases.jsonl --llm compat \
    --base-url <URL>/v1 --model <ID-DEL-MODELO> \
    --repeats 1 --max-tokens 4096 --min-interval 3.1 --max-total-calls 45 \
    --prompt-style example \
    --label "<proveedor>/<modelo> 2026-11-22" --out-json reports/<nombre>.json
python -m centro_harness.compare reports/*.json --out-md reports/comparacion.md
```

> **Estado de verificación.** Lo que está probado en este repo es el cliente HTTP, el manejo de
> modelos de razonamiento, los límites de velocidad y el corte por cuota, **contra servidores
> simulados** (`tests/test_providers.py`, `tests/test_llm_compat.py`). **No se corrió contra
> ningún proveedor real.** Las cuotas de abajo salen de documentación pública y de fuentes
> secundarias consultadas el 22/11/2026; los planes gratuitos cambian sin aviso. Confirmá cada
> dato en el sitio oficial antes de apoyarte en él.

> **Primera medición real (23/11/2026):** `nvidia/nemotron-nano-9b-v2:free`, tomado de un listado
> secundario, devolvió 404 ("no longer available"). El harness lo registró como error de
> infraestructura (19 casos excluidos de las tasas), que es lo correcto. Antes de gastar cuota,
> confirmá el ID en el catálogo vivo del proveedor (en OpenRouter, `GET /api/v1/models`: los
> gratuitos terminan en `:free`). Un solo caso alcanza para probar un ID: `--only salud`.

## Cuánto cuesta una pasada (la cuenta que condiciona todo)

Una pasada de `evals/live_cases.jsonl` (19 casos, `--repeats 1`) usa **al menos ~31 llamadas y
~10 000 tokens**. Ese es el piso, medido con un modelo simulado que nunca se equivoca. Un modelo
débil necesita reparaciones de JSON y reflexiones más largas: contá 40–70 llamadas.

Con decodificación voraz (temperatura 0) repetir un caso casi siempre da lo mismo, así que
`--repeats` mayor a 1 consume cuota sin aportar información en la mayoría de los proveedores.
Preferí agregar casos (variedad) antes que repeticiones.

## Opciones

| Opción | Modelos de tu lista | Límites (según las fuentes) | Datos |
|---|---|---|---|
| **Local / Colab / Kaggle** con pesos abiertos (transformers, vLLM, Ollama) | Los que entren en tu GPU. Una T4 de 16 GB no aloja un modelo de ~30B ni en 4 bits (solo los pesos serían ~15 GB) | Sin cuota. Reproducible: controlás versión y cuantización | No salen de tu máquina |
| **OpenRouter**, modelos con sufijo `:free` | Vistos en un listado secundario (pueden haber cambiado): `nvidia/nemotron-3-nano-30b-a3b:free`, ~~`nvidia/nemotron-nano-9b-v2:free`~~ (**404 el 23/11/2026: ya no lo sirve ningún proveedor**), `qwen/qwen3-next-80b-a3b-instruct:free`, `qwen/qwen3-4b:free`, `z-ai/glm-4.5-air:free`, `openai/gpt-oss-120b:free`. DeepSeek y Kimi no figuraban como gratis | 20 pedidos/min. **50 pedidos/día**, o 1 000/día si compraste al menos 10 créditos. **La cuota es una sola para todos los modelos gratis** | Depende del proveedor que sirve cada modelo; algunos pueden entrenar con tus prompts |
| **NVIDIA API Catalog** (build.nvidia.com) | Familia Nemotron y otros | Sin cifra oficial publicada. Usuarios del foro de NVIDIA mencionan ~40 pedidos/min. Uso orientado a prototipado/desarrollo; un usuario afirma que la producción está prohibida por los términos | No consta en las fuentes consultadas |
| **Z.ai (GLM)** | `GLM-4.7-Flash`, `GLM-4.5-Flash`, `GLM-4.6V-Flash` | No publicados; figuran en la consola con sesión iniciada | No entrena salvo que lo aceptes |
| **Alibaba Model Studio (Qwen)** | Qwen | Crédito de 1 M de tokens por modelo, 90 días, solo región Singapur. **Después de la cuota empieza a facturar automáticamente salvo que actives "Free Quota Only"**, que viene apagado | No entrena sin consentimiento aparte |
| **DeepSeek** | DeepSeek | Sin capa gratuita (prepago) al 18/10/2026 | La política general indica que las entradas pueden usarse para entrenar |
| **Moonshot (Kimi)** | Kimi | Sin capa gratuita: hay que recargar al menos US$ 1 | Entrena salvo acuerdo escrito |
| **Groq** (fuera de tu lista, útil) | `openai/gpt-oss-120b` | 30 pedidos/min, 1 000/día | No entrena |

Créditos de noviembre de 2026, de **una sola fuente secundaria** (verificá antes de registrarte):
Baidu Qianfan (DeepSeek, Qwen3, Kimi K2; 1 M de tokens por modelo, 3 meses) y AMD Token Lab
(DeepSeek V4, Qwen3.8, GLM-5.3; 30 días, ~20 pedidos/min). Desconfiá de los "relays" de terceros
que revenden acceso: tus prompts pasan por un intermediario desconocido.

## Cómo no quemar la cuota

| Flag | Para qué |
|---|---|
| `--min-interval 3.1` | Espacia las peticiones para quedar bajo 20 por minuto (60 s / 20 = 3 s, más margen) |
| `--max-total-calls 45` | Tope global de llamadas. Al alcanzarlo la corrida **se corta** y deja un reporte parcial marcado como `MEDICIÓN INCOMPLETA` (código de salida 3) |
| `--max-tokens 4096` | Obligatorio con modelos de razonamiento (ver abajo) |
| `--repeats 1` | Ver arriba |

Si el proveedor responde 429 y pide esperar más de 60 s (`Retry-After`, o `X-RateLimit-Reset`
lejano), el cliente lo trata como **cuota agotada**: no reintenta y la corrida termina con
reporte parcial. Con esperas cortas respeta `Retry-After`. El formato de esos encabezados varía
entre proveedores y el cliente interpreta el reinicio con el mejor criterio posible (puede ser
relativo o una marca de tiempo): tratalo como mejor esfuerzo.

Para repartir una medición en varios días (la cuota se renueva por día) corré el mismo comando
con otro `--label` cada día y compará los reportes; no hay "reanudar" automático.

## Modelos de razonamiento (DeepSeek, Qwen3, GLM, Kimi, Nemotron...)

Muchos devuelven su razonamiento antes de la respuesta, ya sea dentro de `<think>...</think>` o
en un campo aparte. El cliente:

- **quita** el bloque `<think>` antes de buscar el JSON (si no, las llaves del razonamiento
  pueden confundir al extractor);
- **ignora** los campos `reasoning` / `reasoning_content`: son pensamiento, no la respuesta;
- si la respuesta queda **vacía porque se agotó `max_tokens` pensando**, lanza un error de
  configuración que pide subir `--max-tokens`, en vez de registrar mil fallos del modelo.

Con `--max-tokens` bajo (512 por defecto) estos modelos fallan por razones que no son suyas.
Los modelos de razonamiento también gastan muchos más tokens por llamada: contalo en la cuota.

## Qué significa realmente un resultado de proveedor

Un número obtenido por API mide **"este modelo, tal como lo sirve ese proveedor ese día"**:
cuantización, parámetros de muestreo, plantilla de chat y versión son decisiones del proveedor y
pueden cambiar. Por eso:

- registrá proveedor, modelo y fecha en `--label` (queda en el reporte);
- no compares un modelo por API contra uno local como si fueran "el mismo modelo";
- no saques conclusiones de diferencias entre intervalos de Wilson que se solapan (la herramienta
  `harness-compare` lo marca).

## Privacidad

Las capas gratuitas pueden usar tus prompts para entrenar (varias lo declaran). Los casos de
`evals/live_cases.jsonl` usan **solo datos sintéticos** (cédulas inventadas, trámites de ejemplo).
**No agregues casos con datos reales de ciudadanos o pacientes** para medir contra un proveedor
externo. Si necesitás medir con datos reales, usá un modelo local.

## Lo que el harness NO usa

El harness no usa llamadas nativas a herramientas (*function calling*): el agente emite su acción
como JSON en texto y el harness lo valida. Eso permite medir también modelos o servidores sin
soporte de function calling, a costa de depender de que el modelo siga la instrucción de formato
(para eso existe el bucle de reparación). La opción `--json-mode` pide `response_format:
json_object` y puede no estar soportada por algunos modelos o proveedores; no se activa sola.
