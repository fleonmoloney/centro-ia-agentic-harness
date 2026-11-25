"""Capa de acceso al LLM.

El harness depende de un protocolo mínimo (`LLM`), no de un proveedor. Eso permite:

* correr tests y evals de forma determinista y gratuita (`ScriptedLLM`),
* medir un modelo real en local, Colab o Kaggle, sin cuenta ni costo, con un servidor
  compatible con la API de OpenAI (vLLM, Ollama, llama.cpp) vía `OpenAICompatLLM`,
  o cargando el modelo en el propio proceso con `TransformersLLM`,
* usar OpenAI vía LangChain (`OpenAILLM`, extra opcional).

Un backend puede exponer `last_usage = (tokens_entrada, tokens_salida)` tras cada llamada; la
traza lo usa en lugar de la estimación por caracteres.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Protocol, TypeVar

from pydantic import BaseModel

from centro_harness.jsonutil import extract_json

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Falla de infraestructura del backend (red, HTTP, respuesta ilegible).

    No es un `ValueError`: el bucle de reparación de salidas estructuradas no la reintenta, porque
    pedirle al modelo que "corrija" no arregla un servidor caído.
    """


class TruncatedResponse(LLMError):
    """Respuesta vacía porque el modelo agotó `max_tokens` (típico en modelos de razonamiento)."""

    def __init__(self, message: str, max_tokens: int) -> None:
        super().__init__(message)
        self.max_tokens = max_tokens


class QuotaExhausted(LLMError):
    """El proveedor agotó la cuota (p. ej. límite diario) o pide esperar más de lo razonable.

    A diferencia de un fallo transitorio, reintentar no sirve: el runner de evals corta la corrida
    y entrega un reporte parcial en lugar de seguir quemando llamadas contra un límite agotado.
    """


class LLM(Protocol):
    """Contrato mínimo que necesita el harness."""

    def generate(self, prompt: str, *, label: str = "") -> str:
        """Devuelve texto libre."""
        ...

    def structured(self, prompt: str, schema: type[T], *, label: str = "") -> T:
        """Devuelve una instancia de `schema` ya validada o lanza `ValueError`."""
        ...


def strip_reasoning(text: str) -> str:
    """Quita el razonamiento visible que emiten algunos modelos ("thinking").

    * `... </think> respuesta`  -> `respuesta` (plantillas que ponen `<think>` en el prompt),
    * `<think> ...` sin cierre   -> "" (se cortó pensando: no hay respuesta),
    * sin etiquetas              -> el texto tal cual.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    if text.lstrip().startswith("<think>"):
        return ""
    return text


# --------------------------------------------------------------------------- salida estructurada
PROMPT_STYLES = ("schema", "example")


def _placeholder(prop: dict) -> object:
    """Valor de relleno de un campo, para mostrar la FORMA de la respuesta (no un esquema)."""
    if "enum" in prop:
        return " | ".join(str(v) for v in prop["enum"])
    kind = prop.get("type")
    if kind == "boolean":
        return False
    if kind in ("integer", "number"):
        return 0
    if kind == "array":
        return []
    return "<texto>"


def example_skeleton(schema: type[BaseModel]) -> dict:
    """Instancia de relleno del esquema: mismas claves, valores de ejemplo."""
    props = schema.model_json_schema().get("properties", {})
    return {name: _placeholder(prop) for name, prop in props.items()}


def schema_prompt(prompt: str, schema: type[BaseModel], style: str = "schema") -> str:
    """Agrega al prompt la instrucción y el formato para que un modelo de texto devuelva JSON.

    * `schema` (protocolo original, v1): incluye el esquema JSON completo. Los modelos chicos
      suelen devolver el propio esquema en vez de una instancia (medido con Qwen2.5-3B).
    * `example` (v2): muestra un objeto de relleno con las mismas claves y la lista de campos.
      No agrega ninguna pista sobre qué hacer si falta un dato: eso lo mide el caso.
    """
    if style not in PROMPT_STYLES:
        raise ValueError(f"estilo de prompt desconocido: {style!r} (opciones: {PROMPT_STYLES})")
    if style == "example":
        shape = json.dumps(example_skeleton(schema), ensure_ascii=False, indent=2)
        fields = "\n".join(
            f"- {name}: {prop['description']}"
            for name, prop in schema.model_json_schema().get("properties", {}).items()
            if prop.get("description")
        )
        extra = f"\nSignificado de los campos:\n{fields}" if fields else ""
        return (
            f"{prompt}\n\nRespondé SOLO con un objeto JSON válido (sin texto antes ni después) "
            "con EXACTAMENTE estas claves. Reemplazá los valores de ejemplo por los reales; "
            f"en los campos con opciones separadas por | elegí una sola:\n{shape}{extra}"
        )
    spec = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    return (
        f"{prompt}\n\nRespondé SOLO con un objeto JSON válido (sin texto antes ni después) que "
        f"cumpla este esquema JSON:\n{spec}"
    )


def structured_from_text(
    generate, prompt: str, schema: type[T], record=None, style: str = "schema"
) -> T:
    """Pide JSON como texto, lo extrae y lo valida. Lanza `ValueError` si no sirve.

    Es el camino de los modelos locales: no hay garantía de formato, por eso el resultado
    siempre pasa por Pydantic y, si falla, por el bucle de reparación.
    """
    text = generate(schema_prompt(prompt, schema, style))
    if record is not None:
        record(text)  # para diagnosticar: qué devolvió realmente el modelo
    return schema.model_validate(extract_json(text))


# --------------------------------------------------------------------------- backends
class ScriptedLLM:
    """LLM de guion: devuelve respuestas predefinidas, en orden.

    Cuando queda una sola respuesta en la cola, se repite indefinidamente. Sirve para probar la
    lógica de control del harness (ruteo, reintentos, HITL, trazas). NO mide la calidad de
    ningún modelo.
    """

    def __init__(
        self,
        texts: list[str] | None = None,
        structured: list[dict] | None = None,
    ) -> None:
        self._texts = list(texts or [])
        self._structured = list(structured or [])
        self.prompts: list[str] = []

    @staticmethod
    def _next(queue: list, what: str):
        if not queue:
            raise RuntimeError(f"ScriptedLLM: no hay respuestas de tipo '{what}' en el guion")
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def generate(self, prompt: str, *, label: str = "") -> str:
        self.prompts.append(prompt)
        return str(self._next(self._texts, "texto"))

    def structured(self, prompt: str, schema: type[T], *, label: str = "") -> T:
        self.prompts.append(prompt)
        item = self._next(self._structured, "estructurada")
        self.last_text = json.dumps(item, ensure_ascii=False)  # como el texto crudo de un modelo
        # model_validate lanza ValidationError si el guion trae un dato inválido a propósito.
        return schema.model_validate(item)


class OpenAICompatLLM:
    """Cliente de la API `/chat/completions`, sin dependencias externas (solo biblioteca estándar).

    Sirve para cualquier servidor compatible: vLLM (`--port 8000`), Ollama
    (`http://localhost:11434/v1`), llama.cpp (`llama-server`) u OpenAI. Informa los tokens reales
    que devuelve el servidor en `last_usage`.

    * 429 y 5xx se reintentan `retries` veces con espera exponencial; el resto falla de inmediato.
    * Cualquier falla de red o de formato lanza `LLMError`.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 120.0,
        json_mode: bool = False,
        retries: int = 2,
        backoff: float = 0.5,
        system_prompt: str | None = None,
        min_interval: float = 0.0,
        max_retry_wait: float = 60.0,
        prompt_style: str = "schema",
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("HARNESS_BASE_URL") or "http://localhost:8000/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("HARNESS_MODEL", "")
        if not self.model:
            raise ValueError("Falta el nombre del modelo (parámetro `model` o HARNESS_MODEL)")
        self.api_key = api_key or os.environ.get("HARNESS_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.json_mode = json_mode
        self.retries = retries
        self.backoff = backoff
        self.system_prompt = system_prompt
        self.min_interval = min_interval  # segundos mínimos entre peticiones (límites de RPM)
        self.max_retry_wait = max_retry_wait  # esperas mayores se tratan como cuota agotada
        self.prompt_style = prompt_style
        self._last_request = 0.0
        self.last_usage: tuple[int, int] | None = None

    def _pace(self) -> None:
        if self.min_interval > 0 and self._last_request:
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
        self._last_request = time.monotonic()

    @staticmethod
    def _retry_wait(headers) -> float | None:
        """Segundos que pide esperar el servidor (Retry-After o X-RateLimit-Reset), si los hay.

        Mejor esfuerzo: los proveedores no coinciden en el formato del reinicio, así que un valor
        enorme se interpreta como marca de tiempo (ms o s desde 1970).
        """
        retry_after = headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        reset = headers.get("X-RateLimit-Reset")
        if reset:
            try:
                value = float(reset)
            except ValueError:
                return None
            if value > 1e12:  # milisegundos desde 1970
                return max(0.0, value / 1000 - time.time())
            if value > 1e9:  # segundos desde 1970
                return max(0.0, value - time.time())
            return max(0.0, value)
        return None

    def _post(self, payload: dict) -> dict:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(payload).encode("utf-8")
        for attempt in range(self.retries + 1):
            self._pace()
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")[:300]
                if exc.code in (429, 500, 502, 503, 504):
                    asked = self._retry_wait(exc.headers)
                    if asked is not None and asked > self.max_retry_wait:
                        raise QuotaExhausted(
                            f"HTTP {exc.code} de {url}: el servidor pide esperar ~{asked:.0f} s "
                            f"(más que max_retry_wait={self.max_retry_wait:.0f} s). {body}"
                        ) from exc
                    if attempt < self.retries:
                        time.sleep(asked if asked is not None else self.backoff * 2**attempt)
                        continue
                raise LLMError(f"HTTP {exc.code} de {url}: {body}") from exc
            except json.JSONDecodeError as exc:
                raise LLMError(f"La respuesta de {url} no es JSON: {exc}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt < self.retries:
                    time.sleep(self.backoff * 2**attempt)
                    continue
                raise LLMError(f"No se pudo conectar con {url}: {exc}") from exc
        raise LLMError(f"Sin respuesta de {url}")  # pragma: no cover - el bucle siempre retorna

    def _complete(self, prompt: str, *, json_mode: bool) -> str:
        self.last_usage = None
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._post(payload)
        try:
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError(
                f"Respuesta sin 'choices[0].message.content': {str(data)[:200]}"
            ) from exc
        usage = data.get("usage") or {}
        if "prompt_tokens" in usage and "completion_tokens" in usage:
            self.last_usage = (int(usage["prompt_tokens"]), int(usage["completion_tokens"]))
        answer = strip_reasoning(str(content))
        if not answer.strip() and choice.get("finish_reason") == "length":
            # El modelo gastó todo el límite pensando: no es un error del modelo sino de
            # configuración, y seguir midiendo así daría resultados inválidos.
            raise TruncatedResponse(
                f"Respuesta vacía por límite de tokens (max_tokens={self.max_tokens}): un modelo "
                "de razonamiento gastó todo pensando. Subí --max-tokens (p. ej. 4096).",
                self.max_tokens,
            )
        return answer

    def generate(self, prompt: str, *, label: str = "") -> str:
        return self._complete(prompt, json_mode=False)

    def structured(self, prompt: str, schema: type[T], *, label: str = "") -> T:
        return structured_from_text(
            lambda p: self._complete(p, json_mode=self.json_mode),
            prompt,
            schema,
            record=lambda t: setattr(self, "last_text", t),
            style=self.prompt_style,
        )


class TransformersLLM:
    """Carga un modelo de Hugging Face en el propio proceso (pensado para Colab/Kaggle con GPU).

    Instalar con `pip install '.[local]'`. `quantization="4bit"` usa bitsandbytes (NF4).
    Decodificación voraz (determinista) por defecto. Informa tokens reales en `last_usage`.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        *,
        quantization: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        device_map: str = "auto",
        prompt_style: str = "schema",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise RuntimeError("Falta el extra: pip install '.[local]'") from exc
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.prompt_style = prompt_style
        self.last_usage: tuple[int, int] | None = None
        self._tok = AutoTokenizer.from_pretrained(model_id)
        kwargs: dict = {"device_map": device_map}
        if quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif quantization is not None:
            raise ValueError("quantization debe ser None o '4bit'")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:  # transformers recientes renombraron torch_dtype -> dtype
            self._model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kwargs)
        except TypeError:  # pragma: no cover - versiones anteriores
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, **kwargs
            )
        self._model.eval()

    def generate(self, prompt: str, *, label: str = "") -> str:
        import torch

        self.last_usage = None
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        options: dict = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            options.update(do_sample=True, temperature=self.temperature)
        else:
            options["do_sample"] = False
        with torch.no_grad():
            output = self._model.generate(**inputs, **options)
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output[0][prompt_len:]
        self.last_usage = (int(prompt_len), int(new_tokens.shape[0]))
        return self._tok.decode(new_tokens, skip_special_tokens=True)

    def structured(self, prompt: str, schema: type[T], *, label: str = "") -> T:
        return structured_from_text(
            self.generate,
            prompt,
            schema,
            record=lambda t: setattr(self, "last_text", t),
            style=self.prompt_style,
        )


class OpenAILLM:
    """Adaptador a OpenAI vía `langchain-openai` (instalar con `pip install .[openai]`).

    Requiere la variable de entorno OPENAI_API_KEY. Este adaptador no se ejecuta en CI (no hay
    clave) ni se midió en este repo; para medir modelos usá `OpenAICompatLLM`, que informa
    tokens reales y se prueba contra un servidor HTTP simulado.
    """

    def __init__(self, model: str | None = None, temperature: float = 0.0) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise RuntimeError("Falta el extra: pip install '.[openai]'") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Falta la variable de entorno OPENAI_API_KEY")
        name = model or os.environ.get("HARNESS_OPENAI_MODEL", "gpt-4o-mini")
        self._chat = ChatOpenAI(model=name, temperature=temperature)

    def generate(self, prompt: str, *, label: str = "") -> str:
        return str(self._chat.invoke(prompt).content)

    def structured(self, prompt: str, schema: type[T], *, label: str = "") -> T:
        result = self._chat.with_structured_output(schema).invoke(prompt)
        return schema.model_validate(result)


def build_llm(name: str = "scripted", **options) -> LLM:
    """Fábrica. `scripted` devuelve un guion vacío (cada caso arma el suyo).

    `compat` (servidor OpenAI-compatible) acepta `base_url` y `model`; `transformers` acepta
    `model` (id de Hugging Face) y `quantization`.
    """
    if name == "scripted":
        return ScriptedLLM()
    if name == "compat":
        return OpenAICompatLLM(
            base_url=options.get("base_url"),
            model=options.get("model"),
            json_mode=options.get("json_mode", False),
            max_tokens=options.get("max_tokens", 512),
            retries=options.get("retries", 2),
            min_interval=options.get("min_interval", 0.0),
            prompt_style=options.get("prompt_style", "schema"),
        )
    if name == "transformers":
        names = ("quantization", "max_new_tokens", "prompt_style")
        kwargs = {k: options[k] for k in names if k in options}
        if "max_tokens" in options:  # mismo flag de la CLI para todos los backends
            kwargs["max_new_tokens"] = options["max_tokens"]
        if options.get("model"):
            return TransformersLLM(options["model"], **kwargs)
        return TransformersLLM(**kwargs)
    if name == "openai":
        return OpenAILLM(options.get("model"))
    raise ValueError(
        f"LLM desconocido: {name!r} (opciones: scripted, compat, transformers, openai)"
    )
