"""Capa de acceso al LLM.

El harness depende de un protocolo mínimo (`LLM`), no de un proveedor. Eso permite:

* correr tests y evals de forma determinista y gratuita (`ScriptedLLM`),
* cambiar a un modelo real sin tocar los grafos (`OpenAILLM`, extra opcional).
"""

from __future__ import annotations

import os
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    """Contrato mínimo que necesita el harness."""

    def generate(self, prompt: str, *, label: str = "") -> str:
        """Devuelve texto libre."""
        ...

    def structured(self, prompt: str, schema: type[T], *, label: str = "") -> T:
        """Devuelve una instancia de `schema` ya validada o lanza `ValidationError`."""
        ...


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
        # model_validate lanza ValidationError si el guion trae un dato inválido a propósito.
        return schema.model_validate(self._next(self._structured, "estructurada"))


class OpenAILLM:
    """Adaptador a OpenAI vía `langchain-openai` (instalar con `pip install .[openai]`).

    Requiere la variable de entorno OPENAI_API_KEY. Nota: este adaptador no se ejecuta en CI
    (no hay clave); está pensado para correr los evals contra un modelo real de forma manual.
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


def build_llm(name: str = "scripted") -> LLM:
    """Fábrica simple. `scripted` devuelve un guion vacío (cada caso arma el suyo)."""
    if name == "scripted":
        return ScriptedLLM()
    if name == "openai":
        return OpenAILLM()
    raise ValueError(f"LLM desconocido: {name!r} (opciones: scripted, openai)")
