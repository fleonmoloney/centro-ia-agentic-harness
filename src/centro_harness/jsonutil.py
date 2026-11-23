"""Utilidades de JSON para salidas de modelos de lenguaje."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> Any:
    """Extrae el primer objeto JSON de un texto de modelo.

    Acepta bloques ```json ... ``` y prosa alrededor del objeto, que es lo que suelen devolver
    los modelos pequeños. Lanza `ValueError` si no hay ningún objeto válido.
    """
    candidates = []
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("la respuesta no contiene un objeto JSON válido")
