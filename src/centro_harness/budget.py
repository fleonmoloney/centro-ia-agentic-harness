"""Presupuesto de ejecución: tope de llamadas, tokens y tiempo para una corrida completa.

`max_iterations` (reflexión) y `max_steps` (agente) acotan los *ciclos*. El presupuesto acota
el *costo total*, que es lo que importa cuando cada ciclo hace varias llamadas (generar,
evaluar, reparar). Se verifica ANTES de cada llamada al modelo, de modo que la última llamada
puede pasarse del tope de tokens (no se conoce su tamaño de antemano). Es un tope de seguridad,
no una cuenta exacta.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    """Se alcanzó un tope del presupuesto; no se hizo la llamada que lo habría superado."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"Presupuesto agotado ({reason}): {detail}")
        self.reason = reason
        self.detail = detail


@dataclass
class Budget:
    max_calls: int | None = None
    max_tokens: int | None = None  # entrada + salida
    max_seconds: float | None = None
    _started: float | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        """Reinicia el reloj. Se llama solo al primer `check` si no se llamó antes."""
        self._started = time.monotonic()

    def check(self, *, calls: int, tokens: int) -> None:
        """Lanza `BudgetExceeded` si el próximo paso ya no entra en el presupuesto."""
        if self._started is None:
            self.start()
        if self.max_calls is not None and calls >= self.max_calls:
            raise BudgetExceeded("max_calls", f"{calls} de {self.max_calls} llamadas")
        if self.max_tokens is not None and tokens >= self.max_tokens:
            raise BudgetExceeded("max_tokens", f"{tokens} de {self.max_tokens} tokens")
        if self.max_seconds is not None:
            elapsed = time.monotonic() - (self._started or time.monotonic())
            if elapsed >= self.max_seconds:
                raise BudgetExceeded("max_seconds", f"{elapsed:.1f} s de {self.max_seconds} s")
