"""Integraciones opcionales con frameworks de agentes.

Regla de diseño: el framework ajeno puede razonar y conversar, pero **toda herramienta que
ejecute pasa por `HITLGate`**. Así la supervisión humana, la auditoría y el enmascarado de datos
personales valen igual con LangGraph, BeeAI o AG2. Cada módulo importa su framework al usarse
(extras: `pip install -e ".[beeai]"`, `".[ag2]"`)."""
