"""centro_harness: harness agéntico de referencia (reflexión, HITL, estructurados, evals)."""

from centro_harness.hitl import GateOutcome, HITLGate, Risk, ToolRegistry, ToolSpec
from centro_harness.llm import LLM, OpenAILLM, ScriptedLLM, build_llm
from centro_harness.reflection import Evaluation, build_reflection_graph, run_reflection
from centro_harness.schemas import (
    InstitutionalTicket,
    MedicalProtocol,
    StructuredOutputError,
    structured_with_retry,
)
from centro_harness.tracing import TracedLLM, Tracer

__version__ = "0.2.0"

__all__ = [
    "LLM",
    "Evaluation",
    "GateOutcome",
    "HITLGate",
    "InstitutionalTicket",
    "MedicalProtocol",
    "OpenAILLM",
    "Risk",
    "ScriptedLLM",
    "StructuredOutputError",
    "ToolRegistry",
    "ToolSpec",
    "TracedLLM",
    "Tracer",
    "build_llm",
    "build_reflection_graph",
    "run_reflection",
    "structured_with_retry",
]
