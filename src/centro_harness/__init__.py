"""centro_harness: harness agéntico de referencia (agente, HITL, presupuesto, evals)."""

from centro_harness.agent import AgentAction, AgentRun, AgentRunner
from centro_harness.budget import Budget, BudgetExceeded
from centro_harness.hitl import (
    GateOutcome,
    HITLGate,
    NoPendingRequestError,
    RequestMismatchError,
    Risk,
    ThreadBusyError,
    ToolRegistry,
    ToolSpec,
    sqlite_checkpointer,
)
from centro_harness.llm import (
    LLM,
    LLMError,
    OpenAICompatLLM,
    OpenAILLM,
    QuotaExhausted,
    ScriptedLLM,
    TransformersLLM,
    build_llm,
)
from centro_harness.reflection import Evaluation, build_reflection_graph, run_reflection
from centro_harness.schemas import (
    InstitutionalTicket,
    MedicalProtocol,
    StructuredOutputError,
    structured_with_retry,
)
from centro_harness.tracing import TracedLLM, Tracer

__version__ = "0.6.1"

__all__ = [
    "LLM",
    "AgentAction",
    "AgentRun",
    "AgentRunner",
    "Budget",
    "BudgetExceeded",
    "Evaluation",
    "GateOutcome",
    "HITLGate",
    "InstitutionalTicket",
    "LLMError",
    "MedicalProtocol",
    "NoPendingRequestError",
    "OpenAICompatLLM",
    "OpenAILLM",
    "QuotaExhausted",
    "RequestMismatchError",
    "Risk",
    "ScriptedLLM",
    "StructuredOutputError",
    "ThreadBusyError",
    "ToolRegistry",
    "ToolSpec",
    "TracedLLM",
    "Tracer",
    "TransformersLLM",
    "build_llm",
    "build_reflection_graph",
    "run_reflection",
    "sqlite_checkpointer",
    "structured_with_retry",
]
