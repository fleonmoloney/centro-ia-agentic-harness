"""Integraciones: la puerta HITL manda también dentro de BeeAI y AG2."""

import asyncio
import json

import pytest

from centro_harness.hitl import HITLGate, Risk, ToolRegistry, ToolSpec

CI = "1.234.567-2"


def make_gate():
    executed = []
    registry = ToolRegistry()

    def consultar(**kw):
        executed.append(("consultar", kw))
        return f"ficha de {kw.get('citizen_id', '?')}: en revisión"

    def borrar(**kw):
        executed.append(("borrar", kw))
        return "borrado"

    registry.register(ToolSpec(name="consultar", risk=Risk.LOW, description="Consulta"), consultar)
    registry.register(ToolSpec(name="borrar", risk=Risk.HIGH, description="Borra"), borrar)
    gate = HITLGate(registry, authorized_reviewers={"ana"})
    return gate, executed


# ---------------------------------------------------------------------------------- BeeAI
def _run_tool(tool, payload):
    async def go():
        return await tool.run(payload)  # `Run` de BeeAI es esperable, no una corrutina

    return asyncio.run(go())


def test_beeai_low_risk_tool_runs_and_masks_the_ci():
    pytest.importorskip("beeai_framework")
    from centro_harness.integrations.beeai import gated_tools

    gate, executed = make_gate()
    tools = {t.name: t for t in gated_tools(gate, thread_id="b1")}
    assert set(tools) == {"consultar", "borrar"}
    out = _run_tool(tools["consultar"], {"arguments": {"citizen_id": CI}})
    assert executed == [("consultar", {"citizen_id": CI})]
    assert CI not in out.get_text_content()
    assert "en revisión" in out.get_text_content()


def test_beeai_high_risk_tool_is_pending_and_only_runs_after_approval():
    pytest.importorskip("beeai_framework")
    from centro_harness.integrations.beeai import gated_tools

    gate, executed = make_gate()
    borrar = {t.name: t for t in gated_tools(gate, thread_id="b2")}["borrar"]
    out = _run_tool(borrar, {"arguments": {"id": 7}})
    assert "PENDIENTE DE APROBACIÓN" in out.get_text_content()
    assert executed == []
    pending = gate.pending("b2")
    gate.resume(thread_id="b2", request_id=pending["request_id"], approved=True, reviewer="ana")
    assert executed == [("borrar", {"id": 7})]


# ------------------------------------------------------------------------------------ AG2
def _ag2_run(gate, turns, thread_id):
    from ag2.testing import TestConfig

    from centro_harness.integrations.ag2 import build_agent

    agent = build_agent(
        gate, name="tramites", prompt="Atendés trámites", config=TestConfig(*turns),
        thread_id=thread_id,
    )
    return asyncio.run(agent.ask("hola"))


def _call(name, **arguments):
    from ag2.events import ToolCallEvent

    return ToolCallEvent(name=name, arguments=json.dumps({"arguments": arguments}))


def test_ag2_agent_high_risk_call_does_not_execute_without_approval():
    pytest.importorskip("ag2")
    gate, executed = make_gate()
    reply = _ag2_run(gate, [_call("borrar", id=7), "queda pendiente"], "a1")
    assert reply.body == "queda pendiente"
    assert executed == []
    assert gate.pending("a1")["tool"] == "borrar"


def test_ag2_agent_low_risk_call_runs_through_the_gate_and_is_audited():
    pytest.importorskip("ag2")
    gate, executed = make_gate()
    _ag2_run(gate, [_call("consultar", citizen_id=CI), "ok"], "a2")
    assert executed == [("consultar", {"citizen_id": CI})]
    assert [r.decision for r in gate.audit] == ["auto_approved"]
    assert gate.audit[0].arguments == {"citizen_id": "***"}


def test_ag2_agent_unknown_tool_name_cannot_bypass_the_registry():
    pytest.importorskip("ag2")
    gate, executed = make_gate()
    # el modelo inventa una herramienta: AG2 la rechaza antes de llegar a ejecutar nada
    try:
        _ag2_run(gate, [_call("formatear_disco"), "no pude"], "a3")
    except Exception:
        pass
    assert executed == []


# -------------------------------------------------------------------------------- CrewAI
def _crew(gate, texts, thread_id, budget=None):
    from crewai import Agent, Crew, Task

    from centro_harness.integrations.crewai import HarnessCrewLLM, gated_crew_tools
    from centro_harness.llm import ScriptedLLM
    from centro_harness.tracing import TracedLLM, Tracer

    llm = ScriptedLLM(texts=texts)
    if budget:
        llm = TracedLLM(llm, Tracer(), budget)
    agent = Agent(
        role="Gestor", goal="Atender trámites", backstory="Funcionario",
        llm=HarnessCrewLLM(llm), tools=gated_crew_tools(gate, thread_id=thread_id),
        allow_delegation=False, verbose=False, max_iter=4,
    )
    task = Task(description="Atender el pedido", expected_output="Respuesta", agent=agent)
    return Crew(agents=[agent], tasks=[task], verbose=False).kickoff()


def _react(tool, args):
    return (
        f"Thought: Necesito usar la herramienta\nAction: {tool}\n"
        f"Action Input: {json.dumps({'arguments': args})}"
    )


FINAL = "Thought: I now know the final answer\nFinal Answer: listo"


def test_crewai_high_risk_call_stays_pending():
    pytest.importorskip("crewai")
    gate, executed = make_gate()
    result = _crew(gate, [_react("borrar", {"id": 7}), FINAL], "c1")
    assert "listo" in str(result)
    assert executed == []
    assert gate.pending("c1")["tool"] == "borrar"


def test_crewai_low_risk_call_runs_through_the_gate():
    pytest.importorskip("crewai")
    gate, executed = make_gate()
    _crew(gate, [_react("consultar", {"citizen_id": CI}), FINAL], "c2")
    assert executed == [("consultar", {"citizen_id": CI})]
    assert gate.audit[0].arguments == {"citizen_id": "***"}


def test_crewai_budget_of_the_harness_cuts_the_crew():
    pytest.importorskip("crewai")
    from centro_harness.budget import Budget, BudgetExceeded

    gate, executed = make_gate()
    with pytest.raises(BudgetExceeded):
        _crew(gate, [_react("consultar", {"n": 1})], "c3", budget=Budget(max_calls=1))
