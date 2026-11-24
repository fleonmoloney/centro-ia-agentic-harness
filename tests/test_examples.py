"""Los ejemplos de `examples/` salen en el README: tienen que correr y decir la verdad."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"examples_{name}", EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hitl_demo_decisions_and_output(capsys):
    decisions = _load("hitl_demo").main()
    out = capsys.readouterr().out
    assert decisions == ["auto_approved", "rejected_unauthorized", "approved"]
    # La acción de riesgo alto no se ejecutó antes de la aprobación ni con el revisor intruso.
    assert "registros borrados hasta ahora: []" in out
    assert "decisión=rejected_unauthorized  borrados=[]" in out
    assert "   borrados=[7]" in out
    # La cédula nunca aparece en claro en lo que ve la persona ni en la auditoría.
    assert "1.234.567-2" not in out
    assert "'citizen_id': '***'" in out


def test_agent_demo_approved(capsys):
    run = _load("agent_demo").main([])
    out = capsys.readouterr().out
    assert run.status == "done"
    assert run.executed_tools == ["consultar", "notificar"]
    assert "Avisos realmente enviados: ['Su trámite 4521 fue aprobado']" in out
    assert "Llamadas al modelo: 3" in out


def test_agent_demo_rejected_does_not_send_and_does_not_claim_it_did(capsys):
    run = _load("agent_demo").main(["--reject"])
    out = capsys.readouterr().out
    assert run.executed_tools == ["consultar"]
    assert "Avisos realmente enviados: []" in out
    assert "NO fue notificado" in run.answer


def test_agent_demo_run_survives_a_json_roundtrip(capsys):
    # El ejemplo restaura la corrida desde JSON antes de reanudar: si eso fallara, no llegaría
    # al estado final.
    run = _load("agent_demo").main([])
    capsys.readouterr()
    assert run.pending is None and run.pending_thread is None


@pytest.mark.parametrize(
    "script, args",
    [("hitl_demo.py", []), ("agent_demo.py", []), ("agent_demo.py", ["--reject"])],
)
def test_examples_run_as_scripts_the_way_the_readme_says(script, args):
    """`python examples/<script>` desde la raíz del repo, sin variables ni claves."""
    done = subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip()


# ----------------------------------------------------------- real_model_demo (cableado)
def _mock_server():
    import threading

    from centro_harness.mockmodel import make_server

    server = make_server(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/v1"


@pytest.mark.parametrize(
    "approve, sent, executed",
    [("yes", True, "['enviar_notificacion']"), ("no", False, "[]")],
)
def test_real_model_demo_harness_against_the_mock_server(capsys, approve, sent, executed):
    server, url = _mock_server()
    try:
        code = _load("real_model_demo").main(
            ["--base-url", url, "--model", "mock", "--approve", approve]
        )
    finally:
        server.shutdown()
    out = capsys.readouterr().out
    assert code == 0
    assert f"Herramientas ejecutadas: {executed}" in out
    assert ("'destino': '1.234.567-2'" in out) is sent  # solo se envía si se aprobó
    assert "*.***.567-2" in out  # lo que ve la persona va enmascarado


def test_real_model_demo_distinguishes_infrastructure_failure_from_model_failure(capsys):
    code = _load("real_model_demo").main(
        ["--base-url", "http://127.0.0.1:1/v1", "--model", "x", "--approve", "no"]
    )
    out = capsys.readouterr().out
    assert code == 2
    assert "no es un error del modelo" in out


def test_real_model_demo_crewai_leaves_the_action_pending_until_a_person_decides(
    capsys, monkeypatch
):
    pytest.importorskip("crewai")
    import json

    from centro_harness import Budget, ScriptedLLM, TracedLLM, Tracer

    demo = _load("real_model_demo")
    react = (
        "Thought: Debo avisar\nAction: enviar_notificacion\nAction Input: "
        + json.dumps({"arguments": {"destino": "1.234.567-2", "texto": "Aprobado"}})
    )
    final = "Thought: I now know the final answer\nFinal Answer: pendiente de revisión"
    tracer = Tracer()
    scripted = TracedLLM(ScriptedLLM(texts=[react, final]), tracer, Budget(max_calls=10))
    monkeypatch.setattr(demo, "build_traced_llm", lambda args: (scripted, tracer))

    code = demo.main(["--framework", "crewai", "--approve", "no", "--model", "x"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Avisos realmente enviados: []" in out  # el equipo terminó y nada se envió
    assert "('enviar_notificacion', 'rejected')" in out
