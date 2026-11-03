"""Mini-harness de evals: casos dorados en JSONL, ejecución, reporte y código de salida.

Dos modos:
* scripted (default): corre los casos deterministas con `ScriptedLLM`. Valida la lógica de
  control del harness (reintentos, ruteo, tope de iteraciones, HITL, reglas de negocio,
  modelo de costos). No mide la calidad de ningún modelo.
* openai: corre solo los casos marcados `"live": true` contra un modelo real
  (requiere `pip install '.[openai]'` y OPENAI_API_KEY).

Uso: python -m centro_harness.evals --cases evals/cases.jsonl [--out-md reports/latest.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from centro_harness.cost_model import TaskSpec, ToolDoc, compare_tool_assignment
from centro_harness.hitl import HITLGate, Risk, ToolRegistry, ToolSpec
from centro_harness.llm import LLM, ScriptedLLM, build_llm
from centro_harness.reflection import run_reflection
from centro_harness.schemas import (
    InstitutionalTicket,
    MedicalProtocol,
    StructuredOutputError,
    structured_with_retry,
)
from centro_harness.tracing import TracedLLM, Tracer

SCHEMAS = {"InstitutionalTicket": InstitutionalTicket, "MedicalProtocol": MedicalProtocol}


@dataclass
class CaseResult:
    id: str
    type: str
    status: str  # "pass" | "fail" | "skipped"
    detail: str = ""
    tokens_in_est: int = 0
    tokens_out_est: int = 0
    latency_ms: float = 0.0
    mismatches: list[str] = field(default_factory=list)


def load_cases(path: str | Path) -> list[dict]:
    cases = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: JSON inválido ({exc})") from exc
    return cases


def _compare(expect: dict, actual: dict) -> list[str]:
    """Claves `min_x` exigen actual['x'] >= valor; el resto, igualdad exacta."""
    problems = []
    for key, want in expect.items():
        if key.startswith("min_"):
            got = actual.get(key[4:])
            if got is None or got < want:
                problems.append(f"{key[4:]}: esperado >= {want}, obtenido {got}")
        else:
            got = actual.get(key)
            if got != want:
                problems.append(f"{key}: esperado {want!r}, obtenido {got!r}")
    return problems


# --------------------------------------------------------------------------- tipos de caso
def _run_structured(case: dict, shared_llm: LLM | None) -> tuple[dict, Tracer]:
    tracer = Tracer()
    inner = shared_llm or ScriptedLLM(structured=case["scripted"])
    llm = TracedLLM(inner, tracer)
    schema = SCHEMAS[case["schema"]]
    try:
        obj = structured_with_retry(
            llm, case["prompt"], schema, max_attempts=case.get("max_attempts", 3)
        )
        actual = {"ok": True, **obj.model_dump()}
    except StructuredOutputError:
        actual = {"ok": False}
    actual["attempts"] = tracer.calls
    return actual, tracer


def _run_reflection(case: dict) -> tuple[dict, Tracer]:
    tracer = Tracer()
    scores = case["scores"]
    llm = TracedLLM(
        ScriptedLLM(
            texts=[f"borrador {i}" for i in range(1, case.get("max_iterations", 3) + 1)],
            structured=[{"score": s, "missing": "detalle", "superfluous": ""} for s in scores],
        ),
        tracer,
    )
    final = run_reflection(
        llm,
        case["task"],
        threshold=case.get("threshold", 0.85),
        max_iterations=case.get("max_iterations", 3),
    )
    actual = {
        "status": final["status"],
        "iterations": final["iteration"],
        "final_score": final["score"],
    }
    return actual, tracer


def _run_hitl(case: dict) -> tuple[dict, Tracer]:
    registry = ToolRegistry()
    for tool in case["tools"]:
        registry.register(
            ToolSpec(name=tool["name"], risk=Risk(tool["risk"])),
            lambda _n=tool["name"], **_kw: f"ok:{_n}",
        )
    gate = HITLGate(registry)
    call = case["call"]
    outcome = gate.submit(call["tool"], call.get("arguments", {}), thread_id=case["id"])
    paused = outcome.status == "pending_approval"
    review = case.get("review")
    if paused and review is not None:
        outcome = gate.resume(
            thread_id=case["id"], approved=review["approved"], reviewer=review.get("reviewer")
        )
    actual = {
        "paused": paused,
        "status": outcome.status,
        "decision": outcome.record.decision if outcome.record else None,
        "audit_entries": len(gate.audit),
    }
    return actual, Tracer()


def _run_cost(case: dict) -> tuple[dict, Tracer]:
    tools = [
        ToolDoc(f"tool_{i}", f"Herramienta número {i} para consultar la fuente {i}. " * 3)
        for i in range(case["tools"])
    ]
    n_tools = case["tools"]
    tasks = [
        TaskSpec(f"Resolver la consulta {j} usando la fuente correcta.", f"tool_{j % n_tools}")
        for j in range(case["tasks"])
    ]
    cmp = compare_tool_assignment(tools, tasks)
    actual = {
        "savings_pct": cmp.savings_pct,
        "agent_centric_tokens": cmp.agent_centric_tokens,
        "task_centric_tokens": cmp.task_centric_tokens,
    }
    return actual, Tracer()


def run_case(case: dict, *, mode: str, shared_llm: LLM | None) -> CaseResult:
    live = bool(case.get("live"))
    if (mode == "scripted") == live:
        why = "requiere --llm openai" if live else "caso determinista (solo modo scripted)"
        return CaseResult(case["id"], case["type"], "skipped", why)

    kind = case["type"]
    if kind == "structured":
        actual, tracer = _run_structured(case, shared_llm if live else None)
    elif kind == "reflection":
        actual, tracer = _run_reflection(case)
    elif kind == "hitl":
        actual, tracer = _run_hitl(case)
    elif kind == "cost":
        actual, tracer = _run_cost(case)
    else:
        return CaseResult(case["id"], kind, "fail", f"tipo de caso desconocido: {kind}")

    problems = _compare(case["expect"], actual)
    return CaseResult(
        id=case["id"],
        type=kind,
        status="fail" if problems else "pass",
        detail=json.dumps(actual, ensure_ascii=False, default=str),
        tokens_in_est=tracer.tokens_in,
        tokens_out_est=tracer.tokens_out,
        latency_ms=round(tracer.latency_ms, 3),
        mismatches=problems,
    )


# --------------------------------------------------------------------------- reporte
def summarize(results: list[CaseResult]) -> dict:
    ran = [r for r in results if r.status != "skipped"]
    passed = [r for r in ran if r.status == "pass"]
    by_type: dict[str, dict] = {}
    for r in ran:
        slot = by_type.setdefault(r.type, {"total": 0, "pass": 0})
        slot["total"] += 1
        slot["pass"] += r.status == "pass"
    return {
        "total": len(results),
        "ran": len(ran),
        "skipped": len(results) - len(ran),
        "passed": len(passed),
        "failed": len(ran) - len(passed),
        "pass_rate": round(len(passed) / len(ran), 3) if ran else None,
        "by_type": by_type,
        "tokens_in_est": sum(r.tokens_in_est for r in ran),
        "tokens_out_est": sum(r.tokens_out_est for r in ran),
    }


def to_markdown(results: list[CaseResult], summary: dict, mode: str) -> str:
    lines = [
        "# Reporte de evals",
        "",
        f"Modo: `{mode}`. Casos ejecutados: {summary['ran']} de {summary['total']} "
        f"(omitidos: {summary['skipped']}). Aprobados: {summary['passed']}. "
        f"Fallidos: {summary['failed']}.",
        "",
        "| Caso | Tipo | Resultado | Tokens est. (in/out) | Notas |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        if r.mismatches:
            note = "; ".join(r.mismatches)
        else:
            note = r.detail if r.status == "skipped" else ""
        lines.append(
            f"| {r.id} | {r.type} | {r.status} | {r.tokens_in_est}/{r.tokens_out_est} | {note} |"
        )
    lines += [
        "",
        "Los tokens son estimaciones (~4 caracteres/token), no conteo de proveedor. En modo "
        "`scripted` el modelo es un guion: este reporte valida la lógica del harness, no la "
        "calidad de un LLM.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Corre los evals del harness.")
    parser.add_argument("--cases", default="evals/cases.jsonl")
    parser.add_argument("--llm", choices=["scripted", "openai"], default="scripted")
    parser.add_argument("--out-json")
    parser.add_argument("--out-md")
    args = parser.parse_args(argv)

    shared = build_llm("openai") if args.llm == "openai" else None
    cases = load_cases(args.cases)
    results = [run_case(c, mode=args.llm, shared_llm=shared) for c in cases]
    summary = summarize(results)
    markdown = to_markdown(results, summary, args.llm)

    print(markdown)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(
                {"summary": summary, "results": [asdict(r) for r in results]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(markdown, encoding="utf-8")

    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
