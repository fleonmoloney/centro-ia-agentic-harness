"""Compara reportes JSON de evals (uno por modelo) en una tabla Markdown lista para el README.

Uso: harness-compare reports/qwen-1.5b.json reports/qwen-3b.json [--out-md reports/comparacion.md]

Reglas para no engañar:
* Solo se comparan reportes en modo live y con los MISMOS casos y repeticiones; si no coinciden,
  se avisa en la tabla (una diferencia de repeticiones cambia el ancho del intervalo).
* Cada celda de tasa lleva su intervalo de Wilson al 95 %. Si dos intervalos se solapan, la
  diferencia NO es evidencia de que un modelo sea mejor que otro; la tabla lo marca.
* Las violaciones de invariante (puerta HITL) y los errores de infraestructura se muestran
  aparte: no son "mala calidad del modelo".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("meta", {}).get("llm") == "scripted":
        raise ValueError(f"{path}: es un reporte scripted; solo se comparan reportes live")
    return data


def _cell(block: dict | None) -> str:
    if not block or block.get("rate") is None:
        return "-"
    lo, hi = block["ci95"]
    return f"{block['rate']:.2f} ({lo:.2f}–{hi:.2f}) n={block['total']}"


def _name(meta: dict, fallback: str) -> str:
    base = meta.get("label") or meta.get("model") or fallback
    quant = f" {meta['quantization']}" if meta.get("quantization") else ""
    return f"{base}{quant}"


def _overlap(a: list[float] | None, b: list[float] | None) -> bool:
    return bool(a and b and a[0] <= b[1] and b[0] <= a[1])


def compare(paths: list[str]) -> str:
    reports = [(p, _load(p)) for p in paths]
    names = [_name(r["meta"], Path(p).stem) for p, r in reports]
    types = sorted({t for _, r in reports for t in r["summary"]["by_type"]})

    header = ["Modelo", "Éxito global", *[f"Éxito {t}" for t in types]]
    header += ["Reparación JSON", "Llamadas/ejec.", "Tokens in/out por ejec."]
    header += ["Latencia p50/p95 (s)"]
    rows = []
    for name, (_, r) in zip(names, reports, strict=True):
        s = r["summary"]
        ran = max(1, s["ran"])
        repair = (s.get("structured") or {}).get("repair_rate")
        p50, p95 = s.get("run_latency_ms_p50"), s.get("run_latency_ms_p95")
        row = [name, _cell({"rate": s["pass_rate"], "ci95": s["ci95"], "total": s["ran"]})]
        row += [_cell(s["by_type"].get(t)) for t in types]
        row += [
            "-" if repair is None else f"{repair:.2f}",
            f"{s['llm_calls'] / ran:.1f}",
            f"{s['tokens_in'] // ran}/{s['tokens_out'] // ran} ({s['tokens_source']})",
            "-" if p50 is None else f"{p50 / 1000:.1f}/{p95 / 1000:.1f}",
        ]
        rows.append(row)

    out = ["# Comparación de modelos", "", "| " + " | ".join(header) + " |"]
    out.append("|" + "---|" * len(header))
    out += ["| " + " | ".join(row) + " |" for row in rows]

    notes = []
    case_sets = {tuple(sorted(r["summary"]["per_case"])) for _, r in reports}
    repeats = {r["meta"].get("repeats") for _, r in reports}
    if len(case_sets) > 1:
        notes.append("Los reportes NO cubren los mismos casos: la comparación no es directa.")
    if len(repeats) > 1:
        notes.append(f"Repeticiones distintas entre reportes ({sorted(map(str, repeats))}).")
    for name, (_, r) in zip(names, reports, strict=True):
        s = r["summary"]
        if s["invariant_violations"]:
            notes.append(f"**{name}: {s['invariant_violations']} violación(es) de invariante.**")
        if s["errors"]:
            notes.append(
                f"{name}: {s['errors']} ejecución(es) con error de infraestructura "
                "(excluidas de las tasas)."
            )
    for i in range(len(reports)):
        for j in range(i + 1, len(reports)):
            si, sj = reports[i][1]["summary"], reports[j][1]["summary"]
            ri, rj = si["pass_rate"], sj["pass_rate"]
            if ri is not None and rj is not None and ri != rj:
                verdict = (
                    "los intervalos se solapan: NO hay evidencia de diferencia"
                    if _overlap(si["ci95"], sj["ci95"])
                    else "los intervalos no se solapan"
                )
                notes.append(f"{names[i]} vs {names[j]}: {verdict}.")
    if notes:
        out += ["", "Notas:", *[f"- {n}" for n in notes]]
    out += [
        "",
        "Los intervalos son de Wilson al 95 %. Cada modelo se midió con decodificación voraz; "
        "una repetición idéntica produce casi siempre el mismo resultado, así que las "
        "repeticiones miden poco la variabilidad entre corridas (sí miden la de los casos).",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compara reportes JSON de evals live.")
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--out-md")
    args = parser.parse_args(argv)
    try:
        text = compare(args.reports)
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(text)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
