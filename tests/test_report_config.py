"""El reporte en texto declara las opciones que se apartan del protocolo estándar."""

from centro_harness import evals
from centro_harness.evals import CaseResult, summarize, to_markdown


def test_default_run_has_no_configuration_line():
    assert evals._config_line({"ticket_schema": "neutral", "prompt_style": "schema"}) == ""


def test_config_line_names_every_non_default_option():
    line = evals._config_line(
        {"ticket_schema": "standard", "extra_body": {"reasoning_effort": "low"},
         "system_prompt": "x", "timeout": 600.0, "prompt_style": "example"}
    )
    for expected in ("standard", "reasoning_effort", "system prompt", "600 s", "example"):
        assert expected in line


def test_standard_variant_shows_up_in_the_markdown_of_a_live_report():
    results = [CaseResult("s", "structured", "pass", actual={"ok": True}, expect={}, live=True)]
    meta = {"harness_version": "x", "protocol_version": 6, "ticket_schema": "standard"}
    text = to_markdown(results, summarize(results), "live", meta)
    assert "Configuración: variante de esquema `standard`" in text
    neutral = {**meta, "ticket_schema": "neutral"}
    assert "Configuración:" not in to_markdown(results, summarize(results), "live", neutral)


def test_report_without_the_key_is_read_as_standard():
    assert "`standard`" in evals._config_line({"prompt_style": "schema"})
