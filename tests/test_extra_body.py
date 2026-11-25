"""--extra-body y --system-prompt: llegan al cuerpo de la petición y quedan en el reporte."""

from __future__ import annotations

import argparse

import pytest

from centro_harness import build_llm
from centro_harness.evals import _json_object
from centro_harness.llm import OpenAICompatLLM


def _capture(llm):
    seen = []
    llm._post = lambda payload: (
        seen.append(payload) or {"choices": [{"message": {"content": "ok"}}]}
    )
    return seen


def test_extra_body_and_system_prompt_reach_the_request():
    llm = build_llm(
        "compat", base_url="http://x/v1", model="m",
        extra_body={"reasoning_effort": "low"}, system_prompt="sé breve",
    )
    seen = _capture(llm)
    llm.generate("hola")
    assert seen[0]["reasoning_effort"] == "low"
    assert seen[0]["messages"][0] == {"role": "system", "content": "sé breve"}
    assert seen[0]["model"] == "m"


def test_without_extra_body_the_payload_is_unchanged():
    llm = OpenAICompatLLM("http://x/v1", "m")
    seen = _capture(llm)
    llm.generate("hola")
    assert set(seen[0]) == {"model", "messages", "temperature", "max_tokens"}


@pytest.mark.parametrize("bad", ['{"model": "otro"}', '{"messages": []}', "[1]", "no json"])
def test_extra_body_cannot_override_model_or_messages(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        _json_object(bad)


def test_class_rejects_reserved_keys():
    with pytest.raises(ValueError):
        OpenAICompatLLM("http://x/v1", "m", extra_body={"model": "otro"})
