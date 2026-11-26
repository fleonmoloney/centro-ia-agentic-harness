"""`--timeout`: un endpoint lento no debe terminar en errores de infraestructura por defecto."""

import argparse

import pytest

from centro_harness.evals import _positive_float
from centro_harness.llm import build_llm


def test_timeout_reaches_the_compat_client():
    llm = build_llm("compat", base_url="http://x/v1", model="m", timeout=600.0)
    assert llm.timeout == 600.0


def test_default_timeout_is_unchanged():
    assert build_llm("compat", base_url="http://x/v1", model="m").timeout == 120.0


@pytest.mark.parametrize("bad", ["0", "-5", "abc"])
def test_timeout_flag_rejects_non_positive_or_non_numeric(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_float(bad)


def test_timeout_flag_accepts_seconds():
    assert _positive_float("300") == 300.0
