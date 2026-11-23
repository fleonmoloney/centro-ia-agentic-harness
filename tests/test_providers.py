"""Comportamientos que exigen los proveedores gratuitos: modelos de razonamiento, límites de
RPM, cuota diaria agotada y corte ordenado con reporte parcial."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import centro_harness.llm as llm_module
from centro_harness.evals import main
from centro_harness.llm import LLMError, OpenAICompatLLM, QuotaExhausted, strip_reasoning
from centro_harness.mockmodel import make_server


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        srv = self.server
        srv.hits += 1
        status, payload, headers = srv.script.pop(0) if len(srv.script) > 1 else srv.script[0]
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, *args):
        pass


@pytest.fixture
def serve():
    servers = []

    def _start(script):
        server = HTTPServer(("127.0.0.1", 0), _Handler)
        server.script = [(s, p, h) for s, p, h in script]
        server.hits = 0
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_port}/v1", server

    yield _start
    for s in servers:
        s.shutdown()
        s.server_close()


def ok(content, finish="stop", **extra):
    message = {"content": content, **extra}
    return (200, {"choices": [{"message": message, "finish_reason": finish}]}, {})


def _llm(url, **kw):
    kw.setdefault("backoff", 0.0)
    return OpenAICompatLLM(base_url=url, model="m", **kw)


# ------------------------------------------------------------------ modelos de razonamiento
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<think>pienso mucho {x}</think>\n{\"a\": 1}", '{"a": 1}'),
        ("pensamiento sin etiqueta de apertura</think> respuesta", "respuesta"),
        ("<think>me quedé pensando y se cortó", ""),
        ("sin etiquetas", "sin etiquetas"),
        ("<think>a</think>uno<think>b</think>dos", "dos"),
    ],
)
def test_strip_reasoning(raw, expected):
    assert strip_reasoning(raw) == expected


def test_reasoning_block_is_removed_before_parsing_json(serve):
    # El razonamiento contiene llaves: sin quitarlo, extract_json podría tomar el objeto equivocado.
    content = '<think>Probaría {"x": 1} pero no.</think>{"citizen_id": "1.234.567-2"}'
    url, _ = serve([ok(content)])
    assert _llm(url).generate("x") == '{"citizen_id": "1.234.567-2"}'


def test_separate_reasoning_field_is_never_used_as_the_answer(serve):
    url, _ = serve([ok("respuesta", reasoning_content="pensamiento", reasoning="otro")])
    assert _llm(url).generate("x") == "respuesta"


def test_null_content_with_reasoning_only_is_not_treated_as_an_answer(serve):
    url, _ = serve([ok(None, finish="stop", reasoning_content="solo pensó")])
    assert _llm(url).generate("x") == ""


def test_empty_answer_caused_by_length_is_a_configuration_error(serve):
    url, _ = serve([ok("<think>pensando sin parar", finish="length")])
    with pytest.raises(LLMError, match="max-tokens"):
        _llm(url, max_tokens=512).generate("x")


# ------------------------------------------------------------------ límites de velocidad
@pytest.fixture
def fake_clock(monkeypatch):
    state = {"now": 1000.0, "sleeps": []}
    monkeypatch.setattr(llm_module.time, "monotonic", lambda: state["now"])
    monkeypatch.setattr(llm_module.time, "time", lambda: state["now"])

    def sleep(seconds):
        state["sleeps"].append(seconds)
        state["now"] += seconds

    monkeypatch.setattr(llm_module.time, "sleep", sleep)
    return state


def test_min_interval_spaces_requests(serve, fake_clock):
    url, _ = serve([ok("a")])
    llm = _llm(url, min_interval=3.0)
    llm.generate("1")
    llm.generate("2")
    llm.generate("3")
    assert fake_clock["sleeps"] == [3.0, 3.0]  # la primera no espera


def test_retry_after_is_honored_instead_of_blind_backoff(serve, fake_clock):
    url, srv = serve([(429, {"error": "lento"}, {"Retry-After": "7"}), ok("listo")])
    assert _llm(url, retries=2, backoff=0.5).generate("x") == "listo"
    assert 7.0 in fake_clock["sleeps"]
    assert srv.hits == 2


def test_wait_longer_than_the_limit_is_quota_exhaustion_not_a_retry(serve, fake_clock):
    url, srv = serve([(429, {"error": "diario"}, {"Retry-After": "3600"})])
    with pytest.raises(QuotaExhausted, match="3600"):
        _llm(url, retries=5, max_retry_wait=60).generate("x")
    assert srv.hits == 1  # no insistió contra un límite agotado


def test_rate_limit_reset_header_in_epoch_milliseconds(serve, fake_clock):
    reset_ms = str(int((fake_clock["now"] + 86400) * 1000))
    url, _ = serve([(429, {"error": "diario"}, {"X-RateLimit-Reset": reset_ms})])
    with pytest.raises(QuotaExhausted):
        _llm(url, max_retry_wait=60).generate("x")


def test_plain_429_without_headers_still_retries_with_backoff(serve, fake_clock):
    url, srv = serve([(429, {"error": "lento"}, {}), ok("ya")])
    assert _llm(url, retries=2, backoff=0.5).generate("x") == "ya"
    assert srv.hits == 2


# ------------------------------------------------------------------ corte con reporte parcial
LIVE = "evals/live_cases.jsonl"


def test_global_call_cap_stops_the_run_with_a_partial_report(tmp_path):
    server = make_server(0, "normal")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    out = tmp_path / "parcial.json"
    code = main(
        ["--cases", LIVE, "--llm", "compat", "--model", "mock", "--retries", "0",
         "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
         "--max-total-calls", "10", "--out-json", str(out), "--out-md", str(tmp_path / "p.md")]
    )  # fmt: skip
    server.shutdown()
    server.server_close()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert code == 3
    assert "Tope global de 10 llamadas" in data["summary"]["aborted"]
    assert 0 < data["summary"]["ran"] < 19  # algo se midió, pero no todo
    assert data["summary"]["llm_calls"] <= 10
    assert "MEDICIÓN INCOMPLETA" in (tmp_path / "p.md").read_text(encoding="utf-8")
    assert data["meta"]["max_total_calls"] == 10


def test_exhausted_provider_quota_also_ends_the_run_as_partial(serve, tmp_path):
    url, srv = serve([ok('{"x": 1}'), (429, {"error": "diario"}, {"Retry-After": "86400"})])
    out = tmp_path / "cuota.json"
    code = main(
        ["--cases", LIVE, "--llm", "compat", "--model", "m", "--base-url", url,
         "--retries", "0", "--out-json", str(out)]
    )  # fmt: skip
    data = json.loads(out.read_text(encoding="utf-8"))
    assert code == 3
    assert "86400" in data["summary"]["aborted"]
    assert srv.hits == 2  # tras agotarse la cuota no se siguió llamando


def test_max_tokens_reaches_the_transformers_backend(monkeypatch):
    captured = {}

    class Falso:
        def __init__(self, model_id="x", **kwargs):
            captured.update(model_id=model_id, **kwargs)

    monkeypatch.setattr(llm_module, "TransformersLLM", Falso)
    llm_module.build_llm("transformers", model="Qwen/Qwen2.5-3B-Instruct",
                         quantization="4bit", max_tokens=2048)  # fmt: skip
    assert captured == {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "quantization": "4bit",
        "max_new_tokens": 2048,
    }
