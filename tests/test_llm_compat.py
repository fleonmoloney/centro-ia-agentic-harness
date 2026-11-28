"""OpenAICompatLLM contra un servidor HTTP real (en un hilo), no contra mocks de urllib."""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from centro_harness.llm import LLMError, OpenAICompatLLM
from centro_harness.schemas import InstitutionalTicket, structured_with_retry
from centro_harness.tracing import TracedLLM, Tracer

TICKET = {
    "citizen_id": "1.234.567-2",
    "issue_category": "empleo",
    "urgency_level": "media",
    "recommended_action": "Orientar a DINAE",
}


def chat(content, prompt_tokens=11, completion_tokens=7):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - nombre fijado por http.server
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        srv = self.server
        srv.requests.append(
            {
                "path": self.path,
                "body": body,
                "auth": self.headers.get("Authorization"),
                "agent": self.headers.get("User-Agent"),
            }
        )
        status, payload = srv.script.pop(0) if len(srv.script) > 1 else srv.script[0]
        raw = payload if isinstance(payload, str) else json.dumps(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(raw.encode("utf-8"))

    def log_message(self, *args):  # silencio
        pass


@pytest.fixture
def serve():
    servers = []

    def _start(script):
        server = HTTPServer(("127.0.0.1", 0), _Handler)
        server.script = list(script)
        server.requests = []
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_port}/v1", server

    yield _start
    for s in servers:
        s.shutdown()
        s.server_close()


def _llm(url, **kw):
    kw.setdefault("backoff", 0.0)
    return OpenAICompatLLM(base_url=url, model="modelo-x", **kw)


def test_generate_sends_the_request_and_records_real_usage(serve):
    url, srv = serve([(200, chat("hola"))])
    llm = _llm(url, temperature=0.0, max_tokens=64, system_prompt="sé breve")
    assert llm.generate("saludá") == "hola"
    req = srv.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["body"]["model"] == "modelo-x"
    assert req["body"]["max_tokens"] == 64
    assert req["body"]["messages"][0] == {"role": "system", "content": "sé breve"}
    assert req["body"]["messages"][-1] == {"role": "user", "content": "saludá"}
    assert "response_format" not in req["body"]
    assert llm.last_usage == (11, 7)


def test_api_key_is_sent_as_bearer(serve):
    url, srv = serve([(200, chat("ok"))])
    _llm(url, api_key="secreto").generate("x")
    assert srv.requests[0]["auth"] == "Bearer secreto"


def test_structured_extracts_json_from_fenced_prose(serve):
    reply = "Claro, acá va:\n```json\n" + json.dumps(TICKET) + "\n```\nListo."
    url, srv = serve([(200, chat(reply))])
    ticket = _llm(url).structured("derivá", InstitutionalTicket)
    assert ticket.citizen_id == "1.234.567-2"
    assert "esquema JSON" in srv.requests[0]["body"]["messages"][-1]["content"]


def test_json_mode_is_requested_only_for_structured(serve):
    url, srv = serve([(200, chat(json.dumps(TICKET)))])
    llm = _llm(url, json_mode=True)
    llm.structured("derivá", InstitutionalTicket)
    assert srv.requests[0]["body"]["response_format"] == {"type": "json_object"}


def test_transient_server_errors_are_retried(serve):
    url, srv = serve([(503, {"error": "ocupado"}), (429, {"error": "lento"}), (200, chat("ya"))])
    assert _llm(url, retries=2).generate("x") == "ya"
    assert len(srv.requests) == 3


def test_client_errors_are_not_retried(serve):
    url, srv = serve([(400, {"error": "modelo inexistente"})])
    with pytest.raises(LLMError, match="HTTP 400"):
        _llm(url, retries=3).generate("x")
    assert len(srv.requests) == 1


def test_retries_are_bounded(serve):
    url, srv = serve([(500, {"error": "roto"})])
    with pytest.raises(LLMError, match="HTTP 500"):
        _llm(url, retries=2).generate("x")
    assert len(srv.requests) == 3


def test_unreachable_server_raises_llmerror():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(LLMError, match="No se pudo conectar"):
        _llm(f"http://127.0.0.1:{port}/v1", retries=0, timeout=2).generate("x")


def test_malformed_responses_raise_llmerror(serve):
    url, _ = serve([(200, "esto no es json")])
    with pytest.raises(LLMError, match="no es JSON"):
        _llm(url).generate("x")
    url2, _ = serve([(200, {"sin": "choices"})])
    with pytest.raises(LLMError, match="choices"):
        _llm(url2).generate("x")


def test_model_name_is_required(monkeypatch):
    monkeypatch.delenv("HARNESS_MODEL", raising=False)
    with pytest.raises(ValueError):
        OpenAICompatLLM(base_url="http://x/v1")


def test_repair_loop_works_end_to_end_and_trace_uses_provider_tokens(serve):
    url, srv = serve([(200, chat("no soy json", 20, 3)), (200, chat(json.dumps(TICKET), 25, 30))])
    tracer = Tracer()
    llm = TracedLLM(_llm(url), tracer)
    ticket = structured_with_retry(llm, "derivá", InstitutionalTicket)
    assert ticket.issue_category == "empleo"
    assert len(srv.requests) == 2
    assert "no validó" in srv.requests[1]["body"]["messages"][-1]["content"]
    assert tracer.calls == 2 and tracer.errors == 1
    assert (tracer.tokens_in, tracer.tokens_out) == (45, 33)
    assert tracer.tokens_source == "provider"


def test_infrastructure_failure_is_not_swallowed_by_the_repair_loop(serve):
    url, srv = serve([(500, {"error": "roto"})])
    with pytest.raises(LLMError):
        structured_with_retry(_llm(url, retries=0), "x", InstitutionalTicket)
    assert len(srv.requests) == 1


def test_failed_structured_case_reports_why_and_what_the_model_said():
    from centro_harness.evals import run_case

    case = {
        "id": "x", "type": "structured", "schema": "InstitutionalTicket", "live": True,
        "prompt": "Consulta", "expect": {"ok": True},
    }

    class Charlatan:
        last_text = None

        def generate(self, prompt, *, label=""):
            return "Claro, acá tenés la respuesta"

        def structured(self, prompt, schema, *, label=""):
            from centro_harness.llm import structured_from_text

            return structured_from_text(
                self.generate, prompt, schema, record=lambda t: setattr(self, "last_text", t)
            )

    res = run_case(case, mode="live", shared_llm=Charlatan())
    assert res.status == "fail"
    assert "Claro, acá tenés" in res.actual["last_output"]
    assert res.actual["last_error"]


def test_sends_its_own_user_agent_not_the_urllib_default(serve):
    # Groq (Cloudflare) devuelve 403 "error code: 1010" al User-Agent "Python-urllib/x".
    url, srv = serve([(200, chat("hola"))])
    _llm(url).generate("saludá")
    agent = srv.requests[0]["agent"]
    assert agent.startswith("centro-ia-agentic-harness")
    assert "urllib" not in agent
