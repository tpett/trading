"""Unit tests for the stdlib OAuth + MCP client. Every test monkeypatches
trading.rhmcp._http -- nothing here touches the network."""

from __future__ import annotations

import json
import stat
import time

import pytest

from trading import rhmcp
from trading.rhmcp import McpClient, RhMcpError, Tokens, _parse_rpc_body


def make_tokens(tmp_path, expires_in=3600.0):
    path = tmp_path / "tokens.json"
    Tokens(
        client_id="cid",
        access_token="at-1",
        refresh_token="rt-1",
        expires_at=time.time() + expires_in,
    ).save(path)
    return path


def rpc_response(rpc_id, result):
    return json.dumps({"jsonrpc": "2.0", "id": rpc_id, "result": result}).encode()


def tool_text_result(payload):
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}


class FakeHttp:
    """Scripted _http replacement: pops the next (status, headers, body)
    canned response and records every request for assertions."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, url, data=None, headers=None):
        self.requests.append({"url": url, "data": data, "headers": headers or {}})
        if not self.responses:
            raise AssertionError(f"unexpected extra request to {url}")
        return self.responses.pop(0)


def test_tokens_save_load_roundtrip_with_owner_only_permissions(tmp_path):
    path = make_tokens(tmp_path)
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600  # full trading scope: owner-only, always
    loaded = Tokens.load(path)
    assert loaded.access_token == "at-1"
    assert loaded.refresh_token == "rt-1"


def test_load_missing_token_file_points_at_auth_command(tmp_path):
    with pytest.raises(RhMcpError, match="--auth"):
        Tokens.load(tmp_path / "absent.json")


def test_parse_rpc_body_plain_json():
    body = rpc_response(7, {"ok": True})
    assert _parse_rpc_body({"content-type": "application/json"}, body, 7)["result"] == {"ok": True}


def test_parse_rpc_body_sse_picks_matching_id():
    stream = (
        b'data: {"jsonrpc":"2.0","method":"notifications/progress","params":{}}\n\n'
        b'data: {"jsonrpc":"2.0","id":3,"result":{"answer":42}}\n\n'
    )
    message = _parse_rpc_body({"content-type": "text/event-stream"}, stream, 3)
    assert message["result"] == {"answer": 42}


def test_parse_rpc_body_sse_without_answer_raises():
    stream = b'data: {"jsonrpc":"2.0","id":9,"result":{}}\n\n'
    with pytest.raises(RhMcpError, match="rpc id 3"):
        _parse_rpc_body({"content-type": "text/event-stream"}, stream, 3)


def test_call_tool_initializes_session_and_returns_tool_json(tmp_path, monkeypatch):
    token_path = make_tokens(tmp_path)
    fake = FakeHttp(
        [
            (
                200,
                {"content-type": "application/json", "mcp-session-id": "sess-1"},
                rpc_response(1, {"serverInfo": {}}),
            ),
            (202, {}, b""),  # notifications/initialized
            (
                200,
                {"content-type": "application/json"},
                rpc_response(2, tool_text_result({"data": {"results": [1, 2]}})),
            ),
        ]
    )
    monkeypatch.setattr(rhmcp, "_http", fake)
    payload = McpClient(token_path).call_tool("get_earnings_calendar", {"days": 14})
    assert payload == {"data": {"results": [1, 2]}}
    init, notified, call = fake.requests
    assert init["headers"]["Authorization"] == "Bearer at-1"
    assert json.loads(init["data"])["method"] == "initialize"
    assert json.loads(notified["data"])["method"] == "notifications/initialized"
    # The session id from initialize must ride every subsequent request.
    assert notified["headers"]["Mcp-Session-Id"] == "sess-1"
    assert call["headers"]["Mcp-Session-Id"] == "sess-1"
    assert json.loads(call["data"])["params"]["arguments"] == {"days": 14}


def test_expired_token_refreshes_first_and_persists_rotation(tmp_path, monkeypatch):
    token_path = make_tokens(tmp_path, expires_in=-10.0)
    fake = FakeHttp(
        [
            (200, {}, json.dumps({"token_endpoint": "https://t/token"}).encode()),  # discovery
            (
                200,
                {},
                json.dumps(
                    {"access_token": "at-2", "refresh_token": "rt-2", "expires_in": 3600}
                ).encode(),
            ),
            (200, {"content-type": "application/json"}, rpc_response(1, {})),
            (202, {}, b""),
            (
                200,
                {"content-type": "application/json"},
                rpc_response(2, tool_text_result({"ok": 1})),
            ),
        ]
    )
    monkeypatch.setattr(rhmcp, "_http", fake)
    McpClient(token_path).call_tool("t", {})
    refresh = fake.requests[1]
    assert b"grant_type=refresh_token" in refresh["data"]
    assert b"refresh_token=rt-1" in refresh["data"]
    saved = Tokens.load(token_path)
    assert saved.access_token == "at-2"
    assert saved.refresh_token == "rt-2"  # rotated token persisted
    assert fake.requests[2]["headers"]["Authorization"] == "Bearer at-2"


def test_refresh_without_rotation_keeps_old_refresh_token(tmp_path, monkeypatch):
    token_path = make_tokens(tmp_path, expires_in=-10.0)
    fake = FakeHttp(
        [
            (200, {}, json.dumps({"token_endpoint": "https://t/token"}).encode()),
            (200, {}, json.dumps({"access_token": "at-2", "expires_in": 3600}).encode()),
            (200, {"content-type": "application/json"}, rpc_response(1, {})),
            (202, {}, b""),
            (
                200,
                {"content-type": "application/json"},
                rpc_response(2, tool_text_result({"ok": 1})),
            ),
        ]
    )
    monkeypatch.setattr(rhmcp, "_http", fake)
    McpClient(token_path).call_tool("t", {})
    assert Tokens.load(token_path).refresh_token == "rt-1"


def test_401_mid_session_refreshes_and_retries_once(tmp_path, monkeypatch):
    token_path = make_tokens(tmp_path)  # not expired locally, revoked server-side
    fake = FakeHttp(
        [
            (401, {}, b"unauthorized"),  # initialize rejected
            (200, {}, json.dumps({"token_endpoint": "https://t/token"}).encode()),
            (200, {}, json.dumps({"access_token": "at-2", "expires_in": 3600}).encode()),
            (200, {"content-type": "application/json"}, rpc_response(1, {})),  # retry
            (202, {}, b""),
            (
                200,
                {"content-type": "application/json"},
                rpc_response(2, tool_text_result({"ok": 1})),
            ),
        ]
    )
    monkeypatch.setattr(rhmcp, "_http", fake)
    assert McpClient(token_path).call_tool("t", {}) == {"ok": 1}
    assert fake.requests[3]["headers"]["Authorization"] == "Bearer at-2"


def test_persistent_401_raises_instead_of_looping(tmp_path, monkeypatch):
    token_path = make_tokens(tmp_path)
    fake = FakeHttp(
        [
            (401, {}, b"unauthorized"),
            (200, {}, json.dumps({"token_endpoint": "https://t/token"}).encode()),
            (200, {}, json.dumps({"access_token": "at-2", "expires_in": 3600}).encode()),
            (401, {}, b"still unauthorized"),
        ]
    )
    monkeypatch.setattr(rhmcp, "_http", fake)
    with pytest.raises(RhMcpError, match="HTTP 401"):
        McpClient(token_path).call_tool("t", {})


def test_rpc_error_response_raises(tmp_path, monkeypatch):
    token_path = make_tokens(tmp_path)
    fake = FakeHttp(
        [
            (
                200,
                {"content-type": "application/json"},
                json.dumps(
                    {"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "boom"}}
                ).encode(),
            ),
        ]
    )
    monkeypatch.setattr(rhmcp, "_http", fake)
    with pytest.raises(RhMcpError, match="boom"):
        McpClient(token_path).call_tool("t", {})


def test_tool_is_error_flag_raises(tmp_path, monkeypatch):
    token_path = make_tokens(tmp_path)
    fake = FakeHttp(
        [
            (200, {"content-type": "application/json"}, rpc_response(1, {})),
            (202, {}, b""),
            (
                200,
                {"content-type": "application/json"},
                rpc_response(2, {"isError": True, "content": [{"type": "text", "text": "denied"}]}),
            ),
        ]
    )
    monkeypatch.setattr(rhmcp, "_http", fake)
    with pytest.raises(RhMcpError, match="errored"):
        McpClient(token_path).call_tool("t", {})


def test_structured_content_preferred_over_text_blocks(tmp_path, monkeypatch):
    token_path = make_tokens(tmp_path)
    fake = FakeHttp(
        [
            (200, {"content-type": "application/json"}, rpc_response(1, {})),
            (202, {}, b""),
            (
                200,
                {"content-type": "application/json"},
                rpc_response(
                    2,
                    {
                        "structuredContent": {"s": 1},
                        "content": [{"type": "text", "text": '{"t": 2}'}],
                    },
                ),
            ),
        ]
    )
    monkeypatch.setattr(rhmcp, "_http", fake)
    assert McpClient(token_path).call_tool("t", {}) == {"s": 1}


def test_tokens_from_response_without_expires_in_is_treated_as_expired():
    tokens = rhmcp._tokens_from_response("cid", {"access_token": "at"}, fallback_refresh="rt")
    assert tokens.expires_at == 0.0  # 0 = unknown = refresh before next use


def test_corrupt_token_file_raises_with_auth_hint(tmp_path):
    path = tmp_path / "tokens.json"
    path.write_text("{ torn")
    with pytest.raises(RhMcpError, match="--auth"):
        Tokens.load(path)
    path.write_text('{"client_id": "cid"}')  # missing fields
    with pytest.raises(RhMcpError, match="--auth"):
        Tokens.load(path)


def test_redirects_are_never_followed():
    # urllib's default handler would copy the Bearer header onto a
    # cross-host redirect; the opener must refuse instead.
    assert rhmcp._NoRedirect().redirect_request() is None


def test_parse_rpc_body_sse_skips_pings_and_joins_multiline_events():
    stream = b'data: ping\n\ndata: {"jsonrpc":"2.0",\ndata: "id":3,"result":{"answer":42}}\n\n'
    message = _parse_rpc_body({"content-type": "text/event-stream"}, stream, 3)
    assert message["result"] == {"answer": 42}


def test_non_json_plain_body_raises_rhmcp_error_not_decode_error():
    with pytest.raises(RhMcpError, match="not JSON"):
        _parse_rpc_body({"content-type": "text/html"}, b"<html>gateway error</html>", 1)


def test_202_on_id_bearing_request_is_an_error(tmp_path, monkeypatch):
    token_path = make_tokens(tmp_path)
    fake = FakeHttp([(202, {}, b"")])  # initialize answered 202: no body will come
    monkeypatch.setattr(rhmcp, "_http", fake)
    with pytest.raises(RhMcpError, match="202"):
        McpClient(token_path).call_tool("t", {})


def test_refresh_request_carries_resource_binding(tmp_path, monkeypatch):
    token_path = make_tokens(tmp_path, expires_in=-10.0)
    fake = FakeHttp(
        [
            (200, {}, json.dumps({"token_endpoint": "https://t/token"}).encode()),
            (200, {}, json.dumps({"access_token": "at-2", "expires_in": 3600}).encode()),
            (200, {"content-type": "application/json"}, rpc_response(1, {})),
            (202, {}, b""),
            (
                200,
                {"content-type": "application/json"},
                rpc_response(2, tool_text_result({"ok": 1})),
            ),
        ]
    )
    monkeypatch.setattr(rhmcp, "_http", fake)
    McpClient(token_path).call_tool("t", {})
    assert b"resource=" in fake.requests[1]["data"]


def _run_auth_flow(monkeypatch, tmp_path, callback_qs):
    """Drive auth_flow end to end: fake _http for discovery/registration/
    token exchange, and a fake browser that 'visits' the consent page by
    hitting the localhost callback (loopback only) with callback_qs."""
    import socket
    import urllib.parse
    import urllib.request as _real_urllib

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    fake = FakeHttp(
        [
            (
                200,
                {},
                json.dumps(
                    {
                        "authorization_endpoint": "https://rh/oauth",
                        "token_endpoint": "https://rh/token",
                        "registration_endpoint": "https://rh/register",
                    }
                ).encode(),
            ),
            (201, {}, json.dumps({"client_id": "cid-new"}).encode()),
            (
                200,
                {},
                json.dumps(
                    {"access_token": "at-new", "refresh_token": "rt-new", "expires_in": 3600}
                ).encode(),
            ),
        ]
    )
    monkeypatch.setattr(rhmcp, "_http", fake)
    seen = {}

    def fake_browser(url):
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        seen.update(query)
        qs = callback_qs(query)
        with _real_urllib.urlopen(f"http://127.0.0.1:{port}/callback?{qs}", timeout=10) as resp:
            resp.read()
        return True

    monkeypatch.setattr(rhmcp.webbrowser, "open", fake_browser)
    rhmcp._CallbackHandler.result = {}
    token_path = tmp_path / "tokens.json"
    rhmcp.auth_flow(token_path, port=port)
    return fake, seen, token_path


def test_auth_flow_pkce_state_and_token_persistence(tmp_path, monkeypatch):
    import urllib.parse as up

    fake, seen, token_path = _run_auth_flow(
        monkeypatch, tmp_path, lambda q: f"code=authcode-1&state={q['state']}"
    )
    register, exchange = fake.requests[1], fake.requests[2]
    assert json.loads(register["data"])["token_endpoint_auth_method"] == "none"
    form = dict(up.parse_qsl(exchange["data"].decode()))
    assert form["code"] == "authcode-1"
    assert form["client_id"] == "cid-new"
    # PKCE: the challenge sent to the consent page must be the unpadded
    # urlsafe SHA-256 of the verifier sent to the token endpoint.
    import base64 as b64
    import hashlib as hl

    expected = b64.urlsafe_b64encode(hl.sha256(form["code_verifier"].encode()).digest())
    assert seen["code_challenge"] == expected.rstrip(b"=").decode()
    assert seen["code_challenge_method"] == "S256"
    saved = Tokens.load(token_path)
    assert saved.client_id == "cid-new"
    assert saved.refresh_token == "rt-new"
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


def test_auth_flow_rejects_state_mismatch(tmp_path, monkeypatch):
    with pytest.raises(RhMcpError, match="state mismatch"):
        _run_auth_flow(monkeypatch, tmp_path, lambda q: "code=authcode-1&state=forged")
