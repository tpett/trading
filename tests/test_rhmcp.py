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
