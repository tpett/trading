"""Minimal OAuth + MCP client for Robinhood's hosted trading MCP server.

The daily earnings-calendar dump must run unattended on the mini with no
LLM and no heavyweight MCP SDK in the loop, so this speaks the two wire
protocols directly with the stdlib only:

- OAuth 2.1 public client (RFC 7591 dynamic registration + PKCE S256 +
  refresh_token grant), discovered from the server's
  /.well-known/oauth-authorization-server metadata. `auth_flow()` is the
  ONE-TIME interactive consent step (browser + localhost callback); every
  later run refreshes silently.
- MCP Streamable HTTP: initialize -> notifications/initialized ->
  tools/call, with responses arriving as plain JSON or as an SSE stream
  depending on the server's mood.

SECURITY: the stored token grants the FULL MCP scope -- including order
placement on a real-money account. The token file is written 0600 and its
path stays outside the repo; nothing here ever journals or logs token
material.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import http.server
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

RESOURCE_URL = "https://agent.robinhood.com/mcp/trading"
DISCOVERY_URL = "https://agent.robinhood.com/.well-known/oauth-authorization-server"
DEFAULT_TOKEN_PATH = Path.home() / ".config" / "trading" / "robinhood-mcp.json"
PROTOCOL_VERSION = "2025-03-26"
CALLBACK_PORT = 8763
_REFRESH_MARGIN_S = 60


class RhMcpError(RuntimeError):
    pass


def _http(
    url: str, data: bytes | None = None, headers: dict[str, str] | None = None
) -> tuple[int, dict[str, str], bytes]:
    """Network touchpoint, isolated for monkeypatching. POST when data is
    given, GET otherwise. HTTP error statuses are returned, not raised, so
    callers can implement 401-refresh-retry."""
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read()


@dataclasses.dataclass
class Tokens:
    client_id: str
    access_token: str
    refresh_token: str
    expires_at: float  # epoch seconds; 0 = unknown, treat as expired

    @classmethod
    def load(cls, path: Path) -> Tokens:
        if not path.exists():
            raise RhMcpError(
                f"no token file at {path} -- run `uv run python "
                "scripts/dump_earnings_calendar.py --auth` once to authorize"
            )
        raw = json.loads(path.read_text())
        return cls(**{f.name: raw[f.name] for f in dataclasses.fields(cls)})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.touch(mode=0o600)
        tmp.write_text(json.dumps(dataclasses.asdict(self), indent=2))
        os.replace(tmp, path)
        path.chmod(0o600)


def _discover() -> dict:
    status, _, body = _http(DISCOVERY_URL)
    if status != 200:
        raise RhMcpError(f"OAuth discovery failed: HTTP {status}")
    return json.loads(body)


def _token_request(token_endpoint: str, form: dict[str, str]) -> dict:
    status, _, body = _http(
        token_endpoint,
        data=urllib.parse.urlencode(form).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if status != 200:
        raise RhMcpError(f"token endpoint HTTP {status}: {body[:200]!r}")
    return json.loads(body)


def _tokens_from_response(client_id: str, payload: dict, fallback_refresh: str = "") -> Tokens:
    expires_at = time.time() + float(payload["expires_in"]) if "expires_in" in payload else 0.0
    return Tokens(
        client_id=client_id,
        access_token=payload["access_token"],
        # Servers MAY rotate the refresh token on use; keep the old one if not.
        refresh_token=payload.get("refresh_token") or fallback_refresh,
        expires_at=expires_at,
    )


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    result: dict = {}

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        type(self).result = {k: v[0] for k, v in params.items()}
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Authorized. You can close this tab.")

    def log_message(self, *args):  # silence request logging
        pass


def auth_flow(token_path: Path = DEFAULT_TOKEN_PATH, port: int = CALLBACK_PORT) -> None:
    """One-time interactive consent. Registers a public client, sends the
    browser to Robinhood's consent page, catches the localhost redirect, and
    persists tokens. Over SSH, forward the callback port first:
    ssh -L 8763:127.0.0.1:8763 mac-m1 -- then open the printed URL locally."""
    meta = _discover()
    redirect_uri = f"http://127.0.0.1:{port}/callback"
    status, _, body = _http(
        meta["registration_endpoint"],
        data=json.dumps(
            {
                "client_name": "trading-earnings-journal",
                "redirect_uris": [redirect_uri],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    if status not in (200, 201):
        raise RhMcpError(f"client registration failed: HTTP {status}: {body[:200]!r}")
    client_id = json.loads(body)["client_id"]

    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=")
    state = secrets.token_urlsafe(16)
    auth_url = (
        meta["authorization_endpoint"]
        + "?"
        + urllib.parse.urlencode(
            {
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code_challenge": challenge.decode(),
                "code_challenge_method": "S256",
                "state": state,
                "resource": RESOURCE_URL,
            }
        )
    )

    server = http.server.HTTPServer(("127.0.0.1", port), _CallbackHandler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    print(f"Open this URL in a browser to authorize:\n\n{auth_url}\n")
    webbrowser.open(auth_url)
    thread.join(timeout=600)
    server.server_close()
    result = _CallbackHandler.result
    if not result:
        raise RhMcpError("no OAuth callback received within 10 minutes")
    if result.get("state") != state:
        raise RhMcpError("OAuth state mismatch; aborting")
    if "code" not in result:
        raise RhMcpError(f"authorization denied: {result}")

    payload = _token_request(
        meta["token_endpoint"],
        {
            "grant_type": "authorization_code",
            "code": result["code"],
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": verifier,
            "resource": RESOURCE_URL,
        },
    )
    _tokens_from_response(client_id, payload).save(token_path)
    print(f"tokens saved to {token_path}")


def _parse_rpc_body(headers: dict[str, str], body: bytes, rpc_id: int) -> dict:
    """A Streamable HTTP response is either a plain JSON-RPC message or an
    SSE stream of them; find the response matching rpc_id."""
    content_type = headers.get("content-type", "")
    if content_type.startswith("text/event-stream"):
        for line in body.decode().splitlines():
            if line.startswith("data:"):
                message = json.loads(line[len("data:") :].strip())
                if message.get("id") == rpc_id:
                    return message
        raise RhMcpError(f"no SSE event answered rpc id {rpc_id}")
    return json.loads(body)


class McpClient:
    """Stateless-per-run MCP caller: refresh token, initialize a session,
    call the tool. Three HTTP round-trips a day is cheap; holding sessions
    across days is not worth the state."""

    def __init__(self, token_path: Path = DEFAULT_TOKEN_PATH):
        self._token_path = token_path
        self._tokens = Tokens.load(token_path)
        self._session_id: str | None = None
        self._rpc_id = 0

    def _refresh(self) -> None:
        meta = _discover()
        payload = _token_request(
            meta["token_endpoint"],
            {
                "grant_type": "refresh_token",
                "refresh_token": self._tokens.refresh_token,
                "client_id": self._tokens.client_id,
                "resource": RESOURCE_URL,
            },
        )
        self._tokens = _tokens_from_response(
            self._tokens.client_id, payload, fallback_refresh=self._tokens.refresh_token
        )
        self._tokens.save(self._token_path)

    def _post(self, message: dict) -> tuple[int, dict[str, str], bytes]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self._tokens.access_token}",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return _http(RESOURCE_URL, data=json.dumps(message).encode(), headers=headers)

    def _rpc(self, method: str, params: dict) -> dict:
        self._rpc_id += 1
        rpc_id = self._rpc_id
        message = {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}
        status, headers, body = self._post(message)
        if status == 401:  # stale access token despite the expiry check: retry once
            self._refresh()
            status, headers, body = self._post(message)
        if status not in (200, 202):
            raise RhMcpError(f"{method}: HTTP {status}: {body[:200]!r}")
        if session := headers.get("mcp-session-id"):
            self._session_id = session
        response = _parse_rpc_body(headers, body, rpc_id)
        if "error" in response:
            raise RhMcpError(f"{method}: {response['error']}")
        return response["result"]

    def _ensure_ready(self) -> None:
        if self._tokens.expires_at < time.time() + _REFRESH_MARGIN_S:
            self._refresh()
        if self._session_id is None:
            self._rpc(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "trading-earnings-journal", "version": "1.0"},
                },
            )
            # Fire-and-forget per spec; servers answer 202 with no body.
            self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Returns the tool's JSON payload (the server wraps JSON in a text
        content block; structuredContent wins when present)."""
        self._ensure_ready()
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise RhMcpError(f"tool {name} errored: {json.dumps(result)[:300]}")
        if "structuredContent" in result:
            return result["structuredContent"]
        for block in result.get("content", []):
            if block.get("type") == "text":
                return json.loads(block["text"])
        raise RhMcpError(f"tool {name} returned no JSON content")
