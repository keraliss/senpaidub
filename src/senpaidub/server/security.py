"""Local-only access control for the web UI.

- The Host header must name this machine (127.0.0.1 / localhost, on our port), which blocks
  DNS-rebinding attacks from other websites open in the same browser.
- `senpaidub serve` prints a link with a random token. Opening `/?token=…` sets an HttpOnly,
  SameSite=Strict session cookie and redirects to the clean URL; every page and API call then
  needs that cookie. Other local web pages can't read or send it.
- Requests that carry an Origin header (browsers add it to cross-site fetches and to POSTs)
  must come from this origin.
"""

from __future__ import annotations

import hmac
from urllib.parse import parse_qsl, urlencode, urlsplit

from starlette.types import ASGIApp, Receive, Scope, Send

COOKIE = "senpai_session"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
PUBLIC_PREFIXES = ("/assets/",)  # static CSS/JS/SVG; nothing private
PUBLIC_PATHS = {"/favicon.ico"}

NEED_LINK_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Senpai Dub</title>
<style>body{font-family:system-ui,sans-serif;background:#F4EFE6;color:#141217;margin:0;padding:48px 16px}
main{max-width:560px;margin:0 auto;border:3px solid #141217;background:#fff;padding:24px;box-shadow:6px 6px 0 #141217}
code{background:#FFD23F;padding:2px 6px}</style></head>
<body><main><h1>Senpai Dub</h1>
<p>Open the link printed by <code>senpaidub serve</code> in your terminal.</p>
<p>It contains a one-time key that keeps other websites from controlling the app.</p></main></body></html>
"""


def split_host(host: str) -> tuple[str, int | None]:
    host = host.strip().lower()
    if host.startswith("["):  # [::1]:7860
        end = host.find("]")
        name, rest = host[: end + 1], host[end + 1 :]
        port = rest[1:] if rest.startswith(":") else ""
    else:
        name, _, port = host.partition(":")
    try:
        return name, int(port) if port else None
    except ValueError:
        return name, -1


def host_allowed(host: str | None, port: int | None) -> bool:
    if not host:
        return False
    name, p = split_host(host)
    if name not in LOCAL_HOSTS:
        return False
    return port is None or p == port


def token_ok(given: str | None, token: str) -> bool:
    return bool(given) and hmac.compare_digest(given.encode(), token.encode())


def _cookie(scope: Scope) -> str | None:
    for k, v in scope.get("headers") or []:
        if k == b"cookie":
            for part in v.decode("latin-1").split(";"):
                name, _, value = part.strip().partition("=")
                if name == COOKIE:
                    return value
    return None


def _header(scope: Scope, name: bytes) -> str | None:
    for k, v in scope.get("headers") or []:
        if k == name:
            return v.decode("latin-1")
    return None


async def _send_simple(send: Send, status: int, body: bytes, content_type: str, extra: list | None = None) -> None:
    headers = [(b"content-type", content_type.encode()), (b"content-length", str(len(body)).encode()),
               (b"cache-control", b"no-store")] + (extra or [])  # fmt: skip
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class LocalOnlyMiddleware:
    """Pure ASGI middleware (keeps SSE streaming and disconnects working)."""

    def __init__(self, app: ASGIApp, token: str, port: int | None = None) -> None:
        self.app = app
        self.token = token
        self.port = port

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope.get("path") or "/"
        is_api = path.startswith("/api/")
        host = _header(scope, b"host")
        if not host_allowed(host, self.port):
            await _send_simple(send, 400, b'{"error": "Bad Host header: Senpai Dub only answers on localhost."}',
                               "application/json")  # fmt: skip
            return
        origin = _header(scope, b"origin")
        if origin and origin != "null":
            o = urlsplit(origin)
            if o.scheme != "http" or not host_allowed(o.netloc, self.port):
                await _send_simple(send, 403, b'{"error": "Cross-origin request refused."}', "application/json")
                return

        # First visit: /?token=… → set the cookie and redirect to the same URL without the token.
        query = dict(parse_qsl((scope.get("query_string") or b"").decode("latin-1"), keep_blank_values=True))
        if not is_api and "token" in query and scope.get("method") in ("GET", "HEAD"):
            if token_ok(query.get("token"), self.token):
                rest = {k: v for k, v in query.items() if k != "token"}
                location = path + (f"?{urlencode(rest)}" if rest else "")
                cookie = f"{COOKIE}={self.token}; HttpOnly; SameSite=Strict; Path=/"
                await _send_simple(send, 303, b"", "text/plain",
                                   [(b"location", location.encode()), (b"set-cookie", cookie.encode())])  # fmt: skip
                return
            await _send_simple(send, 401, NEED_LINK_HTML.encode(), "text/html; charset=utf-8")
            return

        if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
            await self.app(scope, receive, send)
            return
        if not token_ok(_cookie(scope), self.token):
            if is_api:
                await _send_simple(send, 401, b'{"error": "Open the link printed by senpaidub serve."}',
                                   "application/json")  # fmt: skip
            else:
                await _send_simple(send, 401, NEED_LINK_HTML.encode(), "text/html; charset=utf-8")
            return
        await self.app(scope, receive, send)
