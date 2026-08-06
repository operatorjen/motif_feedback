from __future__ import annotations

import hmac
import secrets
from urllib.parse import urlparse

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from .constants import LOCAL_SESSION_TOKEN_BYTES


class LocalSessionGuard:
    def __init__(self) -> None:
        self.token = secrets.token_urlsafe(LOCAL_SESSION_TOKEN_BYTES)

    @staticmethod
    def origin_is_local(origin: str) -> bool:
        try:
            parsed = urlparse(origin)
        except ValueError:
            return False
        return parsed.scheme in {"http", "https"} and parsed.hostname in {
            "127.0.0.1",
            "localhost",
            "::1",
        }

    def valid_token(self, supplied: str | None) -> bool:
        return bool(supplied) and hmac.compare_digest(supplied, self.token)


class LocalSecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, guard: LocalSessionGuard) -> None:
        super().__init__(app)
        self.guard = guard

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.url.path.startswith("/api/"):
            origin = request.headers.get("origin")
            if origin and not self.guard.origin_is_local(origin):
                return JSONResponse({"detail": "Untrusted browser origin."}, status_code=403)
            supplied_token = request.headers.get("x-motif-token")
            if not self.guard.valid_token(supplied_token):
                return JSONResponse({"detail": "Missing or invalid local session token."}, status_code=403)

        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "base-uri 'none'; "
            "object-src 'none'; "
            "frame-src 'self' blob:; "
            "frame-ancestors 'none'; "
            "form-action 'self'; "
            "img-src 'self' data:; "
            "style-src 'self'; "
            "script-src 'self'; "
            "connect-src 'self'"
        )
        if request.url.path.startswith(("/api/", "/static/")):
            response.headers["Cache-Control"] = "no-store"
        return response
