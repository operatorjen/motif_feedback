import asyncio

import httpx
from fastapi import FastAPI
from starlette.middleware.trustedhost import TrustedHostMiddleware

from motif_feedback.security import LocalSecurityMiddleware, LocalSessionGuard


def make_app() -> tuple[FastAPI, LocalSessionGuard]:
    guard = LocalSessionGuard()
    app = FastAPI()
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["testserver", "localhost"])
    app.add_middleware(LocalSecurityMiddleware, guard=guard)

    @app.get("/api/session")
    def session():
        return {"token": guard.token}

    @app.post("/api/change")
    def change():
        return {"ok": True}

    return app, guard


def request(app: FastAPI, method: str, path: str, **kwargs) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_mutation_requires_local_session_token():
    app, guard = make_app()

    assert request(app, "POST", "/api/change").status_code == 403
    response = request(
        app,
        "POST",
        "/api/change",
        headers={"X-Motif-Token": guard.token},
    )

    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


def test_mutation_rejects_untrusted_browser_origin_even_with_token():
    app, guard = make_app()

    response = request(
        app,
        "POST",
        "/api/change",
        headers={
            "X-Motif-Token": guard.token,
            "Origin": "https://attacker.example",
        },
    )

    assert response.status_code == 403

def test_security_policy_rejects_inline_application_scripts():
    app, _guard = make_app()
    response = request(app, "GET", "/api/session")

    assert response.status_code == 200
    policy = response.headers["content-security-policy"]
    assert "script-src 'self'" in policy
    assert "'unsafe-inline'" not in policy
