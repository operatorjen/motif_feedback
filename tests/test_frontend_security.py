from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_demo_iframe_keeps_an_opaque_origin_and_embedded_policy():
    html = (ROOT / "motif_feedback" / "static" / "index.html").read_text(encoding="utf-8")

    assert 'sandbox="allow-scripts"' in html
    assert "allow-same-origin" not in html
    assert 'csp="default-src \'none\'' in html


def test_demo_errors_use_text_content_and_meta_refresh_is_removed():
    source = (ROOT / "motif_feedback" / "static" / "js" / "demo_sandbox.js").read_text(
        encoding="utf-8"
    )

    assert "error.textContent =" in source
    assert 'meta.httpEquiv.toLowerCase() === "refresh"' in source


def test_fresh_install_requires_explicit_runtime_setup():
    html = (ROOT / "motif_feedback" / "static" / "index.html").read_text(encoding="utf-8")
    source = (ROOT / "motif_feedback" / "static" / "js" / "app.js").read_text(encoding="utf-8")

    assert 'id="provider-agent-a" required' in html
    assert 'id="model-agent-a" type="text"' in html
    assert "!session.setup_complete || !session.key_configured" in source
    assert "sessionData.runtime || {" in source
