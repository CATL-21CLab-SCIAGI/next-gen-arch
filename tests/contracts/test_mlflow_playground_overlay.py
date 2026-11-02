from pathlib import Path

from archlab.serving.mlflow_playground_overlay import install, script_src


def test_overlay_injects_stream_script_and_sse_headers(tmp_path):
    static_root = tmp_path / "build"
    static_root.mkdir()
    (static_root / "index.html").write_text(
        '<html><head><script src="/static-files/static/js/main.js"></script></head><body></body></html>'
    )
    gateway = tmp_path / "gateway_api.py"
    gateway.write_text(
        'return StreamingResponse(body, media_type="text/event-stream")\n'
        'return StreamingResponse(other, media_type="text/event-stream")\n'
    )
    receipt = install(static_root, gateway)
    html = (static_root / "index.html").read_text()
    js = (static_root / "archlab-playground-stream.js").read_text()
    patched = gateway.read_text()
    assert receipt["script"]["tag"] in html
    assert script_src(html).startswith("/static-files/archlab-playground-stream.js")
    assert "window.__archlabPlaygroundStream" in js
    assert html.count("archlab-playground-stream.js") == 1
    assert patched.count("X-Accel-Buffering") == 2
    assert install(static_root, gateway)["gateway_sse"]["patched"] is False
    assert Path(receipt["script"]["script"]).is_file()
