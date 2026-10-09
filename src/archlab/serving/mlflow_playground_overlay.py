"""Install a playground streaming overlay into the MLflow tracking UI.

The stock MLflow 3.16 playground POSTs without stream=true and waits for a
complete JSON reply. This copies a fetch interceptor into the UI, rewrites the
lazy playground chat function to consume SSE, and serves the overlay JS from
NAS so later edits do not require rebuilding the image.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCRIPT_NAME = "archlab-playground-stream.js"
SOURCE_JS = Path(__file__).with_name("mlflow_playground_stream.js")
SCRIPT_TAG = re.compile(r"\s*<script src=\"[^\"]*archlab-playground-stream\.js[^\"]*\"></script>")
STATIC_PREFIX = re.compile(r'src="([^"]*static-files/)')
SSE_MEDIA = 'media_type="text/event-stream"'
SSE_HEADERS = (
    'media_type="text/event-stream", '
    'headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}'
)
CHAT_FN = re.compile(
    r"const (\w+)=async (\w+)=>\{try\{const (\w+)=await\(0,(\w+)\.(\w+)\)\(\(0,(\w+)\.(\w+)\)"
    r'\("gateway/mlflow/v1/chat/completions"\),\{method:"POST",'
    r'headers:\{"Content-Type":"application/json"\},body:JSON\.stringify\(\2\)\}\);'
    r"return await \3\.json\(\)"
)
LIVE_JS_MARKER = "archlab-playground-stream-live"
SERVER_HOOK = f"""
    if str(path).split("?", 1)[0].startswith({SCRIPT_NAME!r}):
        from flask import send_file
        _archlab_js = {str(SOURCE_JS)!r}
        _resp = send_file(_archlab_js, mimetype="application/javascript", max_age=0)
        _resp.headers["Cache-Control"] = "no-store"
        return _resp
"""


def mlflow_paths():
    import mlflow

    root = Path(mlflow.__file__).resolve().parent
    return {
        "static": root / "server" / "js" / "build",
        "gateway_api": root / "server" / "gateway_api.py",
        "server_init": root / "server" / "__init__.py",
    }


def script_src(html, filename=SCRIPT_NAME):
    match = STATIC_PREFIX.search(html)
    prefix = match.group(1) if match else "/static-files/"
    return f"{prefix}{filename}?v=3"


def rewrite_chat_function(source):
    def replace(match):
        name, arg, response, fetch_mod, fetch_attr, ajax_mod, ajax_attr = match.groups()
        fallback = (
            f"(async {arg}=>{{const {response}=await(0,{fetch_mod}.{fetch_attr})"
            f"((0,{ajax_mod}.{ajax_attr})(\"gateway/mlflow/v1/chat/completions\"),"
            f"{{method:\"POST\",headers:{{\"Content-Type\":\"application/json\"}},"
            f"body:JSON.stringify({arg})}});return await {response}.json()}})"
        )
        return (
            f"const {name}=async {arg}=>{{try{{return await "
            f"(window.__archlabPlaygroundChat||{fallback})"
            f"({arg},(0,{fetch_mod}.{fetch_attr}),(0,{ajax_mod}.{ajax_attr}))"
        )

    rewritten, count = CHAT_FN.subn(replace, source, count=1)
    return rewritten, count


def install_script(static_root):
    index = Path(static_root) / "index.html"
    destination = Path(static_root) / SCRIPT_NAME
    if not index.is_file():
        raise FileNotFoundError(f"MLflow UI index is missing: {index}")
    destination.write_text(SOURCE_JS.read_text(), encoding="utf-8")
    html = index.read_text(encoding="utf-8")
    tag = f'<script src="{script_src(html)}"></script>'
    html = SCRIPT_TAG.sub("", html)
    # Run before the deferred main bundle so window.fetch is wrapped first.
    needle = "<head>"
    if needle in html:
        html = html.replace(needle, needle + tag, 1)
    elif "</head>" in html:
        html = html.replace("</head>", tag + "\n</head>", 1)
    else:
        html = html.rstrip() + "\n" + tag + "\n"
    index.write_text(html, encoding="utf-8")
    return {"index": str(index), "script": str(destination), "tag": tag}


def install_playground_chunk(static_root):
    patched = []
    for path in Path(static_root).glob("static/js/*.chunk.js"):
        original = path.read_text(encoding="utf-8")
        if "gateway/mlflow/v1/chat/completions" not in original:
            continue
        rewritten, count = rewrite_chat_function(original)
        if count:
            path.write_text(rewritten, encoding="utf-8")
            patched.append({"path": str(path), "replacements": count})
    return patched


def install_gateway_sse_headers(gateway_api):
    path = Path(gateway_api)
    text = path.read_text(encoding="utf-8")
    if "X-Accel-Buffering" in text:
        return {"path": str(path), "patched": False, "reason": "already patched"}
    if SSE_MEDIA not in text:
        raise ValueError(f"gateway streaming responses were not found in {path}")
    path.write_text(text.replace(SSE_MEDIA, SSE_HEADERS), encoding="utf-8")
    return {"path": str(path), "patched": True, "replacements": text.count(SSE_MEDIA)}


def install_live_js_route(server_init):
    path = Path(server_init)
    text = path.read_text(encoding="utf-8")
    if LIVE_JS_MARKER in text:
        return {"path": str(path), "patched": False, "reason": "already patched"}
    needle = "def serve_static_file(path):\n"
    if needle not in text:
        raise ValueError(f"serve_static_file was not found in {path}")
    hook = f"    # {LIVE_JS_MARKER}\n" + SERVER_HOOK
    path.write_text(text.replace(needle, needle + hook, 1), encoding="utf-8")
    return {"path": str(path), "patched": True}


def install(static_root=None, gateway_api=None, server_init=None):
    if static_root is None or gateway_api is None or server_init is None:
        detected = mlflow_paths()
        static_root = Path(static_root) if static_root is not None else detected["static"]
        gateway_api = Path(gateway_api) if gateway_api is not None else detected["gateway_api"]
        server_init = Path(server_init) if server_init is not None else detected["server_init"]
    return {
        "script": install_script(static_root),
        "playground_chunk": install_playground_chunk(static_root),
        "gateway_sse": install_gateway_sse_headers(gateway_api),
        "live_js": install_live_js_route(server_init),
    }


def main():
    json.dump(install(), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
