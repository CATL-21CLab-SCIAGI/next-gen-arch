"""Launch one private, eager SGLang worker for a verified full V4.1 export."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path


class _PrivateKey(str):
    def __repr__(self):
        return "'<redacted>'"


def server_arguments(model, variant, host, port):
    return [
        "--model-path", str(model), "--tokenizer-path", str(model),
        "--served-model-name", f"deepseek-v41-{variant}-latest",
        "--host", host, "--port", str(port),
        "--tp", "8", "--ep-size", "8", "--dtype", "bfloat16",
        "--load-format", "safetensors", "--attention-backend", "dsv4",
        "--moe-runner-backend", "triton", "--disable-shared-experts-fusion",
        "--enable-fp32-lm-head", "--disable-radix-cache", "--disable-cuda-graph",
        "--disable-custom-all-reduce",
        # One additional target token is needed to score a full 16K window.
        "--context-length", "32768", "--max-total-tokens", "65536",
        "--chunked-prefill-size", "256", "--max-running-requests", "4",
        "--mem-fraction-static", "0.86",
        "--reasoning-parser", "deepseek-v41", "--tool-call-parser", "deepseekv41",
        "--json-model-override-args", json.dumps({"architectures": ["DeepseekV4ForCausalLM"]}),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=28181)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--model-source-sha256", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "archlab.sglang_models"
    os.environ["ARCHLAB_SGLANG_MODEL_SHA256"] = args.model_source_sha256
    os.environ["SGLANG_DEFAULT_THINKING"] = "1"
    os.environ["SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE"] = "0"
    os.environ["SGLANG_OPT_FUSE_WQA_WKV"] = "0"
    os.environ["SGLANG_SHARED_EXPERT_TP1"] = "1"
    os.environ["SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED"] = "0"
    config = json.loads((args.model / "config.json").read_text())
    metadata = config["archlab"]
    direct = args.model / "DIRECT_CHECKPOINT.json"
    if direct.exists():
        complete = json.loads(direct.read_text())
        if (complete["format"] != "archlab-v41-direct-load-v1" or complete["weights_copied"]
                or complete["complete_sha256"] != metadata["complete_sha256"]
                or complete["source"]["checkpoint"] != metadata["full_checkpoint"]):
            raise ValueError("direct checkpoint admission failed")
    else:
        complete = json.loads((args.model / "EXPORT_COMPLETE.json").read_text())
        if (not complete["all_source_tensor_names_exported"] or not complete["payload_checksums_verified"]
                or complete["requantized"] or complete["optimizer_exported"]):
            raise ValueError("export admission failed")
    if (metadata["checkpoint_cursor"] != complete["source"]["cursor"]
            or metadata["variant"] != complete["source"]["variant"]):
        raise ValueError("checkpoint provenance differs")
    token = args.token_file.read_text().strip()
    if len(token) < 24:
        raise ValueError("missing team authentication key")
    from sglang.launch_server import run_server
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM as NativeV41
    from sglang.srt.models.registry import ModelRegistry
    from sglang.srt.server_args import prepare_server_args

    actual = hashlib.sha256(Path(inspect.getfile(NativeV41)).read_bytes()).hexdigest()
    if actual != args.model_source_sha256:
        raise ValueError("runtime model implementation fingerprint differs")
    resolved, _ = ModelRegistry.resolve_model_cls(["DeepseekV4ForCausalLM"])
    if resolved.__module__ != "archlab.sglang_models.deepseek_v41":
        raise ValueError("the external full-checkpoint implementation was not registered")
    server_args = prepare_server_args(server_arguments(
        args.model, metadata["variant"], args.host, args.port))
    server_args.api_key = _PrivateKey(token)
    if args.check_only:
        print(json.dumps(dict(passed=True, variant=metadata["variant"],
                              cursor=metadata["checkpoint_cursor"],
                              model_class=resolved.__module__ + "." + resolved.__name__,
                              model_source_sha256=actual, weights_loaded=False)))
        return
    run_server(server_args)


if __name__ == "__main__":
    main()
