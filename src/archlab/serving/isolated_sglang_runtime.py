"""Execute the pinned image in an unprivileged user/private mount namespace.

The host NeMo/PyTorch installation is left untouched. Only GPU driver libraries,
devices, system information and the existing project storage are bound in.
The image supplies the CUDA toolkit, PyTorch, NCCL, SGLang and other packages.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def prepare(config):
    rootfs = Path(config["rootfs"])
    manifest = json.loads(Path(config["image_manifest"]).read_text())
    ready = json.loads((rootfs.parent / "RUNTIME_COPY_COMPLETE.json").read_text())
    if ready["image_id"] != manifest["id"]:
        raise ValueError("runtime filesystem and pinned image identity differ")
    cache = Path(config["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    cuda_driver = Path("/usr/local/cuda/compat/lib")
    nvml = Path("/lib/x86_64-linux-gnu/libnvidia-ml.so.1").resolve(strict=True)
    if not (cuda_driver / "libcuda.so.1").exists():
        raise ValueError("expected the existing DLC CUDA compatibility driver")
    for folder in ("dev", "proc", "sys", "mnt/nas", "mnt/oss", "run/archlab-driver", "run/archlab-cuda-compat",
                   "usr/local/nvidia/bin"):
        (rootfs / folder).mkdir(parents=True, exist_ok=True)
    for name in ("etc/hosts", "etc/resolv.conf", "run/archlab-driver/libnvidia-ml.so.1",
                 "usr/local/nvidia/bin/nvidia-smi"):
        target = rootfs / name
        if not target.exists() and not target.is_symlink():
            target.touch(exist_ok=False)
    guest = dict(item.split("=", 1) for item in manifest["environment"])
    guest["LD_LIBRARY_PATH"] = "/run/archlab-cuda-compat:/run/archlab-driver:" + guest.get("LD_LIBRARY_PATH", "")
    guest["PYTHONPATH"] = config["source"] + (":" + guest["PYTHONPATH"] if guest.get("PYTHONPATH") else "")
    guest.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                 CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", OMP_NUM_THREADS="4",
                 CUBLAS_WORKSPACE_CONFIG=":4096:8",
                 TRITON_CACHE_DIR=str(cache / "triton"),
                 TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
                 TORCH_EXTENSIONS_DIR=str(cache / "torch-extensions"))
    bindings = [(p, p, p in ("/dev", "/proc", "/sys")) for p in
                ("/dev", "/proc", "/sys", "/mnt/nas", "/mnt/oss", "/etc/resolv.conf", "/etc/hosts")]
    bindings.extend([(str(cuda_driver), "/run/archlab-cuda-compat", False),
                     (str(nvml), "/run/archlab-driver/libnvidia-ml.so.1", False)])
    if executable := shutil.which("nvidia-smi"):
        bindings.append((str(Path(executable).resolve()), "/usr/local/nvidia/bin/nvidia-smi", False))
    return rootfs, guest, bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--namespace-child", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    remaining = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if not remaining:
        raise ValueError("a Python command is required")
    if not args.namespace_child:
        environment = os.environ.copy()
        environment["ARCHLAB_PARENT_MOUNT_NAMESPACE"] = os.readlink("/proc/self/ns/mnt")
        argv = ["/usr/bin/unshare", "--user", "--map-root-user", "--mount", "--propagation", "private",
                sys.executable, "-m", "archlab.serving.isolated_sglang_runtime",
                "--namespace-child", "--config", str(args.config), "--", *remaining]
        os.execve(argv[0], argv, environment)
    parent_namespace = os.environ.get("ARCHLAB_PARENT_MOUNT_NAMESPACE")
    if not parent_namespace or parent_namespace == os.readlink("/proc/self/ns/mnt"):
        raise ValueError("refusing to change mounts outside the new private namespace")
    uid_map = Path("/proc/self/uid_map").read_text().split()
    if len(uid_map) != 3 or uid_map[0] != "0" or uid_map[2] != "1":
        raise ValueError("expected a single-user, unprivileged namespace")
    config = json.loads(args.config.read_text())
    rootfs, environment, bindings = prepare(config)
    for source, destination, recursive in bindings:
        subprocess.run(["/usr/bin/mount", "--rbind" if recursive else "--bind", source,
                        str(rootfs / destination.lstrip("/"))], check=True)
    os.chroot(rootfs)
    os.chdir(config["working_directory"])
    os.execve("/opt/sglang/bin/python", ["/opt/sglang/bin/python", *remaining], environment)


if __name__ == "__main__":
    main()
