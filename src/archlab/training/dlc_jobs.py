"""CPU DLC jobs around agentic RL: sandbox pools, Ray agent pods and task-image exports, built by one job body.

Site settings (workspace, quota, VPC, image, NAS, the GPU-full node list) come from a private JSON
(`ARCHLAB_DLC_SITE`, default `.runtime/dlc-site.json`). CPU pods go only to nodes whose GPUs are all occupied, so
CPU work never blocks a free GPU; refresh `gpu_full_nodes` from the quota node listing before growing a pool.
usage: python -m archlab.training.dlc_jobs sandbox-pool --name N --pods P --slots S --source <snapshot> --registry <dir>
       python -m archlab.training.dlc_jobs agent-pods --name N --pods P --ray-cpus R --head-dir <dir>
       python -m archlab.training.dlc_jobs status <job id>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

SITE = Path(os.environ.get("ARCHLAB_DLC_SITE", Path(__file__).resolve().parents[3] / ".runtime/dlc-site.json"))
IMAGES = "/mnt/nas/evergreen/sandbox-images/mimo-v2.6-rl-oss"
EXPORT_MIRROR = "<private-registry>/pai-sandbox-tasks/mimo-v2.6-rl-oss:{}"


def site() -> dict:
    return json.loads(SITE.read_text())


def job_body(config: dict, *, name: str, command: str, pods: int, cpu: int, memory_gi: int,
             image: str | None = None, shared_memory_gi: int | None = None, nodes: str | None = None) -> dict:
    resources = {"CPU": str(cpu), "Memory": f"{memory_gi}Gi", "GPU": "0"}
    if shared_memory_gi:
        resources["SharedMemory"] = f"{shared_memory_gi}Gi"
    spec = {"Type": "Worker", "Image": image or config["image"], "PodCount": pods, "ResourceConfig": resources}
    nodes = config["gpu_full_nodes"] if nodes is None else nodes
    if nodes:
        spec["AssignNodeSpec"] = {"EnableAssignNode": True, "NodeNames": nodes}
    return {"WorkspaceId": config["workspace"], "DisplayName": name, "JobType": "PyTorchJob",
            "ResourceId": config["quota"], "Priority": config["priority"], "Accessibility": "PRIVATE",
            "UserCommand": command, "UserVpc": config["vpc"], "JobSpecs": [spec], "DataSources": [config["nas"]]}


def sandbox_pool_command(source: str, registry: str, slots: int, log_dir: str) -> str:
    """Pool servers from an immutable source snapshot; each restarts in place and re-announces in `registry`."""
    return (f"cd /tmp; while true; do PYTHONPATH={source}/src python3 -m archlab.training.dlc_sandbox serve "
            f"{IMAGES} {registry} --slots {slots} --port 8765 >> {log_dir}/serve-$(hostname).log 2>&1; sleep 5; done")


def agent_pods_command(head_dir: str, ray_cpus: int) -> str:
    """CPU Ray workers that join whichever head `<head_dir>/ray-head.json` names, rejoining after a head restart."""
    read = f"python3 -c \"import json;print(json.load(open('{head_dir}/ray-head.json'))['address'])\""
    return (f"cd /tmp; mkdir -p {head_dir}/agent-pods; while true; do if [ -f {head_dir}/ray-head.json ]; then "
            f"A=$({read}); /opt/venv/bin/ray stop --force >/dev/null 2>&1; "
            f"NO_PROXY=\"*\" no_proxy=\"*\" /opt/venv/bin/ray start --address=$A --num-cpus={ray_cpus} --num-gpus=0 "
            f"--resources='{{\"agent_cpu\": {ray_cpus}}}' --disable-usage-stats --block "
            f">> {head_dir}/agent-pods/$(hostname).log 2>&1; fi; sleep 15; done")


def image_export_command(out: str, env_only: bool = False) -> str:
    """Record the image's runtime environment and tar its root filesystem (one file system) to `out`."""
    env = out.removesuffix(".tar.gz") + ".env"
    record = f"set -e; E={env}; env -0 > $E.part; mv $E.part $E"
    if env_only:
        return record
    return (f"{record}; O={out}; cd /; tar --one-file-system --numeric-owner -cpf - --exclude=./proc --exclude=./sys "
            "--exclude=./dev --exclude=./mnt --exclude=./tmp --exclude=./run . | gzip -1 > $O.part; mv $O.part $O")


def aliyun(config: dict, action: str, *arguments: str) -> dict:
    reply = subprocess.run(["aliyun", "pai-dlc", action, "--region", config["region"], "--endpoint",
                            config["endpoint"], *arguments], capture_output=True, text=True)
    if reply.returncode:
        raise RuntimeError(f"{action} failed: {(reply.stderr or reply.stdout)[-600:]}")
    return json.loads(reply.stdout)


def create(body: dict, config: dict | None = None) -> str:
    return aliyun(config or site(), "CreateJob", "--body", json.dumps(body))["JobId"]


def status(job_id: str, config: dict | None = None) -> str:
    return aliyun(config or site(), "GetJob", "--JobId", job_id)["Status"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="kind", required=True)
    for kind in ("sandbox-pool", "agent-pods"):
        command = sub.add_parser(kind)
        command.add_argument("--name", required=True)
        command.add_argument("--pods", type=int, required=True)
        command.add_argument("--cpu", type=int, default=8)
        command.add_argument("--memory", type=int, default=64, help="GiB per pod")
        command.add_argument("--nodes", help="override the site's GPU-full node list")
    pool, agents = sub.choices["sandbox-pool"], sub.choices["agent-pods"]
    pool.add_argument("--slots", type=int, required=True)
    pool.add_argument("--source", required=True, help="immutable source snapshot that serves the pool")
    pool.add_argument("--registry", required=True, help="directory where each server announces itself")
    pool.add_argument("--log-dir", default="/mnt/nas/evergreen/arch/results/dlc-sandbox-20261015")
    agents.add_argument("--ray-cpus", type=int, required=True)
    agents.add_argument("--head-dir", required=True)
    sub.add_parser("status").add_argument("job")
    args = parser.parse_args(argv)
    config = site()
    if args.kind == "status":
        print(status(args.job, config))
        return
    if args.kind == "sandbox-pool":
        command, shared = sandbox_pool_command(args.source, args.registry, args.slots, args.log_dir), None
    else:
        command, shared = agent_pods_command(args.head_dir, args.ray_cpus), 16
    body = job_body(config, name=args.name, command=command, pods=args.pods, cpu=args.cpu, memory_gi=args.memory,
                    shared_memory_gi=shared, nodes=args.nodes)
    print(json.dumps({"JobId": create(body, config), "name": args.name, "pods": args.pods}))


if __name__ == "__main__":
    main()
