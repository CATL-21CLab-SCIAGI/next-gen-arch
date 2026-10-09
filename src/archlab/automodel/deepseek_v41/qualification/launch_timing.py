"""Launch bounded timing stages sequentially on one retained two-node allocation pair."""

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--nodes", nargs=2, required=True)
    p.add_argument("--ssh-prefix", default="", help="Optional SSH alias prefix for plan node names")
    p.add_argument("--stages", nargs="+", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--variant", default="normal")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--microbatch", type=int, default=4)
    p.add_argument("--repeat-batches", type=int, default=0)
    p.add_argument("--table-lr-scale", type=float, default=1.0)
    p.add_argument("--router-rate", type=float, default=0.01)
    p.add_argument("--schedule", action="store_true")
    p.add_argument("--performance-contract", type=Path)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    for stage in a.stages:
        output = a.output / (stage + "-" + a.variant)
        output.mkdir(parents=True, exist_ok=False)
        jobs = []
        for rank, node in enumerate(a.nodes):
            env = dict(plan["environment"])
            env["PYTHONPATH"] = str(a.source / "src") + ":" + env["PYTHONPATH"].split(":", 1)[1]
            command = [
                "/opt/venv/bin/python",
                "-m",
                "torch.distributed.run",
                "--nnodes=2",
                "--nproc-per-node=8",
                "--node-rank",
                str(rank),
                "--master-addr",
                plan["nodes"][a.nodes[0]]["ip"],
                "--master-port",
                str(a.port),
                "--max-restarts=0",
                str(
                    a.source
                    / "src/archlab/automodel/deepseek_v41/qualification/nsight_training_probe.py"
                ),
                "--source",
                str(a.source),
                "--stage",
                stage,
                "--width",
                str(a.width),
                "--microbatch",
                str(a.microbatch),
                "--repeat-batches",
                str(a.repeat_batches),
                "--table-lr-scale",
                str(a.table_lr_scale),
                "--router-rate",
                str(a.router_rate),
                "--variant",
                a.variant,
                "--output",
                str(output),
                "--warmup",
                str(a.warmup),
                "--steps",
                str(a.steps),
            ]
            if a.performance_contract is not None:
                command.extend(["--performance-contract", str(a.performance_contract)])
            if a.schedule:
                command.append("--schedule")
            remote = (
                "exec env "
                + " ".join(shlex.quote(k + "=" + v) for k, v in env.items())
                + " "
                + shlex.join(command)
            )
            log = (output / (node + ".log")).open("x")
            jobs.append(
                subprocess.Popen(
                    ["ssh", a.ssh_prefix + node, remote],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
            )
            log.close()
        codes = [j.wait() for j in jobs]
        (output / "launcher.json").write_text(
            json.dumps(
                dict(exit_codes=codes, source=str(a.source), stage=stage, pid=os.getpid()), indent=2
            )
        )
        if any(codes):
            raise SystemExit(f"{stage} failed: {codes}")


if __name__ == "__main__":
    main()
