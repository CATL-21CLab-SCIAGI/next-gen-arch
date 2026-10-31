from archlab.training import dlc_jobs

SITE = {"workspace": "1", "quota": "q", "priority": 1, "image": "img", "vpc": {"VpcId": "v"},
        "nas": {"Uri": "nas://x/", "MountPath": "/mnt/nas/", "MountAccess": "RW"}, "gpu_full_nodes": "n1,n2"}


def test_cpu_pods_default_to_gpu_full_nodes_and_carry_no_gpu():
    body = dlc_jobs.job_body(SITE, name="pool", command="serve", pods=3, cpu=8, memory_gi=64)
    spec = body["JobSpecs"][0]
    assert spec["ResourceConfig"] == {"CPU": "8", "Memory": "64Gi", "GPU": "0"}
    assert spec["AssignNodeSpec"] == {"EnableAssignNode": True, "NodeNames": "n1,n2"}
    assert body["DataSources"] == [SITE["nas"]] and body["ResourceId"] == "q" and body["UserCommand"] == "serve"
    unpinned = dlc_jobs.job_body(SITE, name="x", command="c", pods=1, cpu=4, memory_gi=16, image="task:1", nodes="")
    assert "AssignNodeSpec" not in unpinned["JobSpecs"][0] and unpinned["JobSpecs"][0]["Image"] == "task:1"


def test_job_commands_restart_in_place():
    pool = dlc_jobs.sandbox_pool_command("/snap", "/registry", 8, "/logs")
    assert "PYTHONPATH=/snap/src python3 -m archlab.training.dlc_sandbox serve" in pool
    assert "/registry --slots 8 --port 8765" in pool and pool.startswith("cd /tmp; while true;")
    agents = dlc_jobs.agent_pods_command("/head", 44)
    assert "/head/ray-head.json" in agents and "--num-cpus=44 --num-gpus=0" in agents and "--block" in agents
    export = dlc_jobs.image_export_command("/root/exports/sha256:ab.tar.gz")
    assert "E=/root/exports/sha256:ab.env" in export and "gzip -1 > $O.part; mv $O.part $O" in export
    assert dlc_jobs.image_export_command("/o.tar.gz", env_only=True).endswith("mv $E.part $E")


def test_site_settings_must_be_named(tmp_path, monkeypatch):
    import json

    import pytest

    monkeypatch.delenv("ARCHLAB_DLC_SITE", raising=False)
    with pytest.raises(RuntimeError, match="site"):
        dlc_jobs.site()
    path = tmp_path / "site.json"
    path.write_text(json.dumps(SITE))
    assert dlc_jobs.site(path) == SITE
    monkeypatch.setenv("ARCHLAB_DLC_SITE", str(path))
    assert dlc_jobs.site() == SITE


def test_export_mirror_uses_private_site_config():
    import pytest

    with pytest.raises(RuntimeError, match="export_mirror"):
        dlc_jobs.export_mirror(SITE, "img1")
    cfg = {**SITE, "export_mirror": "registry.example.invalid/tasks:{}"}
    assert dlc_jobs.export_mirror(cfg, "sha256-ab") == "registry.example.invalid/tasks:sha256-ab"
