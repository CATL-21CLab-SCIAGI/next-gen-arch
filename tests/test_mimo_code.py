import copy
import json

import pytest

from archlab.data.mimo_code import bind_image, instance, select


def row(number):
    task = {
        "dataset_type": "opensource-code",
        "instance_id": f"format-code-task-{number:06d}",
        "docker_image": f"format-code-task-{number:06d}:latest",
        "cwd": "/testbed",
        "problem_statement": f"Repair issue {number}",
        "test_patch": "hidden tests",
        "test_command": "bash /testbed/test.sh",
        "verifier_timeout_sec": 1800,
    }
    return {
        "data_source": "opensource-code",
        "ability": "swe",
        "agent_name": "mimo_swe_agent",
        "prompt": [{"role": "user", "content": task["problem_statement"]}],
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "dataset_type": "opensource-code",
            "instance_id": task["instance_id"],
            "index": number,
            "instance_json": json.dumps(task),
        },
    }


def test_selection_is_stable_disjoint_and_bounded():
    rows = [row(i) for i in range(12)]
    chosen = select(rows, 4, 2, 42)
    assert chosen == select(list(reversed(rows)), 4, 2, 42)
    ids = [instance(x)["instance_id"] for x in chosen]
    assert len(set(ids)) == 6
    assert not set(ids[:4]) & set(ids[4:])
    with pytest.raises(ValueError, match="128"):
        select(rows, 128, 1, 42)


def test_image_binding_preserves_tools_tests_and_source_row():
    original = row(1)
    before = copy.deepcopy(original)
    bound = bind_image(original, {"resolved_image": "publisher/image@sha256:abc"})
    old, new = instance(original), instance(bound)
    assert new.pop("docker_image") == "publisher/image@sha256:abc"
    old.pop("docker_image")
    assert old == new
    assert original == before
    assert bound["prompt"] == original["prompt"]
    assert "hidden tests" not in json.dumps(bound["prompt"])


@pytest.mark.parametrize("mutation", ["noncode", "missing_tests", "prompt_leak", "duplicate"])
def test_rejects_non_agentic_or_ambiguous_data(mutation):
    rows = [row(1), row(2)]
    if mutation == "noncode":
        rows[0]["ability"] = "cyber"
    elif mutation == "missing_tests":
        task = instance(rows[0])
        task.pop("test_patch")
        rows[0]["extra_info"]["instance_json"] = json.dumps(task)
    elif mutation == "prompt_leak":
        rows[0]["prompt"][0]["content"] += " hidden tests"
    else:
        rows[1] = rows[0]
    with pytest.raises(ValueError):
        select(rows, 1, 1, 42)


def test_verifies_payloads_and_detects_tampering(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from archlab.data.mimo_code import REGISTRY, REPO, REVISION, digest, verify

    rows = select([row(1), row(2)], 1, 1, 42)
    index = []
    for split, original in zip(["train", "validation"], rows, strict=True):
        bound = bind_image(original, {"resolved_image": f"{REGISTRY}@sha256:{'a' * 64}"})
        pq.write_table(pa.Table.from_pylist([bound]), tmp_path / f"{split}.parquet")
        task = instance(bound)
        index.append(
            {
                "instance_id": task["instance_id"],
                "split": split,
                "source_index": bound["extra_info"]["index"],
                "prompt_sha256": digest(task["problem_statement"].encode()),
                "test_patch_sha256": digest(task["test_patch"].encode()),
                "resolved_image": task["docker_image"],
            }
        )
    (tmp_path / "index.jsonl").write_text("\n".join(map(json.dumps, index)))
    (tmp_path / "SOURCE_README.md").write_text("license: apache-2.0")
    manifest = {
        "dataset": REPO,
        "revision": REVISION,
        "selection": {"train": 1, "validation": 1, "seed": 42},
        "files": {
            p.name: {"sha256": digest(p.read_bytes()), "bytes": p.stat().st_size}
            for p in tmp_path.iterdir()
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert verify(path)["train"] == 1
    manifest["selection"]["eligibility"] = {
        "selection_basis": "explicit test candidate set",
        "instance_ids": [instance(r)["instance_id"] for r in rows],
    }
    path.write_text(json.dumps(manifest))
    assert verify(path)["validation"] == 1
    manifest["selection"]["eligibility"]["instance_ids"].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="outside the declared candidate set"):
        verify(path)
    del manifest["selection"]["eligibility"]
    path.write_text(json.dumps(manifest))
    (tmp_path / "train.parquet").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify(path)


def test_prompt_eos_curriculum_audit_is_immutable_and_never_truncates(tmp_path):
    from types import SimpleNamespace

    import pyarrow as pa
    import pyarrow.parquet as pq

    from archlab.data.mimo_code import audit_tokenization, digest

    for name, number in (("train", 1), ("validation", 2)):
        pq.write_table(pa.Table.from_pylist([row(number)]), tmp_path / f"{name}.parquet")
    selection = dict(seed=42, method="availability pilot", train=1, validation=1)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(selection=selection, runtime={"verifier": "native"})))
    model = tmp_path / "model"
    model.mkdir()
    (model / "generation_config.json").write_text('{"eos_token_id":9}')
    (model / "tokenizer_config.json").write_text('{"chat_template":"native"}')
    before = {p.name: digest(p.read_bytes()) for p in tmp_path.glob("*.parquet")}
    calls = []

    def render(messages, **kwargs):
        calls.append((messages, kwargs))
        return [1, 9, 2]

    tokenizer = SimpleNamespace(eos_token_id=99, apply_chat_template=render)
    audit = audit_tokenization(manifest, model, 3, tokenizer=tokenizer)
    assert audit["native_eos_ids"] == [9] and audit["tokenizer_eos_id"] == 99
    assert audit["splits"]["train"]["native_eos_tokens"] == 1
    assert audit["splits"]["validation"]["removed_tokens"] == 0
    assert audit["curriculum"]["selection"] == selection
    assert not audit["curriculum"]["reward_adaptive_selection"]
    assert calls[0][1] == dict(tokenize=True, add_generation_prompt=True, return_dict=False)
    assert before == {p.name: digest(p.read_bytes()) for p in tmp_path.glob("*.parquet")}
    with pytest.raises(ValueError, match="no implicit truncation"):
        audit_tokenization(manifest, model, 2, tokenizer=tokenizer)
