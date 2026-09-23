"""CPU provenance and exact weight-reader tests; no model/GPU allocation."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from archlab.automodel.deepseek_v41_rl_model import (
    EXECUTION_CHANGES,
    RUNTIME_FIELDS,
    SOURCE_CHANGE_REASONS,
    _restore_local_weights,
    audit_parent_sources,
    construct_rl_actor,
    read_parent_checkpoint,
    validate_constructed_runtime,
)


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class RLModelTests(unittest.TestCase):
    def parent(self, root, family="scratch"):
        world = 8 if family == "scratch" else 16
        runtime = {
            "world_size": world,
            "ep_size": 8,
            "expert_fsdp_size": world // 8,
            "engram_owners": world,
        }
        contract = {
            "format": "archlab-v41-scratch-comparison-v1"
            if family == "scratch"
            else "archlab-v41-full-training-v1",
            "variant": "normal",
            "world_size": world,
            "tiny": False,
            "cpu_offload": False,
            "runtime": runtime,
            "implementation_sha256": {},
        }
        if family == "full":
            contract.update(runtime)
        marker = {
            "format": "archlab-v41-full-sharded-v1",
            "world_size": world,
            "cursor": {"step": 4537, "supervised_tokens": 1000000000},
            "contract": contract,
            "manifests": [f"rank-{rank:02d}/MANIFEST.json" for rank in range(world)],
        }
        for relative in marker["manifests"]:
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_text("{}")
        (root / "COMPLETE.json").write_text(json.dumps(marker))
        return marker

    def checkpoint(self, root):
        marker = self.parent(root)
        model = torch.nn.Linear(3, 2)
        model.register_buffer("counter", torch.tensor([17], dtype=torch.int64))
        expected = copy.deepcopy(model.state_dict())
        entries = []
        for number, (name, tensor) in enumerate(
            list(model.named_parameters()) + list(model.named_buffers())
        ):
            local = tensor.detach()
            entry = {
                "name": name,
                "shape": list(local.shape),
                "global_shape": list(local.shape),
                "dtype": str(local.dtype),
                "chunks": [],
            }
            for index, chunk in enumerate(local.flatten().split(2)):
                chunk = chunk.clone()
                filename = f"tensor-{number:04d}-{index:03d}.pt"
                torch.save(chunk, root / "rank-00" / filename)
                entry["chunks"].append(
                    {"file": filename, "elements": chunk.numel(), "sha256": tensor_sha(chunk)}
                )
            entries.append(entry)
        manifest = {
            "rank": 0,
            "world_size": marker["world_size"],
            "cursor": marker["cursor"],
            "contract": marker["contract"],
            "tensors": entries,
            "optimizer_states": ["optimizer-DO-NOT-READ.pt"],
        }
        (root / "rank-00/MANIFEST.json").write_text(json.dumps(manifest))
        with torch.no_grad():
            for tensor in list(model.parameters()) + list(model.buffers()):
                tensor.zero_()
        return model, marker, manifest, expected

    def test_mesh_and_variant_admission_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.parent(root)
            marker, receipt = read_parent_checkpoint(
                root, family="scratch", variant="normal", world_size=8
            )
            self.assertEqual(marker["cursor"]["step"], 4537)
            self.assertEqual(
                receipt["marker_sha256"],
                hashlib.sha256((root / "COMPLETE.json").read_bytes()).hexdigest(),
            )
            for kwargs in (
                {"family": "full", "variant": "normal", "world_size": 16},
                {"family": "scratch", "variant": "simplicial", "world_size": 8},
                {"family": "scratch", "variant": "normal", "world_size": 16},
            ):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    read_parent_checkpoint(root, **kwargs)

    def test_missing_rank_and_changed_ownership_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = self.parent(root, "full")
            marker["contract"]["runtime"]["engram_owners"] = 32
            (root / "COMPLETE.json").write_text(json.dumps(marker))
            with self.assertRaisesRegex(ValueError, "ownership"):
                read_parent_checkpoint(root, family="full", variant="normal", world_size=16)
            marker["contract"]["runtime"]["engram_owners"] = 16
            (root / "COMPLETE.json").write_text(json.dumps(marker))
            (root / "rank-15/MANIFEST.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                read_parent_checkpoint(root, family="full", variant="normal", world_size=16)

    def test_source_changes_require_actual_digests_and_allowlisted_reason(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = "automodel/deepseek_v41_official_adapter.py"
            path = root / relative
            path.parent.mkdir()
            old, new = b"old implementation", b"new padding support"
            path.write_bytes(new)
            before, after = hashlib.sha256(old).hexdigest(), hashlib.sha256(new).hexdigest()
            marker = {"contract": {"implementation_sha256": {relative: before}}}
            declaration = {
                relative: {
                    "before_sha256": before,
                    "after_sha256": after,
                    "reason": SOURCE_CHANGE_REASONS[relative],
                }
            }
            with self.assertRaises(ValueError):
                audit_parent_sources(marker, source_root=root)
            receipts = audit_parent_sources(
                marker, source_root=root, declared_source_changes=declaration
            )
            self.assertEqual(receipts[0]["current_sha256"], after)
            declaration[relative]["after_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                audit_parent_sources(marker, source_root=root, declared_source_changes=declaration)
            marker["contract"]["implementation_sha256"] = {relative: "dummy-hash"}
            with self.assertRaisesRegex(ValueError, "invalid source digest"):
                audit_parent_sources(marker, source_root=root)

    def test_runtime_packages_and_backend_cannot_be_waived(self):
        runtime = {key: key for key in RUNTIME_FIELDS}
        runtime.update(
            {
                "geometry": {"width": 640},
                "parameters": 100,
                "adapter_layers": [2, 4],
                "variant": "normal",
                "boundaries": {},
                "right_padding_masked": True,
                "router_auxiliary_loss_coefficient": 0.01,
                "sparse_precision": {"implementation": "deterministic"},
            }
        )
        validate_constructed_runtime({"runtime": runtime}, copy.deepcopy(runtime), family="scratch")
        for field in ("packages", "geometry", "sparse_precision", "automodel_commit", "parameters"):
            changed = copy.deepcopy(runtime)
            changed[field] = "different"
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_constructed_runtime({"runtime": runtime}, changed, family="scratch")

    def test_weight_only_restore_all_chunks_buffers_no_optimizer_or_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, marker, _, expected = self.checkpoint(root)
            with patch("torch.load", wraps=torch.load) as load:
                receipt = _restore_local_weights(model, root, marker, rank=0, expected_device="cpu")
            self.assertTrue(receipt["all_weight_checksums_verified"])
            self.assertFalse(receipt["optimizer_loaded"])
            self.assertFalse(receipt["rng_loaded"])
            self.assertGreater(receipt["verified_chunks"], receipt["tensor_entries"])
            self.assertTrue(
                all(Path(call.args[0]).name.startswith("tensor-") for call in load.call_args_list)
            )
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, expected[name]))

    def test_rank_contract_mismatch_rejected_without_payload_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, marker, manifest, _ = self.checkpoint(root)
            for field, value in (("world_size", 16), ("cursor", {"step": 0}), ("contract", {})):
                changed = copy.deepcopy(manifest)
                changed[field] = value
                (root / "rank-00/MANIFEST.json").write_text(json.dumps(changed))
                with (
                    self.subTest(field=field),
                    patch("torch.load") as load,
                    self.assertRaises(ValueError),
                ):
                    _restore_local_weights(model, root, marker, rank=0, expected_device="cpu")
                load.assert_not_called()

    def test_cpu_resident_actor_is_rejected_before_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, marker, _, _ = self.checkpoint(root)
            with patch("torch.load") as load, self.assertRaisesRegex(ValueError, "not on cuda"):
                _restore_local_weights(model, root, marker, rank=0, expected_device="cuda")
            load.assert_not_called()

    def test_tensor_schema_and_checksum_corruption_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, marker, manifest, _ = self.checkpoint(root)
            bad = copy.deepcopy(manifest)
            bad["tensors"][-1]["global_shape"] = [999]
            (root / "rank-00/MANIFEST.json").write_text(json.dumps(bad))
            with (
                patch("torch.load") as load,
                self.assertRaisesRegex(ValueError, "name/shape/dtype"),
            ):
                _restore_local_weights(model, root, marker, rank=0, expected_device="cpu")
            load.assert_not_called()
            (root / "rank-00/MANIFEST.json").write_text(json.dumps(manifest))
            chunk = manifest["tensors"][0]["chunks"][0]
            torch.save(
                torch.full((chunk["elements"],), 999.0, dtype=torch.float32),
                root / "rank-00" / chunk["file"],
            )
            with self.assertRaisesRegex(ValueError, "checksum"):
                _restore_local_weights(model, root, marker, rank=0, expected_device="cpu")

    def test_nonweight_payload_and_chunk_coverage_cannot_be_smuggled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, marker, manifest, _ = self.checkpoint(root)
            for name in ("../optimizer.pt", "rng.pt", "optimizer-0000.pt"):
                bad = copy.deepcopy(manifest)
                bad["tensors"][0]["chunks"][0]["file"] = name
                (root / "rank-00/MANIFEST.json").write_text(json.dumps(bad))
                with self.subTest(name=name), self.assertRaises(ValueError):
                    _restore_local_weights(model, root, marker, rank=0, expected_device="cpu")

    def test_production_without_parent_and_undeclared_changes_are_rejected(self):
        with patch("torch.distributed.get_world_size", return_value=8):
            with self.assertRaisesRegex(ValueError, "existing complete parent"):
                construct_rl_actor(
                    checkpoint=None,
                    family="scratch",
                    variant="normal",
                    assets="unused",
                    declared_execution_changes=EXECUTION_CHANGES,
                )
            with self.assertRaisesRegex(ValueError, "Declare exactly"):
                construct_rl_actor(
                    checkpoint=None, family="scratch", variant="normal", assets="unused", tiny=True
                )


if __name__ == "__main__":
    unittest.main()
