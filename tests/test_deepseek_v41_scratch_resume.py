import copy
import unittest

from archlab.automodel.deepseek_v41_scratch_resume import (
    TILELANG_PARENT,
    maintenance_resume_changes,
)


class MaintenanceResumeTests(unittest.TestCase):
    def setUp(self):
        self.parent = {
            "project_commit": TILELANG_PARENT,
            "variant": "normal",
            "data_contract": {"sha256": "sealed"},
            "peak_relative_learning_rate": 0.01,
            "runtime": {"geometry": {"width": 640}, "sparse_precision": {"implementation": "v1"}},
            "implementation_sha256": {
                "architecture.py": "fixed",
                "automodel/deepseek_v41_scratch_training.py": "old",
            },
        }
        self.current = copy.deepcopy(self.parent)
        self.current["project_commit"] = "new"
        self.current["runtime"]["resolved_kernel_packages"] = {"tilelang": {"version": "0.1.9"}}
        self.current["runtime"]["sparse_precision"]["sink_gradient_reduction"] = "atomic-fp32"
        self.current["implementation_sha256"]["automodel/deepseek_v41_scratch_training.py"] = "new"
        self.current["implementation_sha256"]["automodel/deepseek_v41_sparse_qualification.py"] = (
            "probe"
        )

    def test_metadata_only_update_and_exact_resume(self):
        self.assertEqual(len(maintenance_resume_changes(self.parent, self.current)), 3)
        self.assertEqual(maintenance_resume_changes(self.parent, self.parent), [])

    def test_training_and_architecture_changes_are_rejected(self):
        for mutation in ("data", "variant", "learning_rate", "kernel", "hash", "inventory"):
            value = copy.deepcopy(self.current)
            if mutation == "data":
                value["data_contract"]["sha256"] = "different"
            if mutation == "variant":
                value["variant"] = "simplicial"
            if mutation == "learning_rate":
                value["peak_relative_learning_rate"] = 0.001
            if mutation == "kernel":
                value["runtime"]["sparse_precision"]["implementation"] = "v2"
            if mutation == "hash":
                value["implementation_sha256"]["architecture.py"] = "changed"
            if mutation == "inventory":
                value["implementation_sha256"]["surprise.py"] = "extra"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                maintenance_resume_changes(self.parent, value)

    def test_unknown_lineage_requires_other_explicit_admission(self):
        self.parent["project_commit"] = "unrecognized"
        self.assertIsNone(maintenance_resume_changes(self.parent, self.current))


if __name__ == "__main__":
    unittest.main()
