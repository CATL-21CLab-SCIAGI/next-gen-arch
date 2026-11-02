"""Check the exact release artifacts without importing the source checkout."""

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

REQUIRED = {
    "archlab/evaluation/capability.py",
    "archlab/reporting/paired_curves.py",
    "archlab/training/admission.py",
    "archlab/training/runtime.py",
    "archlab/architectures/tilelang_attention.py",
    "archlab/training/serving.py",
    "archlab/data/source-formatting-20260923.json",
    "archlab/prompts/capability_regression.yaml",
    "archlab/licenses/verl-Apache-2.0.txt",
}
FORBIDDEN = {"third_party", "Megatron-LM", "NeMo", "results", ".runtime", ".legacy", ".venv", "__pycache__"}


def check(directory: Path):
    wheels, sources = list(directory.glob("*.whl")), list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("expected exactly one wheel and one sdist")
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
        if missing := REQUIRED - names:
            raise ValueError(f"wheel is missing {sorted(missing)}")
        if any(name.startswith(("megatron/", "nemo/", "verl/")) for name in names):
            raise ValueError("wheel vendors a container/upstream dependency")
    with tarfile.open(sources[0]) as source:
        names.update(source.getnames())
    if any(FORBIDDEN.intersection(PurePosixPath(name).parts) for name in names):
        raise ValueError("distribution includes runtime, vendored source or run output")
    print(f"Checked {wheels[0].name} and {sources[0].name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    check(parser.parse_args().directory)
