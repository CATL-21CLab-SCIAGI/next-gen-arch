import os
import tempfile
from pathlib import Path

# Importing the historical dataset module resolves its cache root.  Keep test
# collection hermetic instead of touching a developer's default user cache.
os.environ.setdefault(
    "NANOCHAT_BASE_DIR",
    str(Path(tempfile.gettempdir()) / "next-gen-arch-tests"),
)
