"""Keep an independent JUnit report for every distributed admission rank."""
import os
from pathlib import Path


def pytest_configure(config):
    if (os.environ.get("ARCHLAB_QUALIFICATION_CONFIG") and "RANK" in os.environ
            and config.option.xmlpath):
        path = Path(config.option.xmlpath)
        config.option.xmlpath = str(path.with_name(f"{path.stem}-rank-{os.environ['RANK']}{path.suffix}"))
        if config.option.basetemp:
            path = Path(config.option.basetemp)
            config.option.basetemp = str(path.with_name(f"{path.name}-rank-{os.environ['RANK']}"))
