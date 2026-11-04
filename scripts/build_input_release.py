"""Build release/input_data_snapshot.tar.gz: the point-in-time inputs other than prices.

data/cache/gdelt_days (GDELT daily event counts) and data/cache/store/{alfred, blockchain_info, eia,
noaa, wikipedia} (first-release FRED/ALFRED vintages, blockchain.com charts, EIA stocks, NOAA degree
days, Wikimedia pageviews). Yahoo prices are not included. Per-folder SHA-256 hashes are recorded in
docs/run_manifest.json under input_data.
"""

from __future__ import annotations

import glob
import tarfile
from pathlib import Path

from scripts._common import REPO_ROOT
from scripts.build_manifest import INPUT_NAMESPACES

OUT = REPO_ROOT / "release" / "input_data_snapshot.tar.gz"


def main() -> None:
    files = sorted(glob.glob(str(REPO_ROOT / "data/cache/gdelt_days/*.json")))
    for ns in INPUT_NAMESPACES:
        files += sorted(glob.glob(str(REPO_ROOT / f"data/cache/store/{ns}/*.json")))
    with tarfile.open(OUT, "w:gz") as t:
        for f in files:
            t.add(f, arcname=str(Path(f).relative_to(REPO_ROOT)))
    print(f"{len(files)} files -> {OUT}")


if __name__ == "__main__":
    main()
