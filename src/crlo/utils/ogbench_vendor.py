"""Acquire and verify the pinned upstream OGBench agent implementations."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

OGBENCH_COMMIT = "1d4140997f60c52c6fb0702ec100dc988b18c548"
DEFAULT_VENDOR = Path(__file__).resolve().parents[3] / "vendor" / "OGBench"


def verify_vendor(path: Path) -> None:
    """Reject absent, modified, or differently pinned upstream code."""
    if not (path / "impls/agents/crl.py").is_file():
        raise FileNotFoundError(
            f"Missing OGBench at {path}; run python -m crlo.utils.ogbench_vendor --install"
        )
    head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if head != OGBENCH_COMMIT or dirty:
        raise RuntimeError(f"OGBench must be unmodified at {OGBENCH_COMMIT}; found {head}")


def ensure_ogbench() -> None:
    """Expose the verified upstream import namespace exactly once."""
    path = Path(os.environ.get("CRLO_OGBENCH_VENDOR", DEFAULT_VENDOR)).resolve()
    impls = str(path / "impls")
    if impls not in sys.path:
        verify_vendor(path)
        sys.path.insert(0, impls)


def main(argv: list[str] | None = None) -> int:
    """Install only when explicitly requested, then verify the vendor checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_VENDOR)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    if args.install and not args.path.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/seohongpark/ogbench", str(args.path)], check=True
        )
        subprocess.run(["git", "-C", str(args.path), "checkout", OGBENCH_COMMIT], check=True)
    verify_vendor(args.path)
    print(f"Verified OGBench {OGBENCH_COMMIT} at {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
