#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a clean Orion-Q share bundle.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: <orion-root>/build/orion_q_share",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest file. Default: tmp/Orion/scripts/orion_q_share_manifest.txt",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the existing output directory before copying.",
    )
    return parser.parse_args()


def read_manifest(manifest_path: Path) -> list[str]:
    patterns: list[str] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def expand_patterns(root: Path, patterns: list[str]) -> tuple[list[Path], list[str]]:
    matched: set[Path] = set()
    missing: list[str] = []
    for pattern in patterns:
        hits = [path for path in root.glob(pattern) if path.is_file()]
        if not hits:
            missing.append(pattern)
            continue
        matched.update(hits)
    return sorted(matched), missing


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_manifest(bundle_root: Path, copied_files: list[str], missing_patterns: list[str]) -> None:
    lines = [
        "# Orion-Q share bundle",
        "",
        "## Copied files",
        *copied_files,
    ]
    if missing_patterns:
        lines.extend([
            "",
            "## Manifest patterns with no matches",
            *missing_patterns,
        ])
    (bundle_root / "MANIFEST.generated.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()
    orion_root = script_path.parents[1]
    manifest_path = args.manifest or (orion_root / "scripts" / "orion_q_share_manifest.txt")
    output_root = args.output or (orion_root / "build" / "orion_q_share")
    bundle_root = output_root / "Orion-Q"

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    patterns = read_manifest(manifest_path)
    source_files, missing_patterns = expand_patterns(orion_root, patterns)

    copied: list[str] = []
    for src in source_files:
        rel = src.relative_to(orion_root)
        copy_file(src, bundle_root / rel)
        copied.append(str(rel))

    write_manifest(bundle_root, copied, missing_patterns)

    print(f"Prepared Orion-Q share bundle: {bundle_root}")
    print(f"Copied files: {len(copied)}")
    if missing_patterns:
        print(f"Unmatched manifest entries: {len(missing_patterns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
