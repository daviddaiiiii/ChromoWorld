#!/usr/bin/env python3
"""Estimate a gzip-compressed 2D tar from stratified real-file samples.

No dataset files are changed and no full archive is created. A deterministic
sample of each split/file-extension stratum is tarred and gzip-compressed in
memory; its ratio is applied to that stratum's exact logical tar payload.
"""
import argparse
import gzip
import io
import json
import random
import tarfile
from collections import defaultdict
from pathlib import Path


def estimate(root: Path, sample_budget: int, seed: int) -> dict:
    if sample_budget < 1:
        raise ValueError("sample budget must be positive")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    groups = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts):
            groups[(path.relative_to(root).parts[0], path.suffix.lower())].append(path)
    rows = []
    for (split, extension), paths in sorted(groups.items()):
        rng = random.Random(f"{seed}:{split}:{extension}")
        selected = list(paths)
        rng.shuffle(selected)
        sample = []
        sample_bytes = 0
        for path in selected:
            sample.append(path)
            sample_bytes += path.stat().st_size
            if sample_bytes >= sample_budget:
                break
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for path in sample:
                info = archive.gettarinfo(str(path), arcname=str(Path("2D") / path.relative_to(root)))
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with path.open("rb") as source:
                    archive.addfile(info, source)
        sampled_tar = buffer.getvalue()
        compressed = gzip.compress(sampled_tar, compresslevel=6, mtime=0)
        tar_payload = sum(512 + ((p.stat().st_size + 511) // 512) * 512 for p in paths)
        ratio = len(compressed) / max(len(sampled_tar), 1)
        rows.append({"split": split, "extension": extension, "files": len(paths),
                     "logical_bytes": sum(p.stat().st_size for p in paths),
                     "sample_files": len(sample), "sample_logical_bytes": sample_bytes,
                     "sample_tar_gzip_ratio": ratio,
                     "estimated_archive_bytes": round(tar_payload * ratio)})
    predicted = sum(row["estimated_archive_bytes"] for row in rows)
    return {"method": "stratified sampled USTAR + gzip level 6; estimate, not a full archive",
            "seed": seed, "sample_budget_per_stratum_bytes": sample_budget,
            "strata": rows, "logical_bytes": sum(row["logical_bytes"] for row in rows),
            "estimated_archive_bytes": predicted,
            "planning_range_bytes": [round(predicted * 0.95), round(predicted * 1.05)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "2D")
    parser.add_argument("--sample-mib", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = estimate(args.root, args.sample_mib * 1024 ** 2, args.seed)
    result["existing_3d_archives"] = [
        {"file": path.name, "bytes": path.stat().st_size}
        for path in sorted((args.root.parent / "3D").glob("*.tar.gz"))]
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)
