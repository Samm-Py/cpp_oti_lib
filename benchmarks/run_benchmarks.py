#!/usr/bin/env python3
"""Build and run the otinum optimization isolation benchmarks and collect their
CSVs into one directory.

Each benchmark isolates one optimization and emits the shared tidy schema
(see bench_common.hpp). The arithmetic benchmark is a compile-time switch over
the whole otinum, so it is built once per path; its three binaries' rows are
concatenated into one file. The others toggle their variant internally.

Outputs (one CSV per benchmark, plus metadata.json):
  bench_arithmetic.csv  bench_alignment.csv  bench_layout.csv  bench_fused.csv
"""

import argparse
import datetime as dt
import json
import platform
import subprocess
from pathlib import Path


SINGLE = ["alignment", "layout", "fused"]
ARITH = {"naive": "bench_arithmetic_naive",
         "lookup": "bench_arithmetic_lookup",
         "unrolled": "bench_arithmetic_unrolled"}


def parse_args():
    here = Path(__file__).resolve().parent
    root = here.parent
    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", type=Path, default=root / "build-cuda")
    p.add_argument("--output", type=Path, default=here / "results")
    p.add_argument("--build", action="store_true")
    p.add_argument("--allow-non-cuda", action="store_true")
    return p.parse_args()


def build(build_dir, target):
    subprocess.run(["cmake", "--build", str(build_dir), "--target", target,
                    "--parallel"], check=True)


def run_binary(binary, extra):
    result = subprocess.run([str(binary), *extra], text=True,
                            capture_output=True, check=True)
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    header, rows = lines[0], lines[1:]
    return header, rows


def check_cuda(rows, allow):
    if allow:
        return
    for r in rows:
        if r.split(",", 1)[0] != "Cuda":
            raise SystemExit("expected Cuda backend; use --allow-non-cuda to override")


def main():
    args = parse_args()
    args.build_dir = args.build_dir.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    bench_dir = args.build_dir / "benchmarks"

    if args.build:
        for t in (list(ARITH.values()) + [f"bench_{b}" for b in SINGLE]):
            build(args.build_dir, t)

    written = {}

    # Arithmetic: three path binaries, one file.
    header, all_rows = None, []
    for variant, target in ARITH.items():
        binary = bench_dir / target
        if not binary.is_file():
            print(f"WARNING: missing {binary}; skipping {variant}")
            continue
        header, rows = run_binary(binary, [])
        check_cuda(rows, args.allow_non_cuda)
        all_rows.extend(rows)
        print(f"ran {target}: {len(rows)} rows")
    if header and all_rows:
        path = args.output / "bench_arithmetic.csv"
        path.write_text("\n".join([header, *all_rows]) + "\n")
        written["arithmetic"] = path.name

    # Single-binary benchmarks.
    for b in SINGLE:
        binary = bench_dir / f"bench_{b}"
        if not binary.is_file():
            print(f"WARNING: missing {binary}; skipping {b}")
            continue
        header, rows = run_binary(binary, [])
        check_cuda(rows, args.allow_non_cuda)
        path = args.output / f"bench_{b}.csv"
        path.write_text("\n".join([header, *rows]) + "\n")
        written[b] = path.name
        print(f"ran bench_{b}: {len(rows)} rows")

    root = Path(__file__).resolve().parents[1]
    metadata = {
        "collected_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "build_dir": str(args.build_dir),
        "files": written,
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            capture_output=True).stdout.strip(),
        "nvidia_smi": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True, capture_output=True).stdout.strip(),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {len(written)} CSVs to {args.output}")


if __name__ == "__main__":
    main()
