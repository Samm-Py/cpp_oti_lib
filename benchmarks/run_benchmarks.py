#!/usr/bin/env python3
"""Build and run the otinum optimization isolation benchmarks and collect their
CSVs into one directory.

Each benchmark isolates one optimization and emits the shared tidy schema
(see bench_common.hpp). The arithmetic benchmark is a compile-time switch over
the whole otinum, so it is built once per path; its three binaries' rows are
concatenated into one file. The others toggle their variant internally.

Outputs (one CSV per benchmark, plus metadata.json):
  bench_arithmetic.csv  bench_alignment_source_update_gather.csv
  bench_layout.csv  bench_fused.csv
"""

import argparse
import datetime as dt
import json
import platform
import subprocess
from pathlib import Path


SINGLE = ["layout", "fused"]
ARITH = {"naive": "bench_arithmetic_naive",
         "lookup": "bench_arithmetic_lookup",
         "unrolled": "bench_arithmetic_unrolled"}
ALIGN_SOURCE_UPDATE_GATHER = {
    "natural": "bench_alignment_source_update_gather_natural",
    "aligned": "bench_alignment_source_update_gather_aligned",
}


def parse_args():
    here = Path(__file__).resolve().parent
    root = here.parent
    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", type=Path, default=root / "build-cuda")
    p.add_argument("--output", type=Path, default=here / "results")
    p.add_argument("--build", action="store_true")
    p.add_argument("--allow-non-cuda", action="store_true")
    p.add_argument("--runs", type=int, default=1,
                   help="relaunch each binary this many times and pool the rows; "
                        "separate processes capture cold-start/clock-state "
                        "variance that back-to-back repetitions do not. The "
                        "plotter then takes the median over all pooled rows.")
    p.add_argument("--nodes", type=int, default=None,
                   help="override the problem size (element/node count) for every "
                        "benchmark; forwarded to the binaries as --nodes.")
    p.add_argument("--repetitions", type=int, default=None,
                   help="per-process pooled sample count, forwarded to the "
                        "binaries as --repetitions. Distinct from --runs, which "
                        "relaunches the whole binary.")
    p.add_argument("--shapes", nargs="*", default=None, metavar="M,N",
                   help="restrict to these precompiled algebra shapes, e.g. "
                        "--shapes 1,1 2,1 3,1; forwarded to the binaries as "
                        "--shapes. Default: every compiled shape.")
    return p.parse_args()


def passthrough(args):
    """Extra CLI forwarded to each benchmark binary (--nodes / --shapes)."""
    extra = []
    if args.nodes is not None:
        extra += ["--nodes", str(args.nodes)]
    if args.repetitions is not None:
        extra += ["--repetitions", str(args.repetitions)]
    if args.shapes:
        extra += ["--shapes", *args.shapes]
    return extra


def build(build_dir, target):
    subprocess.run(["cmake", "--build", str(build_dir), "--target", target,
                    "--parallel"], check=True)


def run_binary(binary, extra, runs=1):
    header = None
    rows = []
    for _ in range(runs):
        result = subprocess.run([str(binary), *extra], text=True,
                                capture_output=True, check=True)
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        header = lines[0]
        rows.extend(lines[1:])
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
    extra = passthrough(args)

    if args.build:
        for t in (list(ARITH.values()) + list(ALIGN_SOURCE_UPDATE_GATHER.values()) +
                  [f"bench_{b}" for b in SINGLE]):
            build(args.build_dir, t)

    written = {}

    # Arithmetic: three path binaries, one file.
    header, all_rows = None, []
    for variant, target in ARITH.items():
        binary = bench_dir / target
        if not binary.is_file():
            print(f"WARNING: missing {binary}; skipping {variant}")
            continue
        header, rows = run_binary(binary, extra, args.runs)
        check_cuda(rows, args.allow_non_cuda)
        all_rows.extend(rows)
        print(f"ran {target}: {len(rows)} rows")
    if header and all_rows:
        path = args.output / "bench_arithmetic.csv"
        path.write_text("\n".join([header, *all_rows]) + "\n")
        written["arithmetic"] = path.name

    # Source/update/gather alignment: two real-otinum binaries, one file.
    header, all_rows = None, []
    for variant, target in ALIGN_SOURCE_UPDATE_GATHER.items():
        binary = bench_dir / target
        if not binary.is_file():
            print(f"WARNING: missing {binary}; skipping {variant}")
            continue
        header, rows = run_binary(binary, extra, args.runs)
        check_cuda(rows, args.allow_non_cuda)
        all_rows.extend(rows)
        print(f"ran {target}: {len(rows)} rows")
    if header and all_rows:
        path = args.output / "bench_alignment_source_update_gather.csv"
        path.write_text("\n".join([header, *all_rows]) + "\n")
        written["alignment_source_update_gather"] = path.name

    # Single-binary benchmarks.
    for b in SINGLE:
        binary = bench_dir / f"bench_{b}"
        if not binary.is_file():
            print(f"WARNING: missing {binary}; skipping {b}")
            continue
        header, rows = run_binary(binary, extra, args.runs)
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
        "runs": args.runs,
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
