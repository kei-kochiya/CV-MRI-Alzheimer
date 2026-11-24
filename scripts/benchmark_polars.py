"""Benchmark Polars vs Pandas on the 44,000-image Alzheimer MRI manifest workload."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import pandas as pd
import polars as pl

from alzheimer_mri.data import (
    assign_stratified_splits,
    scan_dataset,
    summarize_manifest,
)


def bench_polars(records_df: pl.DataFrame, parquet_path: Path) -> dict[str, float]:
    t0 = time.perf_counter()
    split_df = assign_stratified_splits(records_df, seed=42)
    t_split = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = summarize_manifest(split_df)
    t_agg = time.perf_counter() - t0

    t0 = time.perf_counter()
    split_df.write_parquet(parquet_path)
    _ = pl.read_parquet(parquet_path)
    t_io = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = split_df.filter((pl.col("split") == "train") & (pl.col("file_bytes") > 5000))
    t_filter = time.perf_counter() - t0

    return {
        "split_ms": t_split * 1000,
        "agg_ms": t_agg * 1000,
        "parquet_io_ms": t_io * 1000,
        "filter_ms": t_filter * 1000,
        "total_ms": (t_split + t_agg + t_io + t_filter) * 1000,
    }


def bench_pandas(records_pd: pd.DataFrame, parquet_path: Path) -> dict[str, float]:
    t0 = time.perf_counter()
    shuffled = records_pd.sort_values("rel_path").sample(frac=1.0, random_state=42)
    cls_rank = shuffled.groupby("class_name").cumcount()
    cls_total = shuffled.groupby("class_name")["rel_path"].transform("count")
    split = pd.Series("test", index=shuffled.index)
    split[cls_rank < (cls_total * 0.70).astype(int)] = "train"
    split[
        (cls_rank >= (cls_total * 0.70).astype(int))
        & (cls_rank < (cls_total * 0.85).astype(int))
    ] = "val"
    split_df = shuffled.assign(split=split).sort_values("rel_path")
    t_split = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = (
        split_df.groupby(["split", "label_idx", "class_name"], as_index=False)
        .agg(count=("rel_path", "count"), avg_kb=("file_bytes", "mean"))
        .sort_values(["split", "label_idx"])
    )
    t_agg = time.perf_counter() - t0

    t0 = time.perf_counter()
    split_df.to_parquet(parquet_path, index=False)
    _ = pd.read_parquet(parquet_path)
    t_io = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = split_df[(split_df["split"] == "train") & (split_df["file_bytes"] > 5000)]
    t_filter = time.perf_counter() - t0

    return {
        "split_ms": t_split * 1000,
        "agg_ms": t_agg * 1000,
        "parquet_io_ms": t_io * 1000,
        "filter_ms": t_filter * 1000,
        "total_ms": (t_split + t_agg + t_io + t_filter) * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Polars vs Pandas on MRI manifest.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    args = parser.parse_args()

    t_scan = time.perf_counter()
    pl_df = scan_dataset(args.data_dir)
    scan_ms = (time.perf_counter() - t_scan) * 1000
    pd_df = pl_df.to_pandas()

    print(f"Scanned {pl_df.height:,} images in {scan_ms:.1f} ms")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Warmup once
        _ = bench_polars(pl_df, tmp / "warm_pl.parquet")
        _ = bench_pandas(pd_df, tmp / "warm_pd.parquet")

        pl_res = bench_polars(pl_df, tmp / "bench_pl.parquet")
        pd_res = bench_pandas(pd_df, tmp / "bench_pd.parquet")

    print("\nOperation        | Polars (ms) | Pandas (ms) | Speedup")
    print("-----------------+-------------+-------------+--------")
    for key, label in [
        ("split_ms", "Stratified split"),
        ("agg_ms", "Group-by summary"),
        ("parquet_io_ms", "Parquet W/R"),
        ("filter_ms", "Filter query"),
        ("total_ms", "Total pipeline"),
    ]:
        p_val = pl_res[key]
        d_val = pd_res[key]
        speedup = d_val / p_val if p_val > 0 else 0.0
        print(f"{label:<16} | {p_val:11.2f} | {d_val:11.2f} | {speedup:5.2f}x")


if __name__ == "__main__":
    main()
