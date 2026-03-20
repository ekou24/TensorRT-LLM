#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Aggregate KV cache transceiver performance metrics.

Supports three input formats:
  1. Python transceiver CSV files (from TLLM_KV_TRANSFER_PERF_LOG_FILE)
  2. Python transceiver logger.info lines (from TLLM_ENABLE_CACHE_TRANSFER_PERF_INFO
     without TLLM_KV_TRANSFER_PERF_LOG_FILE, embedded in server logs)
  3. C++ transceiver CSV files (from TRTLLM_KVCACHE_TIME_OUTPUT_PATH)

Usage:
    # Python transceiver CSV files (recommended)
    python aggregate_kv_perf.py --python-csvs slurm_logs/kv_perf_python/kv_transfer_perf_*.csv

    # Python transceiver from server logs (fallback if no CSV file path was set)
    python aggregate_kv_perf.py --python-logs slurm_logs/disagg_perf_perf_python/3_output_CTX_0.log \\
                                              slurm_logs/disagg_perf_perf_python/3_output_GEN_0.log

    # C++ transceiver CSV files
    python aggregate_kv_perf.py --cpp-csvs slurm_logs/kv_perf_cpp/rank_*_send.csv \\
                                           slurm_logs/kv_perf_cpp/rank_*_recv.csv

    # Both side-by-side + export
    python aggregate_kv_perf.py \\
        --python-csvs slurm_logs/kv_perf_python/kv_transfer_perf_*.csv \\
        --cpp-csvs slurm_logs/kv_perf_cpp/rank_*.csv \\
        -o kv_perf_comparison.csv
"""

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install numpy")


# ---------------------------------------------------------------------------
# Python transceiver log parsing
# ---------------------------------------------------------------------------

# Matches: KVSendTask.print_perf_info: unique_rid=..., peer_rank=..., ...
_PY_SEND_RE = re.compile(
    r"(KVSendTask|AuxSendTask)\.print_perf_info:\s*"
    r"unique_rid=(?P<unique_rid>\d+),\s*"
    r"peer_rank=(?P<peer_rank>\d+),\s*"
    r"transfer_size=(?P<transfer_size>[\d.]+)\s*byte,\s*"
    r"avg_segment_size=(?P<avg_segment_size>[\d.]+)\s*byte,\s*"
    r"entry_count=(?P<entry_count>\d+),\s*"
    r"prepare_args_latency=(?P<prepare_args_latency>[\d.]+)\s*ms,\s*"
    r"queue_latency=(?P<queue_latency>[\d.]+)\s*ms,\s*"
    r"transfer_latency=(?P<transfer_latency>[\d.]+)\s*ms,\s*"
    r"task_latency=(?P<task_latency>[\d.]+)\s*ms,\s*"
    r"throughput=(?P<throughput>[\d.]+)\s*MB/s"
)

_PY_RECV_RE = re.compile(
    r"KVRecvTask\.print_perf_info:\s*"
    r"unique_rid=(?P<unique_rid>\d+),\s*"
    r"peer_rank=(?P<peer_rank>\d+),\s*"
    r"task_latency=(?P<task_latency>[\d.]+)\s*ms"
)


def parse_python_logs(log_files):
    """Parse Python transceiver perf lines from server log files.

    Returns dict of {task_type: [row_dict, ...]}
    """
    rows = defaultdict(list)
    for path in log_files:
        with open(path, errors='replace') as f:
            for line in f:
                m = _PY_SEND_RE.search(line)
                if m:
                    task_type = m.group(1)
                    row = {k: float(v) for k, v in m.groupdict().items()
                           if k not in ("unique_rid", "peer_rank", "entry_count")}
                    row["unique_rid"] = int(m.group("unique_rid"))
                    row["peer_rank"] = int(m.group("peer_rank"))
                    row["entry_count"] = int(m.group("entry_count"))
                    rows[task_type].append(row)
                    continue

                m = _PY_RECV_RE.search(line)
                if m:
                    row = {
                        "unique_rid": int(m.group("unique_rid")),
                        "peer_rank": int(m.group("peer_rank")),
                        "task_latency": float(m.group("task_latency")),
                    }
                    rows["KVRecvTask"].append(row)
    return rows


# ---------------------------------------------------------------------------
# Python transceiver CSV parsing (from TLLM_KV_TRANSFER_PERF_LOG_FILE)
# ---------------------------------------------------------------------------
# CSV format (header written by perf_logger.py):
#   timestamp,task_type,unique_rid,peer_rank,
#   transfer_size_bytes,avg_segment_size_bytes,transfer_entry_count,
#   prepare_args_latency_ms,queue_latency_ms,transfer_latency_ms,task_latency_ms,throughput_mbs
# Data rows have timestamp prepended by the logging formatter:
#   2026-03-13 14:22:16.123,KVSendTask,42,0,1234567,1234567,1,0.523,1.234,5.678,8.901,207.45

def parse_python_csvs(csv_files):
    """Parse Python transceiver CSV files produced by TLLM_KV_TRANSFER_PERF_LOG_FILE.

    Returns dict of {task_type: [row_dict, ...]}
    """
    rows = defaultdict(list)
    for path in csv_files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("timestamp,"):
                    continue

                parts = line.split(",")
                # timestamp is "YYYY-MM-DD HH:MM:SS.mmm" which contains a comma
                # between date and time — but the formatter uses
                # "%(asctime)s.%(msecs)03d,%(message)s" with datefmt="%Y-%m-%d %H:%M:%S"
                # so the timestamp is "2026-03-13 14:22:16.123" (no comma inside).
                # The full line is: timestamp,task_type,unique_rid,peer_rank,...
                # We need at least 2 fields to get task_type.
                if len(parts) < 3:
                    continue

                task_type = parts[1].strip()
                if task_type in ("KVSendTask", "AuxSendTask"):
                    if len(parts) < 12:
                        continue
                    row = {
                        "unique_rid": int(parts[2]),
                        "peer_rank": int(parts[3]),
                        "transfer_size": _safe_float(parts[4]),
                        "avg_segment_size": _safe_float(parts[5]),
                        "entry_count": int(_safe_float(parts[6])),
                        "prepare_args_latency": _safe_float(parts[7]),
                        "queue_latency": _safe_float(parts[8]),
                        "transfer_latency": _safe_float(parts[9]),
                        "task_latency": _safe_float(parts[10]),
                        "throughput": _safe_float(parts[11]),
                    }
                    rows[task_type].append(row)
                elif task_type == "KVRecvTask":
                    # KVRecvTask CSV: KVRecvTask,unique_rid,peer_rank,,,,,,,task_latency,
                    if len(parts) < 11:
                        continue
                    row = {
                        "unique_rid": int(parts[2]),
                        "peer_rank": int(parts[3]),
                        "task_latency": _safe_float(parts[10]),
                    }
                    rows["KVRecvTask"].append(row)

    return rows


# ---------------------------------------------------------------------------
# C++ transceiver CSV parsing
# ---------------------------------------------------------------------------

def parse_cpp_csvs(csv_files):
    """Parse C++ transceiver CSV files (rank_N_send.csv / rank_N_recv.csv).

    Returns dict of {side: [row_dict, ...]} where side is 'send' or 'recv'.
    The variable-length Delay/Duration/Bandwidth triplets are aggregated per
    row into summary fields.
    """
    rows = defaultdict(list)
    for path in csv_files:
        p = Path(path)
        # Determine send vs recv from filename
        if "_send" in p.stem:
            side = "send"
        elif "_recv" in p.stem:
            side = "recv"
        else:
            side = "unknown"

        with open(path) as f:
            header_line = f.readline().strip()
            headers = header_line.split(",")

            # Fixed columns: RequestID, RequestInfo, Preparation, Preprocess,
            #                 Transmissions, Postprocess
            # Then variable triplets: Delay, Duration, Bandwidth(Gbps), ...
            n_fixed = 6

            for line in f:
                vals = line.strip().split(",")
                if len(vals) < n_fixed:
                    continue

                row = {
                    "request_id": int(_safe_float(vals[0])),
                    "request_info_ms": _safe_float(vals[1]),
                    "preparation_ms": _safe_float(vals[2]),
                    "preprocess_ms": _safe_float(vals[3]),
                    "transmissions_ms": _safe_float(vals[4]),
                    "postprocess_ms": _safe_float(vals[5]),
                }

                # Total of the fixed phases
                row["total_ms"] = (
                    row["request_info_ms"]
                    + row["preparation_ms"]
                    + row["preprocess_ms"]
                    + row["transmissions_ms"]
                    + row["postprocess_ms"]
                )

                # Per-chunk metrics — compute chunk count per row since
                # rows can have different numbers of chunks
                n_chunks_row = (len(vals) - n_fixed) // 3
                chunk_delays = []
                chunk_durations = []
                chunk_bandwidths = []
                for i in range(n_chunks_row):
                    base = n_fixed + i * 3
                    if base + 2 < len(vals):
                        chunk_delays.append(_safe_float(vals[base]))
                        chunk_durations.append(_safe_float(vals[base + 1]))
                        chunk_bandwidths.append(_safe_float(vals[base + 2]))

                row["num_chunks"] = len(chunk_durations)
                if chunk_durations:
                    row["total_chunk_duration_ms"] = sum(chunk_durations)
                    row["total_chunk_delay_ms"] = sum(chunk_delays)
                    row["mean_chunk_bandwidth_gbps"] = np.mean(chunk_bandwidths)
                    row["min_chunk_bandwidth_gbps"] = min(chunk_bandwidths)
                    row["max_chunk_bandwidth_gbps"] = max(chunk_bandwidths)
                else:
                    row["total_chunk_duration_ms"] = 0.0
                    row["total_chunk_delay_ms"] = 0.0
                    row["mean_chunk_bandwidth_gbps"] = 0.0
                    row["min_chunk_bandwidth_gbps"] = 0.0
                    row["max_chunk_bandwidth_gbps"] = 0.0

                rows[side].append(row)

    return rows


def _safe_float(s):
    try:
        v = float(s)
        if not math.isfinite(v):
            return 0.0
        return v
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_stats(values):
    """Compute summary statistics for a list of numeric values."""
    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    return {
        "count": len(arr),
        "mean": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "p90": np.percentile(arr, 90),
        "p95": np.percentile(arr, 95),
        "p99": np.percentile(arr, 99),
    }


_STAT_NAMES = ["count", "mean", "median", "std", "min", "p90", "p95", "p99", "max"]


def print_stats_table(title, metrics_dict):
    """Pretty-print a table of aggregated stats."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    if not metrics_dict:
        print("  (no data)")
        return

    # Header
    header = f"  {'Metric':<32s}"
    for s in _STAT_NAMES:
        if s == "count":
            header += f"{'count':>8s}"
        else:
            header += f"{s:>10s}"
    print(header)
    print(f"  {'-' * (32 + 8 + 10 * (len(_STAT_NAMES) - 1))}")

    for metric_name, stats in metrics_dict.items():
        if not stats:
            continue
        row = f"  {metric_name:<32s}"
        for s in _STAT_NAMES:
            v = stats.get(s, 0)
            if s == "count":
                row += f"{int(v):>8d}"
            else:
                row += f"{v:>10.2f}"
        print(row)


def aggregate_python(rows):
    """Aggregate Python transceiver metrics."""
    results = {}

    # KVSendTask
    send_rows = rows.get("KVSendTask", [])
    if send_rows:
        metrics = {
            "prepare_args_latency (ms)": compute_stats([r["prepare_args_latency"] for r in send_rows]),
            "queue_latency (ms)": compute_stats([r["queue_latency"] for r in send_rows]),
            "transfer_latency (ms)": compute_stats([r["transfer_latency"] for r in send_rows]),
            "task_latency (ms)": compute_stats([r["task_latency"] for r in send_rows]),
            "throughput (MB/s)": compute_stats([r["throughput"] for r in send_rows]),
            "transfer_size (bytes)": compute_stats([r["transfer_size"] for r in send_rows]),
            "avg_segment_size (bytes)": compute_stats([r["avg_segment_size"] for r in send_rows]),
            "entry_count": compute_stats([r["entry_count"] for r in send_rows]),
        }
        results["Python KVSendTask"] = metrics

    # AuxSendTask
    aux_rows = rows.get("AuxSendTask", [])
    if aux_rows:
        metrics = {
            "prepare_args_latency (ms)": compute_stats([r["prepare_args_latency"] for r in aux_rows]),
            "queue_latency (ms)": compute_stats([r["queue_latency"] for r in aux_rows]),
            "transfer_latency (ms)": compute_stats([r["transfer_latency"] for r in aux_rows]),
            "task_latency (ms)": compute_stats([r["task_latency"] for r in aux_rows]),
            "throughput (MB/s)": compute_stats([r["throughput"] for r in aux_rows]),
            "transfer_size (bytes)": compute_stats([r["transfer_size"] for r in aux_rows]),
        }
        results["Python AuxSendTask"] = metrics

    # KVRecvTask
    recv_rows = rows.get("KVRecvTask", [])
    if recv_rows:
        metrics = {
            "task_latency (ms)": compute_stats([r["task_latency"] for r in recv_rows]),
        }
        results["Python KVRecvTask"] = metrics

    return results


def aggregate_cpp(rows):
    """Aggregate C++ transceiver metrics."""
    results = {}

    for side in ("send", "recv"):
        side_rows = rows.get(side, [])
        if not side_rows:
            continue

        metrics = {
            "request_info (ms)": compute_stats([r["request_info_ms"] for r in side_rows]),
            "preparation (ms)": compute_stats([r["preparation_ms"] for r in side_rows]),
            "preprocess (ms)": compute_stats([r["preprocess_ms"] for r in side_rows]),
            "transmissions (ms)": compute_stats([r["transmissions_ms"] for r in side_rows]),
            "postprocess (ms)": compute_stats([r["postprocess_ms"] for r in side_rows]),
            "total (ms)": compute_stats([r["total_ms"] for r in side_rows]),
        }

        # Chunk-level metrics
        chunk_rows = [r for r in side_rows if r["num_chunks"] > 0]
        if chunk_rows:
            metrics["chunk_duration_total (ms)"] = compute_stats(
                [r["total_chunk_duration_ms"] for r in chunk_rows]
            )
            metrics["chunk_delay_total (ms)"] = compute_stats(
                [r["total_chunk_delay_ms"] for r in chunk_rows]
            )
            metrics["chunk_bandwidth_mean (Gbps)"] = compute_stats(
                [r["mean_chunk_bandwidth_gbps"] for r in chunk_rows]
            )
            metrics["chunk_bandwidth_min (Gbps)"] = compute_stats(
                [r["min_chunk_bandwidth_gbps"] for r in chunk_rows]
            )
            metrics["chunk_bandwidth_max (Gbps)"] = compute_stats(
                [r["max_chunk_bandwidth_gbps"] for r in chunk_rows]
            )

        results[f"C++ {side}"] = metrics

    return results


def write_csv_output(all_results, output_path):
    """Write all aggregated results to a single CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "metric"] + _STAT_NAMES)
        for section, metrics in all_results.items():
            for metric_name, stats in metrics.items():
                if not stats:
                    continue
                row = [section, metric_name]
                for s in _STAT_NAMES:
                    row.append(f"{stats.get(s, 0):.4f}")
                writer.writerow(row)
    print(f"\nCSV written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate KV cache transceiver performance metrics."
    )
    parser.add_argument(
        "--python-csvs",
        nargs="+",
        metavar="FILE",
        help="Python transceiver CSV files from TLLM_KV_TRANSFER_PERF_LOG_FILE "
             "(e.g. kv_transfer_perf_*.csv)",
    )
    parser.add_argument(
        "--python-logs",
        nargs="+",
        metavar="FILE",
        help="Python transceiver server log files with logger.info output "
             "(e.g. 3_output_CTX_0.log 3_output_GEN_0.log)",
    )
    parser.add_argument(
        "--cpp-csvs",
        nargs="+",
        metavar="FILE",
        help="C++ transceiver CSV files from TRTLLM_KVCACHE_TIME_OUTPUT_PATH "
             "(e.g. rank_0_send.csv rank_0_recv.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Optional: write aggregated results to CSV file",
    )
    args = parser.parse_args()

    if not args.python_csvs and not args.python_logs and not args.cpp_csvs:
        parser.error("Provide at least one of --python-csvs, --python-logs, or --cpp-csvs")

    all_results = {}

    # Merge Python rows from both CSV and log sources before aggregating
    py_rows = defaultdict(list)

    if args.python_csvs:
        print(f"Parsing Python CSVs: {args.python_csvs}")
        csv_rows = parse_python_csvs(args.python_csvs)
        total = sum(len(v) for v in csv_rows.values())
        print(f"  Parsed {total} records "
              f"({len(csv_rows.get('KVSendTask', []))} KVSend, "
              f"{len(csv_rows.get('AuxSendTask', []))} AuxSend, "
              f"{len(csv_rows.get('KVRecvTask', []))} KVRecv)")
        for k, v in csv_rows.items():
            py_rows[k].extend(v)

    if args.python_logs:
        print(f"Parsing Python logs: {args.python_logs}")
        log_rows = parse_python_logs(args.python_logs)
        total = sum(len(v) for v in log_rows.values())
        print(f"  Parsed {total} records "
              f"({len(log_rows.get('KVSendTask', []))} KVSend, "
              f"{len(log_rows.get('AuxSendTask', []))} AuxSend, "
              f"{len(log_rows.get('KVRecvTask', []))} KVRecv)")
        for k, v in log_rows.items():
            py_rows[k].extend(v)

    if py_rows:
        py_agg = aggregate_python(py_rows)
        all_results.update(py_agg)

    if args.cpp_csvs:
        print(f"Parsing C++ CSVs: {args.cpp_csvs}")
        cpp_rows = parse_cpp_csvs(args.cpp_csvs)
        total = sum(len(v) for v in cpp_rows.values())
        print(f"  Parsed {total} records "
              f"({len(cpp_rows.get('send', []))} send, "
              f"{len(cpp_rows.get('recv', []))} recv)")
        cpp_agg = aggregate_cpp(cpp_rows)
        all_results.update(cpp_agg)

    # Print tables
    for section, metrics in all_results.items():
        print_stats_table(section, metrics)

    # Optional CSV output
    if args.output:
        write_csv_output(all_results, args.output)

    print()


if __name__ == "__main__":
    main()
