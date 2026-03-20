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
"""Aggregate KV cache transceiver metrics per config pair (python vs cpp).

Produces a single CSV where each row is a config, with columns for both
Python and C++ transceiver metrics side by side.

Usage:
    poetry run python aggregate_kv_perf_per_config.py \
        --slurm-logs-dir /path/to/slurm_logs \
        -o kv_transfer_per_config.csv
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

# Reuse parsers from aggregate_kv_perf.py
from aggregate_kv_perf import (
    compute_stats,
    parse_cpp_csvs,
    parse_python_logs,
)

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install numpy")


def find_config_pairs(slurm_dir):
    """Find all python/cpp config pairs in slurm_logs directory.

    Returns dict of {base_config: {"python": dir_path, "cpp": dir_path}}
    """
    pairs = defaultdict(dict)
    for entry in sorted(os.listdir(slurm_dir)):
        full_path = os.path.join(slurm_dir, entry)
        if not os.path.isdir(full_path):
            continue
        if entry.endswith("_ERROR"):
            continue

        if "-python" in entry:
            base = entry.replace("disagg_perf_", "").replace("-python", "")
            pairs[base]["python"] = full_path
        elif "-cpp" in entry:
            base = entry.replace("disagg_perf_", "").replace("-cpp", "")
            pairs[base]["cpp"] = full_path

    return pairs


def get_python_metrics(job_dir):
    """Extract KV transfer metrics from python transceiver logs."""
    log_files = sorted(glob.glob(os.path.join(job_dir, "3_output_*.log")))
    if not log_files:
        return None

    rows = parse_python_logs(log_files)
    send_rows = rows.get("KVSendTask", [])
    if not send_rows:
        return None

    return {
        "count": len(send_rows),
        "transfer_latency": compute_stats(
            [r["transfer_latency"] for r in send_rows]
        ),
        "task_latency": compute_stats(
            [r["task_latency"] for r in send_rows]
        ),
        "prepare_args_latency": compute_stats(
            [r["prepare_args_latency"] for r in send_rows]
        ),
        "queue_latency": compute_stats(
            [r["queue_latency"] for r in send_rows]
        ),
        "throughput": compute_stats(
            [r["throughput"] for r in send_rows]
        ),
        "transfer_size": compute_stats(
            [r["transfer_size"] for r in send_rows]
        ),
    }


def get_cpp_metrics(job_dir):
    """Extract KV transfer metrics from C++ transceiver CSVs."""
    csv_files = sorted(
        glob.glob(os.path.join(job_dir, "kv_perf", "rank_*_send.csv"))
    )
    if not csv_files:
        return None

    rows = parse_cpp_csvs(csv_files)
    send_rows = rows.get("send", [])
    if not send_rows:
        return None

    return {
        "count": len(send_rows),
        "transmissions": compute_stats(
            [r["transmissions_ms"] for r in send_rows]
        ),
        "total": compute_stats([r["total_ms"] for r in send_rows]),
        "preparation": compute_stats(
            [r["preparation_ms"] for r in send_rows]
        ),
        "preprocess": compute_stats(
            [r["preprocess_ms"] for r in send_rows]
        ),
        "postprocess": compute_stats(
            [r["postprocess_ms"] for r in send_rows]
        ),
        "bandwidth_gbps": compute_stats(
            [r["mean_chunk_bandwidth_gbps"] for r in send_rows]
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate KV transceiver metrics per config pair."
    )
    parser.add_argument(
        "--slurm-logs-dir",
        required=True,
        help="Path to slurm_logs directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="kv_transfer_per_config.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    pairs = find_config_pairs(args.slurm_logs_dir)
    print(f"Found {len(pairs)} config bases")

    # Only keep pairs that have both python and cpp
    complete = {
        k: v for k, v in pairs.items() if "python" in v and "cpp" in v
    }
    print(f"Complete pairs (both python + cpp): {len(complete)}")

    # CSV columns
    header = [
        "config",
        # Python KVSendTask
        "py_send_count",
        "py_transfer_latency_median_ms",
        "py_transfer_latency_p99_ms",
        "py_task_latency_median_ms",
        "py_task_latency_p99_ms",
        "py_prepare_args_median_ms",
        "py_queue_latency_median_ms",
        "py_throughput_median_mbs",
        "py_transfer_size_median_bytes",
        # C++ send
        "cpp_send_count",
        "cpp_transmissions_median_ms",
        "cpp_transmissions_p99_ms",
        "cpp_total_median_ms",
        "cpp_total_p99_ms",
        "cpp_preparation_median_ms",
        "cpp_preprocess_median_ms",
        "cpp_postprocess_median_ms",
        "cpp_bandwidth_median_gbps",
    ]

    rows = []
    for base in sorted(complete.keys()):
        dirs = complete[base]
        print(f"  Processing: {base}")

        py = get_python_metrics(dirs["python"])
        cpp = get_cpp_metrics(dirs["cpp"])

        row = [base]

        # Python metrics
        if py:
            row.extend([
                py["count"],
                f"{py['transfer_latency'].get('median', 0):.3f}",
                f"{py['transfer_latency'].get('p99', 0):.3f}",
                f"{py['task_latency'].get('median', 0):.3f}",
                f"{py['task_latency'].get('p99', 0):.3f}",
                f"{py['prepare_args_latency'].get('median', 0):.3f}",
                f"{py['queue_latency'].get('median', 0):.3f}",
                f"{py['throughput'].get('median', 0):.2f}",
                f"{py['transfer_size'].get('median', 0):.0f}",
            ])
        else:
            row.extend([""] * 9)

        # C++ metrics
        if cpp:
            row.extend([
                cpp["count"],
                f"{cpp['transmissions'].get('median', 0):.3f}",
                f"{cpp['transmissions'].get('p99', 0):.3f}",
                f"{cpp['total'].get('median', 0):.3f}",
                f"{cpp['total'].get('p99', 0):.3f}",
                f"{cpp['preparation'].get('median', 0):.3f}",
                f"{cpp['preprocess'].get('median', 0):.3f}",
                f"{cpp['postprocess'].get('median', 0):.3f}",
                f"{cpp['bandwidth_gbps'].get('median', 0):.2f}",
            ])
        else:
            row.extend([""] * 9)

        rows.append(row)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nWritten {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
