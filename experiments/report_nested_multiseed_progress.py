#!/usr/bin/env python3
"""Compact status reporter for the strict nested-window batch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


TASKS = [(model, "Full_Preprocessing", seed) for model in ("AttentionNet", "CNNLSTM", "PureLSTM") for seed in range(42, 47)]
TASKS += [("AttentionNet", "No_Filtering", seed) for seed in range(42, 47)]


def alive(path: Path) -> bool:
    try:
        pid = int(path.read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError, FileNotFoundError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    root = Path(args.output_root).resolve()

    while True:
        rows = []
        complete = failed = running = pending = stale = 0
        for model, config, seed in TASKS:
            directory = root / model / config / f"seed_{seed}"
            progress_path = directory / "progress.json"
            value = {}
            if progress_path.exists():
                try:
                    value = json.loads(progress_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    value = {"status": "unreadable"}
            status = value.get("status", "pending")
            if (directory / "COMPLETE").exists():
                status = "complete"
            elif (directory / "failure.json").exists():
                status = "failed"
            elif status == "running" and not alive(directory / "task.pid"):
                status = "stale"
            if status == "complete": complete += 1
            elif status == "failed": failed += 1
            elif status == "running": running += 1
            elif status == "stale": stale += 1
            else: pending += 1
            rows.append((model, config.replace("_Preprocessing", ""), seed, status,
                         value.get("completed_folds", 0), value.get("total_folds", 57),
                         value.get("current_fold", "-"), value.get("candidate_index", "-"),
                         value.get("current_epoch", "-"), value.get("stage", "-"),
                         value.get("updated_at", "-")))
        print("\033[2J\033[H", end="")
        print(f"Nested strict batch: complete={complete}/20 running={running} pending={pending} failed={failed} stale={stale}")
        print("model       config       seed status    folds       fold cand epoch stage                         updated")
        print("----------- ------------ ---- --------- ----------- ---- ---- ----- ---------------------------- -------------------")
        for row in rows:
            print(f"{row[0]:11s} {row[1]:12s} {row[2]:4d} {row[3]:9s} {row[4]:3}/{row[5]:<3}       {str(row[6]):>3} {str(row[7]):>4} {str(row[8]):>5} {str(row[9]):28s} {row[10]}")
        if args.once:
            return 0
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
