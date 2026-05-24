# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from observatory_v2.app import ObservatoryV2App
from observatory_v2.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="AP 二期 Phase1 批量演示运行脚本")
    parser.add_argument("--runs", type=int, default=3, help="连续执行多少次最小 demo run")
    parser.add_argument("--ticks", type=int, default=8, help="每次 demo run 的 tick 数")
    parser.add_argument("--interval-ms", type=int, default=30, help="每个 tick 的等待毫秒数")
    args = parser.parse_args()

    app = ObservatoryV2App(config=load_config())
    rows = []
    for index in range(max(1, int(args.runs))):
        label = f"Phase1 批量演示运行 #{index + 1}"
        result = app.start_demo_run(tick_count=args.ticks, tick_interval_ms=args.interval_ms, label=label)
        app.wait_for_idle(timeout_sec=120.0)
        manifest = app.get_manifest(result["run_id"])
        rows.append(
            {
                "run_id": result["run_id"],
                "status": manifest.get("status", ""),
                "tick_done": manifest.get("tick_done", 0),
                "tick_planned": manifest.get("tick_planned", 0),
                "label": manifest.get("label", ""),
            }
        )
    print(json.dumps({"runs": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
