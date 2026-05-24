# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from observatory_v2.app import ObservatoryV2App
from observatory_v2.config import load_config


def main() -> None:
    app = ObservatoryV2App(config=load_config())
    payload = app.get_live_snapshot()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
