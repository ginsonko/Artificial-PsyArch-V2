# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _check_import(module_name: str) -> dict[str, Any]:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "")
        return {"ok": True, "module": module_name, "version": str(version or "")}
    except Exception as exc:
        return {"ok": False, "module": module_name, "error": str(exc)}


def _check_pil_imagegrab() -> dict[str, Any]:
    try:
        from PIL import ImageGrab  # type: ignore

        return {"ok": True, "module": "PIL.ImageGrab", "available": hasattr(ImageGrab, "grab")}
    except Exception as exc:
        return {"ok": False, "module": "PIL.ImageGrab", "error": str(exc)}


def _check_pyautogui() -> dict[str, Any]:
    try:
        import pyautogui  # type: ignore

        size = pyautogui.size()
        return {"ok": True, "module": "pyautogui", "screen_size": {"width": int(size.width), "height": int(size.height)}}
    except Exception as exc:
        return {"ok": False, "module": "pyautogui", "error": str(exc)}


def _collect() -> dict[str, Any]:
    report = {
        "schema_id": "env_doctor/v1",
        "cwd": str(Path.cwd()),
        "repo_root": str(ROOT),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "env": {
            "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
            "pythonpath": os.environ.get("PYTHONPATH", ""),
        },
        "checks": {},
        "advice": [],
    }
    checks = report["checks"]
    checks["numpy"] = _check_import("numpy")
    checks["PIL"] = _check_import("PIL")
    checks["scipy"] = _check_import("scipy")
    checks["sklearn"] = _check_import("sklearn")
    checks["psutil"] = _check_import("psutil")
    checks["pyautogui"] = _check_pyautogui()
    checks["imagegrab"] = _check_pil_imagegrab()
    checks["cv2"] = _check_import("cv2")
    checks["faiss"] = _check_import("faiss")

    advice: list[str] = report["advice"]
    if not checks["cv2"]["ok"]:
        advice.append("若要运行真视频文件入口，请安装 `opencv-python` 或 `opencv-python-headless`。")
    if not checks["faiss"]["ok"]:
        advice.append("若要启用更强的 ANN 向量索引，请按平台安装 `faiss-cpu`。当前仍可退化到 numpy_flat。")
    if not checks["pyautogui"]["ok"]:
        advice.append("若要启用真实电脑控制，请确认图形桌面可用，并重新安装 `pyautogui` 相关依赖。")
    if not checks["imagegrab"]["ok"]:
        advice.append("若要启用截图感知或自主循环，请确保 Pillow 的 ImageGrab 可用，并在本地图形桌面运行。")
    if sys.version_info < (3, 11):
        advice.append("推荐使用 Python 3.11+，当前版本偏低。")
    if not advice:
        advice.append("核心环境看起来可用。你可以继续启动观测台、运行多模态样例、自主循环或视频流实验。")
    return report


def main() -> None:
    report = _collect()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
