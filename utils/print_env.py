#!/usr/bin/env python3
"""
Print dependency versions used by SLURM job scripts.
"""
import os
import sys
import torch
import argparse
import platform
from importlib import import_module
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.numpy_compat import patch_numpy_core_multiarray

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
# =============================================================================
# 硬件环境检测
# =============================================================================


def get_cpu_info() -> dict:
    """
    获取 CPU 型号和核心数等信息.

    Returns:
        {"model": str, "cores_physical": int, "cores_logical": int}
    """
    cpu_model = "Unknown"
    try:
        # 从 /proc/cpuinfo 读取 CPU 型号 (Linux)
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    cpu_model = line.split(":")[1].strip()
                    break
    except Exception:
        cpu_model = platform.processor() or "Unknown"

    cores_logical = os.cpu_count() or 0
    cores_physical = cores_logical  # 默认值
    try:
        import psutil

        cores_physical = psutil.cpu_count(logical=False) or cores_logical
    except ImportError:
        # 尝试从 /proc/cpuinfo 计算物理核心数
        try:
            with open("/proc/cpuinfo", "r") as f:
                physical_ids = set()
                for line in f:
                    if line.strip().startswith("physical id"):
                        physical_ids.add(line.split(":")[1].strip())
                # 粗略估计: 逻辑核心 / 超线程系数
                cores_physical = (
                    cores_logical // max(1, len(physical_ids) * 2) * len(physical_ids)
                    if physical_ids
                    else cores_logical
                )
        except Exception:
            pass

    return {
        "model": cpu_model,
        "cores_physical": cores_physical,
        "cores_logical": cores_logical,
    }


def get_memory_info() -> dict:
    """
    获取系统内存信息 (总量/已用/可用).

    Returns:
        {"total_gb": float, "available_gb": float, "used_gb": float}
    """
    try:
        import psutil

        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "used_gb": round(mem.used / (1024**3), 2),
        }
    except ImportError:
        # 从 /proc/meminfo 读取
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip().split()[0]  # 单位 kB
                        meminfo[key] = int(val)
                total = meminfo.get("MemTotal", 0) / (1024**2)
                available = meminfo.get("MemAvailable", 0) / (1024**2)
                return {
                    "total_gb": round(total, 2),
                    "available_gb": round(available, 2),
                    "used_gb": round(total - available, 2),
                }
        except Exception:
            return {"total_gb": 0, "available_gb": 0, "used_gb": 0}


def get_gpu_info() -> list:
    """
    获取所有 GPU 的型号、显存等信息.

    Returns:
        list of {"index": int, "name": str, "memory_total_gb": float}
    """
    gpus = []
    if not torch.cuda.is_available():
        return gpus

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mem = getattr(props, "total_memory", getattr(props, "total_mem", 0))
        gpus.append(
            {
                "index": i,
                "name": props.name,
                "memory_total_gb": round(total_mem / (1024**3), 2),
            }
        )
    return gpus


def get_hardware_summary() -> dict:
    """
    汇总所有硬件信息.

    Returns:
        完整的硬件信息字典.
    """
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "gpus": get_gpu_info(),
    }


def print_hardware_table(hw_info):
    """使用 pandas 打印硬件信息表格."""
    if HAS_PANDAS:
        rows = [
            ("Platform", hw_info["platform"]),
            ("Python", hw_info["python_version"]),
            ("PyTorch", hw_info["pytorch_version"]),
            ("CUDA Available", str(hw_info["cuda_available"])),
            ("CUDA Version", hw_info["cuda_version"]),
            ("CPU Model", hw_info["cpu"]["model"]),
            (
                "CPU Cores (physical/logical)",
                f'{hw_info["cpu"]["cores_physical"]}/{hw_info["cpu"]["cores_logical"]}',
            ),
            ("System Memory (total)", f'{hw_info["memory"]["total_gb"]:.1f} GB'),
            (
                "System Memory (available)",
                f'{hw_info["memory"]["available_gb"]:.1f} GB',
            ),
        ]
        for gpu in hw_info["gpus"]:
            rows.append(
                (
                    f"GPU {gpu['index']}",
                    f'{gpu["name"]} ({gpu["memory_total_gb"]:.1f} GB)',
                )
            )

        df = pd.DataFrame(rows, columns=["Item", "Value"]).set_index("Item")
        print("--- Hardware ---")
        print(df.to_string())
        print()
    else:
        print("--- Hardware ---")
        print(f"Platform:    {hw_info['platform']}")
        print(f"Python:      {hw_info['python_version']}")
        print(f"PyTorch:     {hw_info['pytorch_version']}")
        print(f"CUDA:        {hw_info['cuda_version']}")
        print(f"CPU:         {hw_info['cpu']['model']}")
        print(f"Memory:      {hw_info['memory']['total_gb']:.1f} GB")
        for gpu in hw_info["gpus"]:
            print(
                f"  GPU {gpu['index']}:      {gpu['name']} ({gpu['memory_total_gb']:.1f} GB)"
            )
        print()


def get_version(module_name: str) -> tuple[str, str | None]:
    """Return (version, error_message). Never raises import-time exceptions."""
    if module_name in {"accelerate", "transformers"}:
        patch_numpy_core_multiarray()
    try:
        module = import_module(module_name)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        return "IMPORT FAILED", err
    return getattr(module, "__version__", "unknown"), None


def print_required_versions(include_scipy: bool):
    required = [
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
        ("sklearn", "sklearn"),
    ]
    if include_scipy:
        required.append(("scipy", "scipy"))

    print()
    for display_name, module_name in required:
        version, err = get_version(module_name)
        print(f"{display_name:<12}: {version}")
        if err is not None:
            print(f"{'':<12}  -> {err}")


def print_optional_versions():
    pandas_version, err = get_version("pandas")
    if err is None:
        print(f"{'pandas':<12}: {pandas_version}")
    else:
        print(f"{'pandas':<12}: {pandas_version} (评估表格将使用简单格式)")
        print(f"{'':<12}  -> {err}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Print core dependency versions.")
    parser.add_argument(
        "--include-scipy",
        action="store_true",
        help="Also require and print scipy version.",
    )
    args = parser.parse_args()

    hw_info = get_hardware_summary()
    print_hardware_table(hw_info)

    print("--- Software ---")
    print(f"Python:      {sys.version.split()[0]}")
    print(f"PyTorch:     {torch.__version__}")
    print(f"CUDA avail:  {torch.cuda.is_available()}")
    print(f"GPU count:   {torch.cuda.device_count()}")

    print_required_versions(include_scipy=args.include_scipy)
    print_optional_versions()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
