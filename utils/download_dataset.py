#!/usr/bin/env python3
"""
download_dataset.py - 数据集下载脚本
===================================

从 HuggingFace Hub 下载训练/评估数据集到本地存储.

下载列表和目标路径在 config.py 中配置:
    - config.download.datasets:     待下载数据集列表
    - config.paths.datasets_root:   数据集保存根目录
    - config.download.hf_token:     HuggingFace token (私有数据集需要)

使用方法:
    # 下载 config.py 中配置的所有数据集
    python utils/download_dataset.py

    # 通过环境变量指定 HF_TOKEN
    HF_TOKEN=<token> python utils/download_dataset.py

    # 下载指定的单个数据集
    python utils/download_dataset.py --repo_id "org/dataset-name"

    # 自定义保存根目录
    python utils/download_dataset.py --datasets_root /your/custom/path

    # 下载指定 split
    python utils/download_dataset.py --repo_id "your-org/dataset" --split "train"
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import login

# 将项目根目录加入 path, 以便导入 config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def get_hf_token(cfg: Config) -> str:
    """
    获取 HuggingFace token.

    优先级:
    1. 环境变量 HF_TOKEN
    2. config.download.hf_token
    3. 无 token (下载公开数据集)
    """
    token = os.environ.get("HF_TOKEN") or cfg.download.hf_token
    if token:
        log.info("HuggingFace token detected.")
    else:
        log.info("No HuggingFace token set. Only public datasets can be downloaded.")
    return token


def is_dataset_downloaded(local_dir: str) -> bool:
    """
    检查数据集是否已经下载完成.

    判断标准: 目录存在, 且包含 dataset_info.json 和 hf_dataset/ 子目录.
    (dataset_info.json 是本脚本下载完成后写入的元信息文件)

    Args:
        local_dir: 数据集本地目录

    Returns:
        True 表示数据集已存在且看起来完整
    """
    p = Path(local_dir)

    if not p.is_dir():
        return False

    # 检查是否有 dataset_info.json (我们自己写入的标志文件)
    has_info = (p / "dataset_info.json").is_file()

    # 检查是否有 hf_dataset/ 子目录 (保存的 HuggingFace Dataset)
    has_hf = (p / "hf_dataset").is_dir()

    return has_info and has_hf


def download_single_dataset(
    repo_id: str,
    local_dir: str,
    subset: str = None,
    split: str = None,
    token: str = None,
    cache_dir: str = None,
    force: bool = False,
) -> str:
    """
    下载单个 HuggingFace 数据集并保存到本地磁盘.

    会同时保存:
    - HuggingFace Dataset 格式 (save_to_disk, 用于训练加载)
    - JSON 格式备份 (方便查看内容)
    - 数据集信息文件 (dataset_info.json)

    Args:
        repo_id:   HuggingFace 数据集 repo_id
        local_dir: 本地保存路径
        subset:    数据集 subset (如 "default"), None 表示不指定
        split:     数据集 split (如 "train"), None 表示下载全部
        token:     HuggingFace token
        cache_dir: HuggingFace 缓存目录
        force:     是否强制重新下载 (忽略已存在检查)

    Returns:
        "skipped" 表示已存在跳过, "ok" 表示下载成功, "failed" 表示失败
    """
    log.info(f"Downloading dataset: {repo_id}")
    log.info(f"  Subset: {subset or '(all)'}")
    log.info(f"  Split:  {split or '(all)'}")
    log.info(f"  Target: {local_dir}")

    # ---- 检查是否已经下载过 ----
    if not force and is_dataset_downloaded(local_dir):
        # 读取已有的 dataset_info.json 显示信息
        info_path = Path(local_dir) / "dataset_info.json"
        try:
            with open(info_path, "r") as f:
                existing_info = json.load(f)
            sample_count = existing_info.get("total_samples", "unknown")
            download_time = existing_info.get("download_time", "unknown")
        except Exception:
            sample_count = "unknown"
            download_time = "unknown"

        total_size = sum(
            f.stat().st_size for f in Path(local_dir).rglob("*") if f.is_file()
        )
        log.info(
            f"SKIP: Dataset already exists at {local_dir} "
            f"({total_size / (1024**3):.2f} GB, {sample_count} samples, "
            f"downloaded at {download_time}). "
            f"Use --force to re-download."
        )
        return "skipped"

    start_time = time.time()

    try:
        # 创建目标目录
        output_path = Path(local_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 加载数据集
        log.info("Loading dataset from HuggingFace Hub...")
        load_kwargs = {
            "path": repo_id,
            "token": token,
            "cache_dir": cache_dir,
        }
        if subset:
            load_kwargs["name"] = subset
        if split:
            load_kwargs["split"] = split

        ds = load_dataset(**load_kwargs)

        # ---- 打印数据集信息 ----
        if hasattr(ds, "keys"):
            # DatasetDict (多个 split)
            log.info(f"Dataset loaded. Splits: {list(ds.keys())}")
            for split_name, split_data in ds.items():
                log.info(f"  {split_name}: {len(split_data)} samples, columns: {split_data.column_names}")
            total_samples = sum(len(v) for v in ds.values())
        else:
            # 单个 Dataset
            log.info(f"Dataset loaded. Samples: {len(ds)}, columns: {ds.column_names}")
            total_samples = len(ds)

        # ---- 保存为 HuggingFace Dataset 格式 (用于训练时 load_from_disk) ----
        hf_dir = output_path / "hf_dataset"
        log.info(f"Saving HuggingFace Dataset to: {hf_dir}")
        ds.save_to_disk(str(hf_dir))

        # ---- 保存 JSON 备份 (方便人工查看) ----
        json_dir = output_path / "json_backup"
        json_dir.mkdir(exist_ok=True)

        if hasattr(ds, "keys"):
            for split_name, split_data in ds.items():
                json_path = json_dir / f"{split_name}.json"
                # 只保存前 100 条作为预览 (大数据集完整保存太慢)
                preview_count = min(100, len(split_data))
                preview = split_data.select(range(preview_count)).to_dict()
                # 将 numpy/list 类型统一转为 Python 原生类型
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(preview, f, ensure_ascii=False, indent=2, default=str)
                log.info(f"  JSON preview ({preview_count} samples): {json_path}")
        else:
            json_path = json_dir / "data.json"
            preview_count = min(100, len(ds))
            preview = ds.select(range(preview_count)).to_dict()
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(preview, f, ensure_ascii=False, indent=2, default=str)
            log.info(f"  JSON preview ({preview_count} samples): {json_path}")

        # ---- 保存数据集元信息 ----
        info = {
            "repo_id": repo_id,
            "subset": subset,
            "split": split,
            "total_samples": total_samples,
            "download_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if hasattr(ds, "keys"):
            info["splits"] = {k: len(v) for k, v in ds.items()}
            info["columns"] = {k: v.column_names for k, v in ds.items()}
        else:
            info["columns"] = ds.column_names

        info_path = output_path / "dataset_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start_time
        log.info(f"Download complete: {repo_id} ({elapsed / 60:.1f} min, {total_samples} samples)")

        # 查看目录总大小
        total_size = sum(
            f.stat().st_size for f in output_path.rglob("*") if f.is_file()
        )
        log.info(f"Total size on disk: {total_size / (1024**3):.2f} GB")

        return "ok"

    except Exception as e:
        elapsed = time.time() - start_time
        log.error(f"Failed to download {repo_id} after {elapsed / 60:.1f} min: {e}")
        import traceback
        traceback.print_exc()
        log.info("Troubleshooting:")
        log.info("  1. Check network connectivity")
        log.info("  2. Verify the repo_id is correct")
        log.info("  3. For private datasets, ensure HF_TOKEN is set")
        log.info(f"  4. Try manual download: https://huggingface.co/datasets/{repo_id}")
        return "failed"


def download_all_datasets(cfg: Config, datasets_root_override: str = None, force: bool = False):
    """
    下载 config.py 中配置的所有数据集.

    Args:
        cfg:                    全局配置
        datasets_root_override: 如有指定, 覆盖 config.paths.datasets_root
        force:                  是否强制重新下载 (忽略已存在检查)
    """
    datasets_root = datasets_root_override or cfg.paths.datasets_root
    token = get_hf_token(cfg)
    cache_dir = cfg.paths.hf_cache_dir

    # 登录 (如果有 token)
    if token:
        try:
            login(token=token)
            log.info("HuggingFace login successful.")
        except Exception as e:
            log.warning(f"HuggingFace login failed: {e}. Proceeding without login.")

    entries = cfg.download.datasets
    if not entries:
        log.warning("No datasets configured in config.download.datasets. Nothing to download.")
        return

    log.info("=" * 70)
    log.info(f"Datasets root:       {datasets_root}")
    log.info(f"Datasets to download: {len(entries)}")
    if force:
        log.info("Force mode: will re-download even if datasets already exist.")
    log.info("=" * 70)

    results = []  # (repo_id, status)

    for idx, entry in enumerate(entries, 1):
        repo_id = entry["repo_id"]
        local_name = entry.get("local_name") or repo_id.split("/")[-1]
        subset = entry.get("subset")
        split = entry.get("split")
        local_dir = str(Path(datasets_root) / local_name)

        log.info("")
        log.info(f"[{idx}/{len(entries)}] {repo_id}")
        log.info(f"  Local dir: {local_dir}")

        status = download_single_dataset(
            repo_id=repo_id,
            local_dir=local_dir,
            subset=subset,
            split=split,
            token=token,
            cache_dir=cache_dir,
            force=force,
        )
        results.append((repo_id, status))

    # ---- 汇总报告 ----
    log.info("")
    log.info("=" * 70)
    log.info("Download Summary")
    log.info("=" * 70)
    for repo_id, status in results:
        tag = {"ok": "OK", "skipped": "SKIPPED", "failed": "FAILED"}[status]
        log.info(f"  [{tag}] {repo_id}")

    skipped = [r for r in results if r[1] == "skipped"]
    failed = [r for r in results if r[1] == "failed"]
    downloaded = [r for r in results if r[1] == "ok"]

    if skipped:
        log.info(f"{len(skipped)} dataset(s) already existed and were skipped.")
    if downloaded:
        log.info(f"{len(downloaded)} dataset(s) downloaded successfully.")
    if failed:
        log.warning(f"{len(failed)} dataset(s) failed to download.")


def main():
    parser = argparse.ArgumentParser(
        description="GEO UQ Head - Download datasets from HuggingFace Hub"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Download a single dataset by repo_id (overrides config.py list)",
    )
    parser.add_argument(
        "--local_name",
        type=str,
        default=None,
        help="Local directory name under datasets_root (used with --repo_id)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset subset name (used with --repo_id)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to download, e.g. 'train' (used with --repo_id)",
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        default=None,
        help="Override datasets root directory (default from config.py)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-download even if dataset already exists locally",
    )
    args = parser.parse_args()

    cfg = Config()

    if args.repo_id:
        # ---------- 下载单个指定数据集 ----------
        datasets_root = args.datasets_root or cfg.paths.datasets_root
        token = get_hf_token(cfg)
        cache_dir = cfg.paths.hf_cache_dir

        if token:
            try:
                login(token=token)
            except Exception:
                pass

        local_name = args.local_name or args.repo_id.split("/")[-1]
        local_dir = str(Path(datasets_root) / local_name)

        log.info(f"Single dataset download: {args.repo_id} -> {local_dir}")
        status = download_single_dataset(
            repo_id=args.repo_id,
            local_dir=local_dir,
            subset=args.subset,
            split=args.split,
            token=token,
            cache_dir=cache_dir,
            force=args.force,
        )
        sys.exit(0 if status != "failed" else 1)

    else:
        # ---------- 下载 config.py 中配置的所有数据集 ----------
        download_all_datasets(cfg, datasets_root_override=args.datasets_root, force=args.force)


if __name__ == "__main__":
    main()
