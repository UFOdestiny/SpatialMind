#!/usr/bin/env python3
"""
download_models.py - 模型下载脚本
===============================

从 HuggingFace Hub 下载 LLM 模型到本地存储.

下载列表和目标路径在 config.py 中配置:
    - config.download.models:     待下载模型列表
    - config.paths.models_root:   模型保存根目录
    - config.download.hf_token:   HuggingFace token (私有模型需要)

使用方法:
    # 下载 config.py 中配置的所有模型
    python utils/download_models.py

    # 通过环境变量指定 HF_TOKEN (不用写在代码里)
    HF_TOKEN=<token> python utils/download_models.py

    # 下载指定的单个模型 (忽略 config.py 列表, 直接指定)
    python utils/download_models.py --repo_id "org/model-name" --local_name "model-name"

    # 自定义保存根目录
    python utils/download_models.py --models_root /your/custom/path
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

from huggingface_hub import snapshot_download, login

# 将项目根目录加入 path, 以便导入 config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config, DownloadModelEntry

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
    3. 无 token (下载公开模型)
    """
    token = os.environ.get("HF_TOKEN") or cfg.download.hf_token
    if token:
        log.info("HuggingFace token detected.")
    else:
        log.info("No HuggingFace token set. Only public models can be downloaded.")
    return token


def resolve_local_dir(models_root: str, entry: DownloadModelEntry) -> str:
    """
    根据模型配置解析本地保存目录.

    如果 entry.local_name 为空, 则自动取 repo_id 的最后一段.
    例: repo_id="org/model-name" -> local_name="model-name"
    """
    if entry.local_name:
        name = entry.local_name
    else:
        # 取 repo_id 的最后一段作为目录名
        name = entry.repo_id.split("/")[-1]
    return str(Path(models_root) / name)


def is_model_downloaded(local_dir: str) -> bool:
    """
    检查模型是否已经下载完成.

    判断标准: 目录存在, 且包含 config.json 和至少一个权重文件
    (.safetensors 或 .bin).

    Args:
        local_dir: 模型本地目录

    Returns:
        True 表示模型已存在且看起来完整
    """
    p = Path(local_dir)

    if not p.is_dir():
        return False

    # 检查是否有 config.json (HuggingFace 模型的标志文件)
    has_config = (p / "config.json").is_file()

    # 检查是否有权重文件 (.safetensors 或 .bin)
    has_weights = (
        any(p.glob("*.safetensors"))
        or any(p.glob("*.bin"))
        or any(p.glob("model-*.safetensors"))  # 分片权重
        or any(p.glob("pytorch_model-*.bin"))   # 分片权重
    )

    return has_config and has_weights


def download_single_model(
    repo_id: str,
    local_dir: str,
    token: str = None,
    use_symlinks: bool = False,
    resume: bool = True,
    force: bool = False,
) -> str:
    """
    下载单个模型.

    Args:
        repo_id:      HuggingFace repo_id
        local_dir:    本地保存路径
        token:        HuggingFace token (可为 None)
        use_symlinks: 是否使用符号链接
        resume:       是否允许断点续传
        force:        是否强制重新下载 (忽略已存在检查)

    Returns:
        "skipped" 表示已存在跳过, "ok" 表示下载成功, "failed" 表示失败
    """
    log.info(f"Downloading: {repo_id}")
    log.info(f"Target dir:  {local_dir}")

    # ---- 检查是否已经下载过 ----
    if not force and is_model_downloaded(local_dir):
        total_size = sum(
            f.stat().st_size for f in Path(local_dir).rglob("*") if f.is_file()
        )
        log.info(
            f"SKIP: Model already exists at {local_dir} "
            f"({total_size / (1024**3):.2f} GB). "
            f"Use --force to re-download."
        )
        return "skipped"

    start_time = time.time()

    try:
        # 创建目标目录
        os.makedirs(local_dir, exist_ok=True)

        # 使用 huggingface_hub 的 snapshot_download
        result = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=use_symlinks,
            token=token,
            resume_download=resume,
        )

        elapsed = time.time() - start_time
        log.info(f"Download complete: {repo_id} ({elapsed / 60:.1f} min)")

        # 查看目录大小
        total_size = sum(
            f.stat().st_size for f in Path(local_dir).rglob("*") if f.is_file()
        )
        log.info(f"Total size: {total_size / (1024**3):.2f} GB")

        return "ok"

    except Exception as e:
        elapsed = time.time() - start_time
        log.error(f"Failed to download {repo_id} after {elapsed / 60:.1f} min: {e}")
        import traceback
        traceback.print_exc()
        return "failed"


def download_all_models(cfg: Config, models_root_override: str = None, force: bool = False):
    """
    下载 config.py 中配置的所有模型.

    Args:
        cfg:                 全局配置
        models_root_override: 如有指定, 覆盖 config.paths.models_root
        force:               是否强制重新下载 (忽略已存在检查)
    """
    models_root = models_root_override or cfg.paths.models_root
    token = get_hf_token(cfg)

    # 登录 (如果有 token)
    if token:
        try:
            login(token=token)
            log.info("HuggingFace login successful.")
        except Exception as e:
            log.warning(f"HuggingFace login failed: {e}. Proceeding without login.")

    entries = cfg.download.models
    if not entries:
        log.warning("No models configured in config.download.models. Nothing to download.")
        return

    log.info("=" * 70)
    log.info(f"Models root:    {models_root}")
    log.info(f"Models to download: {len(entries)}")
    if force:
        log.info("Force mode: will re-download even if models already exist.")
    log.info("=" * 70)

    results = []  # (repo_id, status)

    for idx, entry in enumerate(entries, 1):
        local_dir = resolve_local_dir(models_root, entry)

        log.info("")
        log.info(f"[{idx}/{len(entries)}] {entry.repo_id}")
        log.info(f"  Size (est.): {entry.size or 'unknown'}")
        log.info(f"  Local dir:   {local_dir}")

        status = download_single_model(
            repo_id=entry.repo_id,
            local_dir=local_dir,
            token=token,
            use_symlinks=cfg.download.use_symlinks,
            resume=cfg.download.resume_download,
            force=force,
        )
        results.append((entry.repo_id, status))

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
        log.info(f"{len(skipped)} model(s) already existed and were skipped.")
    if downloaded:
        log.info(f"{len(downloaded)} model(s) downloaded successfully.")
    if failed:
        log.warning(f"{len(failed)} model(s) failed to download.")


def main():
    parser = argparse.ArgumentParser(
        description="GEO UQ Head - Download models from HuggingFace Hub"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Download a single model by repo_id (overrides config.py list)",
    )
    parser.add_argument(
        "--local_name",
        type=str,
        default=None,
        help="Local directory name under models_root (used with --repo_id)",
    )
    parser.add_argument(
        "--models_root",
        type=str,
        default=None,
        help="Override models root directory (default from config.py)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-download even if model already exists locally",
    )
    args = parser.parse_args()

    cfg = Config()

    if args.repo_id:
        # ---------- 下载单个指定模型 ----------
        models_root = args.models_root or cfg.paths.models_root
        token = get_hf_token(cfg)
        if token:
            try:
                login(token=token)
            except Exception:
                pass

        entry = DownloadModelEntry(
            repo_id=args.repo_id,
            local_name=args.local_name or "",
        )
        local_dir = resolve_local_dir(models_root, entry)

        log.info(f"Single model download: {args.repo_id} -> {local_dir}")
        status = download_single_model(
            repo_id=args.repo_id,
            local_dir=local_dir,
            token=token,
            use_symlinks=cfg.download.use_symlinks,
            resume=cfg.download.resume_download,
            force=args.force,
        )
        sys.exit(0 if status != "failed" else 1)

    else:
        # ---------- 下载 config.py 中配置的所有模型 ----------
        download_all_models(cfg, models_root_override=args.models_root, force=args.force)


if __name__ == "__main__":
    main()
