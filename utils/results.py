#!/usr/bin/env python3
"""
Summarize SpatialMind evaluation outputs with pandas.

Features:
1) Compare metrics across heads — separate tables per dataset (ID / OOD).
2) Report efficiency metrics (Params/FLOPs/Train/Inference/Memory).
3) Visualize training curves and evaluation bar charts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.number_utils import first_number, to_float

# ── Model short-name registry ──────────────────────────────────────────────
MODEL_SHORT_NAMES: Dict[str, str] = {}


def _short_model_name(name: str) -> str:
    return MODEL_SHORT_NAMES.get(name, name)


METRIC_ALIASES = {
    "accuracy": "Acc",
    "precision": "Prec",
    "recall": "Recall",
    "ece": "ECE",
    "nll": "NLL",
    "brier": "Brier",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "f1": "F1",
    "aurc": "AURC",
    "risk_at_80_cov": "Risk@80",
    "risk_at_90_cov": "Risk@90",
}

# Reverse map: display name → internal key
_DISPLAY_TO_KEY = {v: k for k, v in METRIC_ALIASES.items()}

LOWER_IS_BETTER = {"ece", "nll", "brier", "aurc", "risk_at_80_cov", "risk_at_90_cov"}

# ── Plot styling ────────────────────────────────────────────────────────────
FONT_FAMILY = "Times New Roman"
TITLE_SIZE = 14
LABEL_SIZE = 10
TICK_SIZE = 9
SPINE_WIDTH = 2


def _to_float(value: Any) -> float:
    parsed = to_float(value)
    if parsed is None:
        return float("nan")
    return parsed


def _extract_metric(
    overall: Dict[str, Any],
    key: str,
    trainer_metrics: Optional[Dict[str, Any]] = None,
) -> float:
    trainer_metrics = trainer_metrics or {}

    if key == "accuracy":
        candidates = [
            overall.get("accuracy", overall.get("acc")),
            trainer_metrics.get("test_accuracy"),
            trainer_metrics.get("eval_accuracy"),
        ]
    elif key == "precision":
        candidates = [
            overall.get("precision"),
            trainer_metrics.get("test_precision"),
            trainer_metrics.get("eval_precision"),
        ]
    elif key == "recall":
        candidates = [
            overall.get("recall"),
            trainer_metrics.get("test_recall"),
            trainer_metrics.get("eval_recall"),
        ]
    elif key == "pr_auc":
        candidates = [
            overall.get("pr_auc", overall.get("prauc")),
            trainer_metrics.get("test_pr_auc"),
            trainer_metrics.get("eval_pr_auc"),
        ]
    elif key == "roc_auc":
        candidates = [
            overall.get("roc_auc", overall.get("auroc")),
            trainer_metrics.get("test_roc_auc"),
            trainer_metrics.get("eval_roc_auc"),
        ]
    elif key == "f1":
        candidates = [
            overall.get("f1", overall.get("f1_score")),
            trainer_metrics.get("test_f1"),
            trainer_metrics.get("eval_f1"),
        ]
    else:
        candidates = [
            overall.get(key),
            trainer_metrics.get(f"test_{key}"),
            trainer_metrics.get(f"eval_{key}"),
        ]

    for value in candidates:
        parsed = _to_float(value)
        if pd.notna(parsed):
            return parsed
    return float("nan")


def _first_number(*values: Any) -> float:
    return first_number(*values, default=float("nan"))


def _discover_reports(results_root: Path) -> List[Dict[str, Any]]:
    report_rows: List[Dict[str, Any]] = []
    for report_path in sorted(results_root.rglob("evaluation_report.json")):
        try:
            with report_path.open("r", encoding="utf-8") as f:
                report = json.load(f)
        except json.JSONDecodeError:
            continue

        relative = report_path.relative_to(results_root)
        parts = relative.parts

        # Determine dataset scope: "eval_ood/<dataset>/*" → OOD, else ID
        dataset_scope = "ID"
        ood_dataset = ""
        if len(parts) >= 2 and parts[0] == "eval_ood":
            dataset_scope = "OOD"
            ood_dataset = parts[1]

        run_name = "."
        inferred_head = None
        if "eval" in parts:
            eval_idx = parts.index("eval")
            run_name = str(Path(*parts[:eval_idx])) if eval_idx > 0 else "."
            if eval_idx + 1 < len(parts):
                inferred_head = parts[eval_idx + 1]
        elif "eval_ood" in parts:
            eval_idx = parts.index("eval_ood")
            # head is after dataset name: eval_ood/<dataset>/<head>/...
            if eval_idx + 2 < len(parts):
                inferred_head = parts[eval_idx + 2]
        elif "evaluation" in parts:
            eval_idx = parts.index("evaluation")
            run_name = str(Path(*parts[:eval_idx])) if eval_idx > 0 else "."
            if eval_idx > 0:
                inferred_head = parts[eval_idx - 1]
        elif report_path.parent != results_root:
            run_name = str(report_path.parent.relative_to(results_root))

        head = report.get("head_type") or inferred_head or report_path.parent.name
        report_rows.append(
            {
                "run_name": run_name,
                "head": head,
                "report_path": str(report_path),
                "report": report,
                "dataset_scope": dataset_scope,
                "ood_dataset": ood_dataset,
            }
        )
    return report_rows


def _build_tables(
    report_rows: Iterable[Dict[str, Any]],
    heads: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, Any]] = []
    efficiency_rows: List[Dict[str, Any]] = []

    for row in report_rows:
        head = row["head"]
        if heads and head not in heads:
            continue

        report = row["report"]
        overall = report.get("overall_metrics", {})
        trainer_metrics = report.get("trainer_metrics", {})
        efficiency = report.get("efficiency", {})

        base_model_name = report.get("base_model_name") or Path(
            report.get("base_model_path", "")
        ).name
        if not base_model_name:
            base_model_name = "unknown"
        base_model_short = _short_model_name(base_model_name)

        method_type = report.get("method_type", "supervised")

        metric_row = {
            "Model": base_model_short,
            "Head": head,
            "Type": method_type,
            "N": report.get("total_samples"),
            "dataset_scope": row["dataset_scope"],
            "ood_dataset": row["ood_dataset"],
        }
        for key, col in METRIC_ALIASES.items():
            metric_row[col] = _extract_metric(overall, key, trainer_metrics)
        metric_rows.append(metric_row)

        eff_row = {
            "Model": base_model_short,
            "Head": head,
            "dataset_scope": row["dataset_scope"],
            "ood_dataset": row["ood_dataset"],
            "Params(M)": _first_number(
                efficiency.get("params_m"),
                _to_float(report.get("trainable_params")) / 1e6,
            ),
            "FLOPs(G)": _first_number(efficiency.get("flops_g")),
            "Train(h)": _first_number(efficiency.get("train_time_h")),
            "Epoch(s)": _first_number(efficiency.get("epoch_s")),
            "Infer(s)": _first_number(
                efficiency.get("inference_s"),
                report.get("inference_time_s"),
                trainer_metrics.get("test_runtime"),
                trainer_metrics.get("eval_runtime"),
            ),
            "CPU(GB)": _first_number(
                efficiency.get("cpu_memory_gb"),
                efficiency.get("inference_cpu_memory_gb"),
                efficiency.get("train_cpu_memory_gb"),
            ),
            "GPU(GB)": _first_number(
                efficiency.get("gpu_memory_gb"),
                efficiency.get("inference_gpu_memory_gb"),
                efficiency.get("train_gpu_memory_gb"),
            ),
        }
        efficiency_rows.append(eff_row)

    return pd.DataFrame(metric_rows), pd.DataFrame(efficiency_rows)


def _print_df(df: pd.DataFrame, title: str, digits: int = 4) -> None:
    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)
    if df.empty:
        print("  (no data)")
        return

    df = df.copy()
    # Drop internal columns before printing
    for col in ("dataset_scope", "ood_dataset"):
        if col in df.columns:
            df = df.drop(columns=[col])

    # Set Head as index (Model only shown once per group if multiple models)
    index_cols = [c for c in ["Model", "Head"] if c in df.columns]
    if index_cols:
        df = df.sort_values(index_cols).set_index(index_cols)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(float_format=lambda x: f"{x:.{digits}f}"))


def _print_grouped_tables(
    metric_df: pd.DataFrame,
    efficiency_df: pd.DataFrame,
    title: str,
) -> None:
    """Print metric and efficiency tables, split by dataset scope and model."""
    if metric_df.empty:
        print(f"\n  No evaluation data found for: {title}")
        return

    # Group by (dataset_scope, ood_dataset, Model) for separate tables
    groups = []
    if "dataset_scope" in metric_df.columns:
        for (scope, ood_ds), sub_m in metric_df.groupby(
            ["dataset_scope", "ood_dataset"], dropna=False, sort=True
        ):
            sub_e = efficiency_df[
                (efficiency_df["dataset_scope"] == scope)
                & (efficiency_df["ood_dataset"] == ood_ds)
            ]
            if scope == "ID":
                label = f"{title} | In-Distribution"
            else:
                label = f"{title} | OOD ({ood_ds})"
            groups.append((label, sub_m, sub_e))
    else:
        groups.append((title, metric_df, efficiency_df))

    for label, sub_m, sub_e in groups:
        # Further split by Model if multiple base models
        models = sub_m["Model"].unique()
        for model in sorted(models):
            m = sub_m[sub_m["Model"] == model]
            e = sub_e[sub_e["Model"] == model]
            tag = f"{label} | {model}" if len(models) > 1 else label

            _print_df(m, f"{tag} | Metrics", digits=4)
            _print_df(e, f"{tag} | Efficiency", digits=4)


# ── Visualizer ──────────────────────────────────────────────────────────────

def _setup_plot_style(plt):
    """Configure matplotlib with Times New Roman and consistent styling."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [FONT_FAMILY, "DejaVu Serif"],
        "font.size": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": TICK_SIZE,
        "figure.titlesize": TITLE_SIZE,
    })


def _style_axes(ax, plt=None):
    """Apply consistent spine width and tick styling."""
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, direction="in")


class TrainingVisualizer:
    """Visualize training curves from train_results.json files."""

    # Metrics to plot (exclude time-related)
    PLOT_METRICS = [
        ("eval_accuracy", "Accuracy"),
        ("eval_precision", "Precision"),
        ("eval_recall", "Recall"),
        ("eval_f1", "F1"),
        ("eval_pr_auc", "PR-AUC"),
        ("eval_roc_auc", "ROC-AUC"),
        ("eval_ece", "ECE"),
        ("eval_loss", "Loss"),
    ]

    def __init__(self, results_root: Path, output_dir: Path):
        self.results_root = results_root
        self.output_dir = output_dir

    def _discover_train_results(self) -> List[Dict[str, Any]]:
        """Find all train_results.json files."""
        rows = []
        for path in sorted(self.results_root.rglob("train_results.json")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                continue

            head = data.get("head_type", path.parent.name)
            base_model = _short_model_name(
                data.get("head_config", {}).get("cache_dir", "").split("/")[-1]
                if "cache_dir" in data.get("head_config", {})
                else "unknown"
            )
            # Infer dataset from cache_dir path
            cache_dir = data.get("cache_dir", "")
            dataset = "unknown"
            parts = Path(cache_dir).parts
            for i, p in enumerate(parts):
                if p == "cached_features" and i + 1 < len(parts):
                    # cached_features/<dataset>/<model> or cached_features/example/<dataset>/<model>
                    candidate = parts[i + 1]
                    if candidate == "example" and i + 2 < len(parts):
                        dataset = parts[i + 2]
                    else:
                        dataset = candidate
                    break

            rows.append({
                "head": head,
                "model": base_model,
                "dataset": dataset,
                "data": data,
                "path": str(path),
            })
        return rows

    def _extract_curves(self, log_history: List[Dict]) -> Dict[str, List]:
        """Extract per-epoch metric curves from log_history (eval entries only)."""
        curves: Dict[str, List] = {}
        for entry in log_history:
            if "epoch" not in entry:
                continue
            if not any(k.startswith("eval_") for k in entry):
                continue
            epoch = entry["epoch"]
            curves.setdefault("epoch_eval", []).append(epoch)
            for metric_key, _ in self.PLOT_METRICS:
                if metric_key in entry:
                    curves.setdefault(metric_key, []).append(entry[metric_key])
        return curves

    def generate_figures(self) -> List[str]:
        """Generate training curve figures. Returns list of saved file paths."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN] matplotlib not installed, skipping training curve plots.")
            return []

        _setup_plot_style(plt)

        rows = self._discover_train_results()
        if not rows:
            print("[WARN] No train_results.json files found.")
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Group by (dataset, model)
        groups: Dict[tuple, List] = {}
        for row in rows:
            key = (row["dataset"], row["model"])
            groups.setdefault(key, []).append(row)

        saved_files = []
        for (dataset, model), group_rows in sorted(groups.items()):
            fig_path = self.output_dir / f"training_curves_{dataset}_{model}.png"
            self._plot_group(group_rows, dataset, model, fig_path, plt)
            saved_files.append(str(fig_path))
            print(f"  Saved: {fig_path}")

        return saved_files

    def _plot_group(self, rows, dataset, model, fig_path, plt):
        """Plot training curves for one (dataset, model) group."""
        n_plots = len(self.PLOT_METRICS)
        n_cols = 4
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows),
            squeeze=False,
        )
        fig.suptitle(
            f"Training Curves — {dataset} / {model}",
            fontsize=TITLE_SIZE, fontweight="bold", y=1.02,
        )

        # Collect all heads for consistent colors
        heads = sorted(set(r["head"] for r in rows))
        cmap = plt.get_cmap("tab20" if len(heads) > 10 else "tab10")
        colors = {h: cmap(i % cmap.N) for i, h in enumerate(heads)}

        for idx, (metric_key, metric_label) in enumerate(self.PLOT_METRICS):
            r, c = divmod(idx, n_cols)
            ax = axes[r][c]
            has_data = False

            for row in sorted(rows, key=lambda r: r["head"]):
                log_history = row["data"].get("log_history", [])
                curves = self._extract_curves(log_history)

                epochs = curves.get("epoch_eval", [])
                values = curves.get(metric_key, [])

                if epochs and values and len(epochs) == len(values):
                    ax.plot(
                        epochs, values,
                        label=row["head"],
                        color=colors[row["head"]],
                        linewidth=1.2,
                        alpha=0.85,
                    )
                    has_data = True

            ax.set_title(metric_label, fontsize=LABEL_SIZE)
            ax.set_xlabel("Epoch", fontsize=TICK_SIZE)
            ax.grid(True, alpha=0.3)
            _style_axes(ax, plt)

            if not has_data:
                ax.text(
                    0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=TICK_SIZE, color="gray",
                )

        # Hide unused axes
        for idx in range(n_plots, n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            axes[r][c].set_visible(False)

        # Single legend at bottom
        handles, labels = [], []
        for ax_row in axes:
            for ax in ax_row:
                h, l = ax.get_legend_handles_labels()
                if h:
                    handles, labels = h, l
                    break
            if handles:
                break
        if handles:
            fig.legend(
                handles, labels,
                loc="lower center",
                ncol=min(len(heads), 7),
                fontsize=TICK_SIZE,
                bbox_to_anchor=(0.5, -0.02),
            )

        fig.tight_layout()
        fig.savefig(fig_path, dpi=400, bbox_inches="tight")
        plt.close(fig)


class EvaluationVisualizer:
    """Generate bar chart visualizations for evaluation metrics and efficiency."""

    METRIC_COLS = ["Acc", "Prec", "Recall", "F1", "PR-AUC", "ROC-AUC", "ECE", "AURC", "Risk@80"]
    EFFICIENCY_COLS = ["Params(M)", "Infer(s)", "GPU(GB)"]

    def __init__(
        self,
        metric_df: pd.DataFrame,
        efficiency_df: pd.DataFrame,
        output_dir: Path,
    ):
        self.metric_df = metric_df
        self.efficiency_df = efficiency_df
        self.output_dir = output_dir

    def generate_figures(self) -> List[str]:
        """Generate evaluation bar charts. Returns list of saved file paths."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("[WARN] matplotlib not installed, skipping evaluation bar charts.")
            return []

        _setup_plot_style(plt)

        if self.metric_df.empty:
            print("[WARN] No evaluation data for bar charts.")
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        # Group by (dataset_scope, ood_dataset, Model)
        groups = []
        if "dataset_scope" in self.metric_df.columns:
            for (scope, ood_ds, model), sub_m in self.metric_df.groupby(
                ["dataset_scope", "ood_dataset", "Model"], dropna=False, sort=True
            ):
                sub_e = self.efficiency_df[
                    (self.efficiency_df["dataset_scope"] == scope)
                    & (self.efficiency_df["ood_dataset"] == ood_ds)
                    & (self.efficiency_df["Model"] == model)
                ]
                if scope == "ID":
                    dataset_label = "ID"
                else:
                    dataset_label = f"OOD_{ood_ds}"
                groups.append((dataset_label, model, sub_m, sub_e))
        else:
            for model in self.metric_df["Model"].unique():
                sub_m = self.metric_df[self.metric_df["Model"] == model]
                sub_e = self.efficiency_df[self.efficiency_df["Model"] == model]
                groups.append(("eval", model, sub_m, sub_e))

        for dataset_label, model, m_df, e_df in groups:
            fig_path = self.output_dir / f"eval_bars_{dataset_label}_{model}.png"
            self._plot_eval_bars(m_df, e_df, dataset_label, model, fig_path, plt, np)
            saved_files.append(str(fig_path))
            print(f"  Saved: {fig_path}")

        return saved_files

    def _plot_eval_bars(self, m_df, e_df, dataset_label, model, fig_path, plt, np):
        """Create bar charts for metrics and efficiency."""
        # Merge metric and efficiency data
        heads = sorted(m_df["Head"].unique())
        n_heads = len(heads)

        if n_heads == 0:
            return

        # Prepare data
        metric_data = {col: [] for col in self.METRIC_COLS}
        efficiency_data = {col: [] for col in self.EFFICIENCY_COLS}

        for head in heads:
            m_row = m_df[m_df["Head"] == head]
            e_row = e_df[e_df["Head"] == head]

            for col in self.METRIC_COLS:
                val = m_row[col].values[0] if not m_row.empty and col in m_row.columns else float("nan")
                metric_data[col].append(val)

            for col in self.EFFICIENCY_COLS:
                val = e_row[col].values[0] if not e_row.empty and col in e_row.columns else float("nan")
                efficiency_data[col].append(val)

        # Create figure with subplots
        n_metric_plots = len(self.METRIC_COLS)
        n_eff_plots = len(self.EFFICIENCY_COLS)
        n_cols = 4
        n_rows = ((n_metric_plots + n_eff_plots) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows),
            squeeze=False,
        )
        fig.suptitle(
            f"Evaluation Results — {dataset_label} / {model}",
            fontsize=TITLE_SIZE, fontweight="bold", y=1.02,
        )

        x = np.arange(n_heads)
        bar_width = 0.7
        cmap = plt.get_cmap("tab20" if n_heads > 10 else "tab10")
        colors = [cmap(i % cmap.N) for i in range(n_heads)]

        all_plots = [(col, metric_data[col], "metric") for col in self.METRIC_COLS]
        all_plots += [(col, efficiency_data[col], "efficiency") for col in self.EFFICIENCY_COLS]

        for idx, (col, values, plot_type) in enumerate(all_plots):
            r, c = divmod(idx, n_cols)
            ax = axes[r][c]

            values_arr = np.array(values)
            valid_mask = ~np.isnan(values_arr)

            if not np.any(valid_mask):
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=TICK_SIZE, color="gray")
                ax.set_title(col, fontsize=LABEL_SIZE)
                _style_axes(ax, plt)
                continue

            ax.bar(x, np.nan_to_num(values_arr), bar_width, color=colors, edgecolor="black", linewidth=0.5)

            ax.set_title(col, fontsize=LABEL_SIZE)
            ax.set_xticks(x)
            ax.set_xticklabels(heads, rotation=45, ha="right", fontsize=TICK_SIZE - 1)
            ax.grid(axis="y", alpha=0.3)
            _style_axes(ax, plt)

        # Hide unused axes
        total_plots = len(all_plots)
        for idx in range(total_plots, n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            axes[r][c].set_visible(False)

        fig.tight_layout()
        fig.savefig(fig_path, dpi=400, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize SpatialMind evaluation results.")
    parser.add_argument("--results-root", required=True, help="Root directory of results.")
    parser.add_argument(
        "--heads",
        nargs="*",
        default=None,
        help="Optional head filter, e.g. --heads uq",
    )
    parser.add_argument(
        "--title",
        default="SpatialMind Summary",
        help="Title prefix for printed tables.",
    )
    parser.add_argument(
        "--figure-dir",
        default=None,
        help="Directory to save training curve and evaluation figures.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.is_dir():
        print(f"Results root not found: {results_root}")
        return 1

    report_rows = _discover_reports(results_root)
    if not report_rows:
        print("No evaluation_report.json files found.")
        return 0

    metric_df, efficiency_df = _build_tables(report_rows, heads=args.heads)

    _print_grouped_tables(
        metric_df, efficiency_df,
        title=args.title,
    )

    # Generate figures if requested
    if args.figure_dir:
        figure_dir = Path(args.figure_dir)

        print("--- Training curve plots ---")
        train_viz = TrainingVisualizer(results_root, figure_dir)
        train_viz.generate_figures()

        print("--- Evaluation bar charts ---")
        eval_viz = EvaluationVisualizer(
            metric_df, efficiency_df, figure_dir,
        )
        eval_viz.generate_figures()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
