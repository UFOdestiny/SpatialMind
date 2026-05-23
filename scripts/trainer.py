#!/usr/bin/env python3
"""Trainer and TrainingArguments builders for SpatialMind scripts."""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import torch

from utils.numpy_compat import patch_numpy_core_multiarray

# Defensive compatibility patch for direct imports of this module.
patch_numpy_core_multiarray()

from transformers import Trainer, TrainingArguments

from config import Config
from data.cached_features import CachedFeatureDataset, ChunkAwareSampler

log = logging.getLogger(__name__)


def resolve_greater_is_better(
    metric_for_best_model: str,
    configured_greater_is_better: Optional[bool],
) -> bool:
    """Resolve whether larger metric values are better."""
    if configured_greater_is_better is not None:
        return configured_greater_is_better

    metric_name = (metric_for_best_model or "").lower()
    if metric_name.endswith("ece") or metric_name.endswith("loss"):
        return False
    return True


def build_train_training_arguments(
    cfg: Config,
    output_dir: str,
    warmup_steps: int,
    report_to_override=None,
) -> TrainingArguments:
    """Build TrainingArguments for training."""
    use_mixed_precision = torch.cuda.is_available()
    greater_is_better = resolve_greater_is_better(
        cfg.training.metric_for_best_model,
        cfg.training.greater_is_better,
    )
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        learning_rate=cfg.training.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        fp16=cfg.training.fp16 and use_mixed_precision,
        bf16=cfg.training.bf16 and use_mixed_precision,
        eval_strategy=cfg.training.eval_strategy,
        save_strategy=cfg.training.save_strategy,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=greater_is_better,
        seed=cfg.training.seed,
        report_to=(
            cfg.training.report_to
            if report_to_override is None
            else report_to_override
        ),
        # DataLoader优化
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        dataloader_pin_memory=cfg.training.dataloader_pin_memory,
        dataloader_prefetch_factor=cfg.training.dataloader_prefetch_factor if cfg.training.dataloader_num_workers > 0 else None,
        dataloader_persistent_workers=cfg.training.dataloader_persistent_workers if cfg.training.dataloader_num_workers > 0 else False,
        dataloader_drop_last=cfg.training.dataloader_drop_last,
        logging_steps=cfg.training.logging_steps,
        logging_first_step=True,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        remove_unused_columns=False,
        label_names=["claim_labels"],
        logging_strategy=cfg.training.logging_strategy,
        disable_tqdm=cfg.training.disable_tqdm,
    )


def build_eval_training_arguments(
    cfg: Config,
    output_dir: str,
    per_device_eval_batch_size: int,
) -> TrainingArguments:
    """Build TrainingArguments for evaluation-only runs."""
    use_mixed_precision = torch.cuda.is_available()
    return TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=per_device_eval_batch_size,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        remove_unused_columns=False,
        label_names=["claim_labels"],
        report_to="none",
        fp16=cfg.training.fp16 and use_mixed_precision,
        bf16=cfg.training.bf16 and use_mixed_precision,
        disable_tqdm=cfg.training.disable_tqdm,
    )


class HeadTrainer(Trainer):
    """Custom Trainer that only optimizes head + feature extractor parameters.
    
    Handles variable-length claim labels that HuggingFace Trainer doesn't support natively.
    """

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to handle variable-length claim labels.
        
        HuggingFace Trainer expects labels to be a fixed-size tensor per batch.
        Our claim_labels is a list of tensors (one per sample, variable claims).
        We flatten claim_labels to match the flattened logits from wrapper.py.
        """
        # Save claim_labels before parent class potentially removes them
        claim_labels = inputs.get("claim_labels")

        # Some Trainer paths may request loss-only eval even when compute_metrics exists.
        # Force full outputs when claim labels are available so eval metrics can be computed.
        effective_prediction_loss_only = prediction_loss_only
        if claim_labels is not None and self.compute_metrics is not None:
            effective_prediction_loss_only = False

        # Call parent's prediction_step
        loss, logits, _ = super().prediction_step(
            model, inputs, effective_prediction_loss_only, ignore_keys
        )

        if logits is None:
            return loss, logits, None

        # Flatten claim_labels to match flattened logits
        if claim_labels is not None and isinstance(claim_labels, list):
            labels = torch.cat([cl.to(logits.device) for cl in claim_labels], dim=0)
        elif isinstance(claim_labels, torch.Tensor):
            labels = claim_labels.to(logits.device)
        else:
            labels = None

        return loss, logits, labels

    def get_eval_dataloader(self, eval_dataset=None):
        """Never drop the last batch for eval; otherwise small eval sets can become empty."""
        old_drop_last = self.args.dataloader_drop_last
        try:
            self.args.dataloader_drop_last = False
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.args.dataloader_drop_last = old_drop_last

    def get_test_dataloader(self, test_dataset):
        """Never drop the last batch for test-time metrics."""
        old_drop_last = self.args.dataloader_drop_last
        try:
            self.args.dataloader_drop_last = False
            return super().get_test_dataloader(test_dataset)
        finally:
            self.args.dataloader_drop_last = old_drop_last

    def _determine_best_metric(self, metrics, trial):
        """Robust best-metric selection with fallback when configured metric is absent."""
        try:
            return super()._determine_best_metric(metrics=metrics, trial=trial)
        except KeyError:
            metric_name = self.args.metric_for_best_model or ""
            metric_key = metric_name if metric_name.startswith("eval_") else f"eval_{metric_name}"
            fallback_order = [
                "eval_pr_auc",
                "eval_roc_auc",
                "eval_f1",
                "eval_accuracy",
                "eval_loss",
                "eval_runtime",
            ]
            fallback_key = next((k for k in fallback_order if k in metrics), None)
            if fallback_key is None:
                log.warning(
                    "Best-metric key '%s' missing and no fallback metric found. "
                    "Skipping best-model update this evaluation step.",
                    metric_key,
                )
                return False

            old_metric = self.args.metric_for_best_model
            old_greater = self.args.greater_is_better
            self.args.metric_for_best_model = fallback_key
            if fallback_key.endswith("loss") or fallback_key.endswith("runtime"):
                self.args.greater_is_better = False
            elif old_greater is None:
                self.args.greater_is_better = True

            log.warning(
                "Best-metric key '%s' missing; temporarily using '%s'.",
                metric_key,
                fallback_key,
            )
            try:
                return super()._determine_best_metric(metrics=metrics, trial=trial)
            finally:
                self.args.metric_for_best_model = old_metric
                self.args.greater_is_better = old_greater

    def create_optimizer(self):
        """Only optimize trainable (head + feature extractor) parameters."""
        if self.optimizer is None:
            trainable = self.model.get_trainable_params()
            self.optimizer = torch.optim.AdamW(
                trainable,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer

    def _get_train_sampler(self, train_dataset=None):
        """Use chunk-aware sampling to minimize disk I/O for chunked datasets.

        Iterates all samples within one chunk before moving to the next,
        reducing chunk loads from O(steps) random loads to O(num_chunks)
        sequential loads per epoch.
        """
        ds = train_dataset if train_dataset is not None else self.train_dataset
        if isinstance(ds, CachedFeatureDataset) and not ds._preload_all:
            log.info(
                "Using ChunkAwareSampler (%d chunks) to reduce I/O",
                len(ds.chunk_files),
            )
            return ChunkAwareSampler(
                ds,
                seed=self.args.seed,
                drop_last=self.args.dataloader_drop_last,
                batch_size=self.args.per_device_train_batch_size,
            )
        return super()._get_train_sampler(train_dataset)

    def _save(self, output_dir=None, state_dict=None):
        """Save head (and feature extractor if present) artifacts."""
        _ = state_dict
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Keep HuggingFace checkpoint contract for resume_from_checkpoint.
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        torch.save(self.model.head.state_dict(), os.path.join(output_dir, "head_weights.pth"))

        if hasattr(self.model, "feature_extractor"):
            torch.save(
                self.model.feature_extractor.state_dict(),
                os.path.join(output_dir, "fe_weights.pth"),
            )

        if hasattr(self, "_head_config"):
            with open(os.path.join(output_dir, "head_config.json"), "w", encoding="utf-8") as f:
                json.dump(self._head_config, f, indent=2)

        with open(os.path.join(output_dir, "training_args.json"), "w", encoding="utf-8") as f:
            f.write(self.args.to_json_string())

    def _load_best_model(self):
        """Load best checkpoint weights for head (+ feature extractor if present)."""
        best_path = self.state.best_model_checkpoint
        if best_path is None:
            if self.args.load_best_model_at_end:
                raise RuntimeError(
                    "load_best_model_at_end=True, but no best checkpoint was recorded."
                )
            log.warning("No best model checkpoint found, skipping load_best_model_at_end.")
            return

        log.info("Loading best model from %s", best_path)

        head_path = os.path.join(best_path, "head_weights.pth")
        if not os.path.isfile(head_path):
            raise FileNotFoundError(f"Expected head checkpoint not found: {head_path}")
        self.model.head.load_state_dict(torch.load(head_path, map_location="cpu"))
        log.info("  Loaded head weights from %s", head_path)

        fe_path = os.path.join(best_path, "fe_weights.pth")
        if hasattr(self.model, "feature_extractor") and os.path.isfile(fe_path):
            self.model.feature_extractor.load_state_dict(torch.load(fe_path, map_location="cpu"))
            log.info("  Loaded feature extractor weights from %s", fe_path)
