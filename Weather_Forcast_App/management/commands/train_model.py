"""
Django management command: train_model

Cách dùng:
    python3 manage.py train_model
    python3 manage.py train_model --config path/to/custom_config.json
    python3 manage.py train_model --config path/to/custom_config.json --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

from django.core.management.base import BaseCommand


# Config mặc định (relative tới project root — manage.py cwd)
_DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "Machine_learning_model"
    / "config"
    / "train_config.json"
)


class Command(BaseCommand):
    help = "Train the weather forecast ML model using train_config.json"

    def add_arguments(self, parser):
        parser.add_argument(
            "--config",
            type=str,
            default=None,
            help=(
                f"Path to train config JSON/YAML. "
                f"Default: {_DEFAULT_CONFIG}"
            ),
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Load config và validate nhưng không chạy training thật.",
        )

    def handle(self, *args, **options):
        # ── Resolve config path ───────────────────────────────────── #
        cfg_arg = options.get("config")
        if cfg_arg:
            cfg_path = Path(cfg_arg).expanduser().resolve()
        else:
            cfg_path = _DEFAULT_CONFIG

        if not cfg_path.exists():
            self.stderr.write(
                self.style.ERROR(f"Config not found: {cfg_path}")
            )
            sys.exit(1)

        self.stdout.write(self.style.NOTICE(f"Using config: {cfg_path}"))

        # ── Import train module ───────────────────────────────────── #
        try:
            from Weather_Forcast_App.Machine_learning_model.trainning.train import (
                _load_config,
                run_training,
            )
        except ImportError as e:
            self.stderr.write(self.style.ERROR(f"Import error: {e}"))
            sys.exit(1)

        # ── Load & validate config ────────────────────────────────── #
        try:
            config = _load_config(cfg_path)
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Failed to load config: {e}"))
            sys.exit(1)

        self.stdout.write(
            f"  Model type : {config.get('model', {}).get('type', '?')}\n"
            f"  Target     : {config.get('target_column', '?')}\n"
            f"  Data file  : {config.get('data', {}).get('filename', '?')}"
        )

        if options["dry_run"]:
            self.stdout.write(self.style.SUCCESS("Dry-run: config OK, skipping training."))
            return

        # ── Run training ──────────────────────────────────────────── #
        self.stdout.write(self.style.NOTICE("\nStarting training..."))
        try:
            info = run_training(config)
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Training failed: {e}"))
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # ── Print results ─────────────────────────────────────────── #
        self.stdout.write(self.style.SUCCESS("\n" + "=" * 60))
        self.stdout.write(self.style.SUCCESS("  TRAINING COMPLETE"))
        self.stdout.write("=" * 60)
        self.stdout.write(f"  Model    : {info['model']['model_path']}")
        self.stdout.write(f"  Pipeline : {info['transform']['pipeline_path']}")
        self.stdout.write(f"  Metrics  : {info['artifacts']['metrics']}")

        # Print key metrics
        try:
            import json
            with open(info["artifacts"]["metrics"], encoding="utf-8") as f:
                metrics = json.load(f)

            self.stdout.write("\n-- Metrics --")
            for split in ("train", "valid", "test"):
                m = metrics.get(split, {})
                if m:
                    r2       = m.get("R2", m.get("r2", "N/A"))
                    mae      = m.get("MAE", "N/A")
                    rain_acc = m.get("Rain_Detection_Accuracy", "N/A")
                    f1       = m.get("Rain_F1", "N/A")
                    csi      = m.get("CSI", "N/A")
                    mbe      = m.get("MBE", "N/A")
                    pearson  = m.get("Pearson", "N/A")
                    r2_str      = f"{r2:.4f}"       if isinstance(r2,       float) else str(r2)
                    mae_str     = f"{mae:.4f}"      if isinstance(mae,      float) else str(mae)
                    acc_str     = f"{rain_acc:.4f}" if isinstance(rain_acc, float) else str(rain_acc)
                    f1_str      = f"{f1:.4f}"       if isinstance(f1,       float) else str(f1)
                    csi_str     = f"{csi:.4f}"      if isinstance(csi,      float) else str(csi)
                    mbe_str     = f"{mbe:+.4f}"     if isinstance(mbe,      float) else str(mbe)
                    pearson_str = f"{pearson:.4f}"  if isinstance(pearson,  float) else str(pearson)
                    self.stdout.write(
                        f"  {split:5s}  R2={r2_str}  MAE={mae_str}  RainAcc={acc_str}"
                        f"  F1={f1_str}  CSI={csi_str}  MBE={mbe_str}  Pearson={pearson_str}"
                    )

            train_time = metrics.get("training_time_seconds", "N/A")
            if isinstance(train_time, (int, float)):
                self.stdout.write(f"  Training time  : {train_time}s")

            diag = metrics.get("diagnostics", {})
            if diag:
                self.stdout.write(f"\n-- Diagnostics --")
                self.stdout.write(f"  Overfit status : {diag.get('overfit_status', 'N/A')}")
                self.stdout.write(f"  Details        : {diag.get('overfit_details', 'N/A')}")
        except Exception as e:
            self.stdout.write(f"  (Could not read metrics: {e})")

        self.stdout.write(self.style.SUCCESS("=" * 60))
