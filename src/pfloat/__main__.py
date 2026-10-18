"""Command line: python -m pfloat <stage> [--config config/default.yaml]"""

from __future__ import annotations

import argparse
import logging
import time

from .config import Config
from .monitoring import stage_monitor
from .pipeline import run_all, stage_features, stage_production, stage_select_threshold, stage_simulate, stage_train
from .report import stage_report

STAGES = {
    "simulate": stage_simulate,
    "features": stage_features,
    "train": stage_train,
    "select-threshold": stage_select_threshold,
    "production": stage_production,
    "monitor": stage_monitor,
    "report": stage_report,
    "all": run_all,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="pfloat", description="Payment float risk pipeline")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    t0 = time.time()
    STAGES[args.stage](Config.load(args.config))
    logging.getLogger("pfloat").info("done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
