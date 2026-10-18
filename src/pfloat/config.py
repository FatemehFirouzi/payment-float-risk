"""Configuration loading. One YAML file drives every stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    root: Path

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path).resolve()
        with open(path) as f:
            raw = yaml.safe_load(f)
        # Paths in the config are relative to the repo root (parent of config/).
        return cls(raw=raw, root=path.parent.parent)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def path(self, key: str) -> Path:
        p = self.root / self.raw[key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_dir(self) -> Path:
        return self.path("data_dir")

    @property
    def artifacts_dir(self) -> Path:
        return self.path("artifacts_dir")

    @property
    def reports_dir(self) -> Path:
        return self.path("reports_dir")

    def date(self, section: str, key: str) -> pd.Timestamp:
        return pd.Timestamp(self.raw[section][key])

    @property
    def as_of(self) -> pd.Timestamp:
        return self.date("simulation", "as_of_date")

    @property
    def production_start(self) -> pd.Timestamp:
        return self.date("splits", "selection_end") + pd.Timedelta(days=1)
