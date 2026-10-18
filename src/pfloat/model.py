"""Train, calibrate, save and score the payment return model.

* Out-of-time splits: train on Jan-Jun, calibrate on Jul, select the threshold
  on Aug, evaluate on production months once they mature.
* `scale_pos_weight` handles the class imbalance but inflates probabilities,
  so a Platt calibrator (logistic regression on the raw log-odds), fit on a
  later month, maps raw output back to real return rates. Platt keeps the
  score continuous; isotonic is the alternative when the shape is not sigmoid.
* The production score is round(P_calibrated x 1000), clipped to 1..999, so a
  score of 20 reads as a 2.0% expected return rate.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from . import metrics
from .config import Config
from .features import MODEL_FEATURES

MODEL_VERSION = "pfloat_xgb_v1"


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


@dataclass
class ScoringModel:
    booster: xgb.XGBClassifier
    calibrator: LogisticRegression
    features: list[str]
    version: str = MODEL_VERSION

    def raw_proba(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            # Fail loudly: silently filling NaN would degrade the model unnoticed.
            raise ValueError(f"missing model features: {missing}")
        return self.booster.predict_proba(df[self.features].astype(float))[:, 1]

    def proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict_proba(_logit(self.raw_proba(df)))[:, 1]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return to_score(self.proba(df))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(directory / "model.json")
        with open(directory / "calibrator.pkl", "wb") as f:
            pickle.dump(self.calibrator, f)
        (directory / "model_meta.json").write_text(json.dumps({"version": self.version, "features": self.features}, indent=2))

    @classmethod
    def load(cls, directory: Path) -> ScoringModel:
        meta = json.loads((directory / "model_meta.json").read_text())
        booster = xgb.XGBClassifier()
        booster.load_model(directory / "model.json")
        with open(directory / "calibrator.pkl", "rb") as f:
            calibrator = pickle.load(f)
        return cls(booster=booster, calibrator=calibrator, features=meta["features"], version=meta["version"])


def to_score(p: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(p, dtype=float) * 1000.0), 1, 999).astype(int)


def split_masks(df: pd.DataFrame, cfg: Config) -> dict[str, pd.Series]:
    t = df["initiated_at"]
    start = cfg.date("simulation", "start_date")
    train_end = cfg.date("splits", "train_end") + pd.Timedelta(days=1)
    cal_end = cfg.date("splits", "calibration_end") + pd.Timedelta(days=1)
    sel_end = cfg.date("splits", "selection_end") + pd.Timedelta(days=1)
    return {
        "train": (t >= start) & (t < train_end),
        "calibration": (t >= train_end) & (t < cal_end),
        "selection": (t >= cal_end) & (t < sel_end),
        "production": t >= sel_end,
    }


def train(features: pd.DataFrame, cfg: Config) -> tuple[ScoringModel, dict]:
    masks = split_masks(features, cfg)
    labelled = features["label"].notna()
    tr = features[masks["train"] & labelled]
    ca = features[masks["calibration"] & labelled]
    se = features[masks["selection"] & labelled]

    pos = tr["label"].sum()
    params = dict(cfg["model"]["params"])
    params["scale_pos_weight"] = float((len(tr) - pos) / pos)
    booster = xgb.XGBClassifier(random_state=cfg.seed, n_jobs=4, **params)
    booster.fit(tr[MODEL_FEATURES].astype(float), tr["label"].astype(int), verbose=False)

    raw_ca = booster.predict_proba(ca[MODEL_FEATURES].astype(float))[:, 1]
    calibrator = LogisticRegression(C=1e6)
    calibrator.fit(_logit(raw_ca), ca["label"].astype(int).to_numpy())
    model = ScoringModel(booster=booster, calibrator=calibrator, features=list(MODEL_FEATURES))

    report = {"version": model.version, "scale_pos_weight": params["scale_pos_weight"], "splits": {}}
    for name, part in (("train", tr), ("calibration", ca), ("selection", se)):
        raw = model.raw_proba(part)
        cal = model.proba(part)
        report["splits"][name] = {
            "raw": metrics.summary(part["label"], raw),
            "calibrated": metrics.summary(part["label"], cal),
        }
    report["feature_importance"] = shap_importance(model, se)
    return model, report


def shap_importance(model: ScoringModel, df: pd.DataFrame, n: int = 20000) -> dict[str, float]:
    """Mean |SHAP| per feature using XGBoost's built-in TreeSHAP."""
    sample = df.sample(min(n, len(df)), random_state=0)
    dm = xgb.DMatrix(sample[model.features].astype(float), feature_names=model.features)
    contrib = model.booster.get_booster().predict(dm, pred_contribs=True)[:, :-1]
    imp = np.abs(contrib).mean(axis=0)
    order = np.argsort(-imp)
    return {model.features[k]: float(imp[k]) for k in order}
