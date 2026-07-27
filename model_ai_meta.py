"""
MODEL_AI Layer 5 - Meta-labeling (optional)
===========================================
López de Prado's meta-labeling pattern: instead of predicting DIRECTION (which
Model A already attempts, and which is the low-signal task), train a secondary
classifier to predict P(this trade is profitable) and use it to filter and size.

Labels come from a triple-barrier: for each historical signal, label 1 if price
touches +k*ATR before -k*ATR within H bars, else 0.

Off by default (``use_meta_label: False``) — it adds ~1-2 min per run because
the training set is rebuilt from scratch each time. Flip it on in CONFIG once
you want the extra filter.

LOOK-AHEAD: the panel handed in here is already truncated to as_of by
``model_ai._slice``. Training samples are additionally restricted to dates at
least H bars before the end, so a label can never peek past the decision date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import model_ai

# Triple-barrier geometry.
BARRIER_ATR_MULT = 2.0   # k
BARRIER_HORIZON = 10     # H (trading days)
SAMPLE_EVERY = 5         # sample history weekly to bound training cost
MAX_SAMPLE_DATES = 120   # hard cap on training dates
MIN_TRAIN_ROWS = 100     # below this, the model isn't worth trusting

FEATURES = ["score", "regime_on", "vol_20", "mom_12_1", "mom_3", "rsi_14"]


def _triple_barrier_label(px: pd.Series, start_idx: int, atr_val: float) -> int | None:
    """
    1 if +k*ATR is touched before -k*ATR within H bars, else 0.
    None if there isn't a full horizon of forward data (sample is dropped).
    """
    if not np.isfinite(atr_val) or atr_val <= 0:
        return None
    end_idx = start_idx + BARRIER_HORIZON
    if end_idx >= len(px):
        return None

    entry = px.iloc[start_idx]
    upper = entry + BARRIER_ATR_MULT * atr_val
    lower = entry - BARRIER_ATR_MULT * atr_val

    for i in range(start_idx + 1, end_idx + 1):
        p = px.iloc[i]
        if p >= upper:
            return 1
        if p <= lower:
            return 0
    # Neither barrier touched: call it on the sign of the horizon return.
    return int(px.iloc[end_idx] > entry)


def build_training_set(data: model_ai.PriceData, cfg: dict) -> pd.DataFrame:
    """Walk history, recompute features at each sample date, attach labels."""
    dates = data.close.index
    # Leave room for the forward horizon; require enough warm-up for 12m momentum.
    warmup = max(cfg["mom_long_lookback"] + cfg["mom_long_skip"], cfg["trend_sma"]) + 5
    usable = dates[warmup:len(dates) - BARRIER_HORIZON - 1]
    if len(usable) == 0:
        return pd.DataFrame()

    sample_dates = usable[::SAMPLE_EVERY][-MAX_SAMPLE_DATES:]
    rows = []

    for dt in sample_dates:
        sliced = model_ai._slice(data, dt)
        feat = model_ai.compute_features(sliced, cfg)
        if feat.empty:
            continue
        selected = model_ai.select_names(feat, cfg)
        if selected.empty:
            continue

        regime = model_ai.detect_regime(sliced, cfg)
        pos = dates.get_loc(dt)

        for ticker, row in selected.iterrows():
            px = data.close[ticker]
            a = model_ai.atr(data.high[ticker], data.low[ticker], px).iloc[pos]
            label = _triple_barrier_label(px, pos, a)
            if label is None:
                continue
            rows.append({
                "score": row["score"],
                "regime_on": float(regime["factor"]),
                "vol_20": row["vol_20"],
                "mom_12_1": row["mom_12_1"],
                "mom_3": row["mom_3"],
                "rsi_14": row["rsi_14"],
                "label": label,
            })

    return pd.DataFrame(rows)


def train(data: model_ai.PriceData, cfg: dict) -> RandomForestClassifier | None:
    """Fit the meta-model. Returns None when there isn't enough labeled data."""
    train_df = build_training_set(data, cfg)
    if len(train_df) < MIN_TRAIN_ROWS:
        print(f"[meta] only {len(train_df)} labeled samples "
              f"(< {MIN_TRAIN_ROWS}) — skipping meta-label filter")
        return None
    if train_df["label"].nunique() < 2:
        print("[meta] labels are single-class — skipping meta-label filter")
        return None

    model = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model.fit(train_df[FEATURES], train_df["label"])
    base_rate = train_df["label"].mean()
    print(f"[meta] trained on {len(train_df)} samples "
          f"(base win rate {base_rate:.1%})")
    return model


def conviction_for(weights: pd.Series, selected: pd.DataFrame,
                   data: model_ai.PriceData, cfg: dict) -> tuple[pd.Series, float]:
    """
    Filter names below the P(win) threshold and derive a conviction multiplier.

    Returns (surviving weights renormalized, conviction in [0, 1]). Falls back
    to (weights, 1.0) — i.e. behaves as if Layer 5 were off — whenever the
    meta-model can't be trained, so a cold start degrades to plain MODEL_AI
    rather than to no trading at all.
    """
    model = train(data, cfg)
    if model is None:
        return weights, 1.0

    regime = model_ai.detect_regime(data, cfg)
    X = pd.DataFrame({
        "score": selected["score"],
        "regime_on": float(regime["factor"]),
        "vol_20": selected["vol_20"],
        "mom_12_1": selected["mom_12_1"],
        "mom_3": selected["mom_3"],
        "rsi_14": selected["rsi_14"],
    })[FEATURES]

    p_win = pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    threshold = cfg["meta_prob_threshold"]
    keep = p_win[p_win > threshold].index

    for t in p_win.sort_values(ascending=False).index:
        mark = "KEEP" if t in keep else "drop"
        print(f"[meta] {t:9} P(win)={p_win[t]:.2f} {mark}")

    if len(keep) == 0:
        return pd.Series(dtype=float), 0.0

    surviving = weights.loc[keep]
    surviving = surviving / surviving.sum()
    conviction = float(np.clip((p_win.loc[keep].mean() - 0.5) / 0.5, 0.0, 1.0))
    return surviving, conviction
