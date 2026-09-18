"""Sanity checks for the pipeline: no look-ahead in features, weights rescale."""
import numpy as np
import pandas as pd

from msci_em.pipeline import (Config, build_features, build_target, derive_basics,
                              load_and_clean, predictions_to_weights)

PATH = "data/raw/MSCI_EM_Portfolio_since_Dec-21.xlsx"


def _features(long):
    basics = derive_basics(long, verbose=False)
    feats, cols = build_features(basics, Config(), verbose=False)
    return feats, cols


def test_features_use_past_only():
    long = load_and_clean(PATH, verbose=False)
    full, cols = _features(long)
    # drop the last two periods and recompute: features at earlier periods must
    # be identical, otherwise something is looking forward
    cut = long["quarter_idx"].max() - 2
    trunc, _ = _features(long[long["quarter_idx"] <= cut])
    a = full[full["quarter_idx"] <= cut].set_index(["stock", "quarter_idx"])[cols].sort_index()
    b = trunc.set_index(["stock", "quarter_idx"])[cols].sort_index().reindex(a.index)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_target_is_next_period_weight_change():
    long = load_and_clean(PATH, verbose=False)
    feats, cols = _features(long)
    train, latest = build_target(feats, cols, verbose=False)
    w = long.set_index(["stock", "quarter_idx"])["weight"]
    row = train.iloc[0]
    nxt = w[(row["stock"], row["quarter_idx"] + 1)]
    assert np.isclose(row["target"], nxt / row["weight"] - 1)
    assert train["quarter_idx"].max() < latest["quarter_idx"].max()


def test_weights_rescale_to_100():
    pw = predictions_to_weights(np.array([50.0, 30.0, 20.0]), np.array([0.5, -2.0, 0.0]))
    assert np.isclose(pw.sum(), 100) and pw[1] == 0
    long = load_and_clean(PATH, verbose=False)
    sums = long.groupby("quarter")["weight"].sum()
    assert np.allclose(sums, 100)
