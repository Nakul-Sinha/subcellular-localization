"""Strict validator for submission.csv against the challenge schema."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

CLASSES = ["nucleoplasm","nuclear_membrane","nucleoli","nucleoli_fibrillar_center",
    "nuclear_speckles","nuclear_bodies","endoplasmic_reticulum","golgi_apparatus",
    "vesicles","plasma_membrane","cytosol","mitochondria","microtubules","centrosome",
    "actin_filaments","intermediate_filaments"]
LOC_COLS = [f"loc_{c}" for c in CLASSES]

def main(sub_path="./working/submission.csv", test_path="./dataset/public/test.csv"):
    sub = pd.read_csv(sub_path)
    test = pd.read_csv(test_path)

    assert "id" in sub.columns, "missing id column"
    for c in LOC_COLS:
        assert c in sub.columns, f"missing column {c}"
    assert sub["id"].is_unique, "duplicate ids"
    assert set(sub["id"]) >= set(test["id"]), "missing test ids (would be scored all-absent)"
    extra = set(sub["id"]) - set(test["id"])
    if extra:
        print(f"note: {len(extra)} extra ids will be ignored by the grader")

    vals = sub[LOC_COLS].to_numpy()
    assert np.isfinite(vals).all(), "NaN/Inf in loc_* values"
    assert (vals >= 0).all() and (vals <= 1).all(), "values outside [0,1]"
    assert not sub[["id"] + LOC_COLS].isna().any().any(), "NaNs present"

    print(f"OK: {sub.shape[0]} rows, {len(LOC_COLS)} loc columns, "
          f"range [{vals.min():.3f}, {vals.max():.3f}]")
    print("Submission validation passed.")

if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args)
