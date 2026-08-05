#!/usr/bin/env python3
"""
swap_analysis.py -- scores the swapped-label falsification arm.

    python3 swap_analysis.py --data handoff_upload --out results/analysis

The full_swapcountry arm renders every backstory with a WRONG country under
the fixed seed-888 derangement (SWAP_MAP in the pipeline); the results file
keeps the TRUE country codes. Each item therefore yields two between-country
correlations over the same 30 cells:

  r_label  silicon means matched to the human means of the LABELED country
  r_true   silicon means matched to the human means of the TRUE country

The label-prior account predicts r_label near the true-label pair's level
(the full_noregion arm) and r_true near zero: cell means inherit the
labeled country's prior, so scoring them against the true country amounts
to correlating one permutation of country means against another.

Output: swap_scores.csv (per item, r_label, r_true, direction) and a
printed summary. Assertions: 42 items, 30 cells per item, SWAP_MAP is a
derangement, and the human means used for the label side equal the human
means of the labeled countries in the benchmark file.
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import pandas as pd
from scipy import stats


def load_swap_map(pipeline_path: str) -> dict:
    spec = importlib.util.spec_from_file_location("pipe", pipeline_path)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception:
        pass  # module-level side effects past SWAP_MAP are irrelevant here
    sm = getattr(m, "SWAP_MAP")
    assert sorted(sm) == sorted(set(sm.values())) and all(k != v for k, v in sm.items()), \
        "SWAP_MAP is not a derangement"
    return sm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="handoff_upload")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--pipeline", default="silicon_sampling_extended_v12.py")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sm = load_swap_map(a.pipeline)
    s = pd.read_csv(os.path.join(
        a.data, "silicon_full_country_scatter_qwen_1p_full_swapcountry.csv"))
    # human benchmark means per (true) country x item, from any full arm
    ref = pd.read_csv(os.path.join(
        a.data, "silicon_full_country_scatter_qwen_1p.csv"))
    human = ref.set_index(["variable", "cntry"]).survey_mean

    dirt = pd.read_csv(os.path.join(a.data, "item_direction_table.csv"))
    rev = dirt.set_index("variable")["direction"].eq("reverse")

    rows = []
    for v, g in s.groupby("variable"):
        assert len(g) == 30, f"{v}: {len(g)} cells"
        lab = g.cntry.map(sm)                       # the country each cell CLAIMED
        h_true = g.set_index("cntry").survey_mean   # human means, true country
        h_lab = pd.Series([human.get((v, c)) for c in lab], index=g.cntry)
        sil = g.set_index("cntry").silicon_mean
        # the arm's own survey_mean column must equal the true-country benchmark
        assert (h_true - g.set_index("cntry").survey_mean).abs().max() < 1e-12
        rows.append(dict(
            variable=v, reverse=bool(rev.get(v, False)),
            r_label=stats.pearsonr(h_lab, sil)[0],
            r_true=stats.pearsonr(h_true, sil)[0]))
    out = pd.DataFrame(rows).sort_values("r_label", ascending=False)
    assert len(out) == 42
    out.round(6).to_csv(os.path.join(a.out, "swap_scores.csv"), index=False)

    fwd = out[~out.reverse]
    print(f"vs LABELED country: median r {out.r_label.median():+.3f} "
          f"(forward items {fwd.r_label.median():+.3f})")
    print(f"vs TRUE country:    median r {out.r_true.median():+.3f} "
          f"(forward items {fwd.r_true.median():+.3f})")
    print("reference points: true-label pair (full_noregion) 0.416, "
          "no-label arm 0.344, noise floor 0.060")


if __name__ == "__main__":
    main()
