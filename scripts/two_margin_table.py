#!/usr/bin/env python3
"""
two_margin_table.py -- the per-block two-margin table behind the rebuilt
Figure 4 and the Section 4.2 numbers.

    python3 two_margin_table.py --data handoff_upload --out results/analysis

For every block of the 20-variable profile partition, two marginal
contributions to aggregate recovery, both as per-item Fisher-z differences
in the between-country correlation:

  add margin     z(minimal + block) - z(minimal)      sparse context
  remove margin  z(full) - z(full - block)            saturated context

The two margins are the extreme coalition positions, so together they
bracket every ordering of additions (the answer to the cumulative-ordering
objection). The country row's remove margin is the region-free pair
(full_noregion vs full_nocountry), the paper's established identification;
its add margin is minimal vs demo_only. The region row's add margin uses
minimal_region (NUTS code alone on the minimal base).

Output: two_margin_by_block.csv (block, margin, fwd/rev medians, counts,
n items beyond the 0.060 band) and two_margin_by_item.csv (per item).
Assertions: 42 items per arm, the country remove margin must reproduce
clean_ablations.csv to 1e-9.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

BLOCKS = [
    # block label, add contrast (with, base), remove contrast (with, without)
    ("country label",     ("qwen_1p_minimal", "qwen_1p_demo_only"),
                          ("qwen_1p_full_noregion", "qwen_1p_full_nocountry")),
    ("NUTS region code",  ("qwen_1p_minimal_region", "qwen_1p_minimal"),
                          ("qwen_1p", "qwen_1p_full_noregion")),
    ("political identity", ("qwen_1p_minimal_politics", "qwen_1p_minimal"),
                          ("qwen_1p", "qwen_1p_full_nopolitical")),
    ("ascriptive",        (None, None),   # gender/age/birth year ARE the minimal base
                          ("qwen_1p", "qwen_1p_full_noascriptive")),
    ("education, activity, income", ("qwen_1p_minimal_ses", "qwen_1p_minimal"),
                          ("qwen_1p", "qwen_1p_full_nosocioecon")),
    ("union, internet use", ("qwen_1p_minimal_membership", "qwen_1p_minimal"),
                          ("qwen_1p", "qwen_1p_full_nomembership")),
    ("household",         ("qwen_1p_minimal_household", "qwen_1p_minimal"),
                          ("qwen_1p", "qwen_1p_full_nohousehold")),
    ("migration, citizenship", ("qwen_1p_minimal_civic", "qwen_1p_minimal"),
                          ("qwen_1p", "qwen_1p_full_nocivic")),
    ("minority status",   ("qwen_1p_minimal_minority", "qwen_1p_minimal"),
                          ("qwen_1p", "qwen_1p_full_nominority")),
    ("domicile",          ("qwen_1p_minimal_domicil", "qwen_1p_minimal"),
                          ("qwen_1p", "qwen_1p_full_nodomicil")),
]

NOISE = 0.060


def r_bc(tag: str, d: str) -> pd.Series:
    s = pd.read_csv(os.path.join(d, f"silicon_full_country_scatter_{tag}.csv"))
    r = pd.Series({v: stats.pearsonr(g.survey_mean, g.silicon_mean)[0]
                   for v, g in s.groupby("variable")}).sort_index()
    assert len(r) == 42, f"{tag}: expected 42 items, found {len(r)}"
    return r


def z(x):
    return np.arctanh(np.clip(np.asarray(x, dtype=float), -0.999999, 0.999999))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="handoff_upload")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--allow_missing", action="store_true",
                    help="skip contrasts whose arms have not landed yet")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    dirt = pd.read_csv(os.path.join(a.data, "item_direction_table.csv"))
    rev = dirt.set_index("variable")["direction"].eq("reverse")

    cache: dict[str, pd.Series] = {}

    def get(tag):
        if tag not in cache:
            cache[tag] = r_bc(tag, a.data)
        return cache[tag]

    rows, per_item = [], []
    for block, (aw, ab), (rw, rwo) in BLOCKS:
        for margin, w, b in (("add", aw, ab), ("remove", rw, rwo)):
            if w is None:
                continue
            try:
                dz = pd.Series(z(get(w)) - z(get(b)), index=get(w).index)
            except FileNotFoundError:
                if a.allow_missing:
                    print(f"  SKIP {block} / {margin}: arm not landed")
                    continue
                raise
            rv = rev.reindex(dz.index).fillna(False)
            rows.append(dict(
                block=block, margin=margin, with_arm=w, base_arm=b,
                fwd_median_dz=dz[~rv].median(), rev_median_dz=dz[rv].median(),
                fwd_improved=int((dz[~rv] > 0).sum()), n_fwd=int((~rv).sum()),
                rev_improved=int((dz[rv] > 0).sum()), n_rev=int(rv.sum()),
                n_beyond_noise=int((dz.abs() > NOISE).sum()),
                with_median_rbc=get(w).median(), base_median_rbc=get(b).median()))
            for v, val in dz.items():
                per_item.append(dict(block=block, margin=margin, variable=v,
                                     reverse=bool(rv[v]), dz=val))

    tab = pd.DataFrame(rows)
    # the country remove margin must reproduce the shipped ablation numbers
    abl = pd.read_csv(os.path.join(a.data, "clean_ablations.csv"))
    ref = abl[abl.block == "country label"].iloc[0]
    got = tab[(tab.block == "country label") & (tab.margin == "remove")]
    if len(got):
        assert abs(got.iloc[0].fwd_median_dz - ref.fwd_median_dz) < 1e-9
        assert abs(got.iloc[0].rev_median_dz - ref.rev_median_dz) < 1e-9

    tab.round(6).to_csv(os.path.join(a.out, "two_margin_by_block.csv"), index=False)
    pd.DataFrame(per_item).round(6).to_csv(
        os.path.join(a.out, "two_margin_by_item.csv"), index=False)
    with pd.option_context("display.width", 140):
        print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
