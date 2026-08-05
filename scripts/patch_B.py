#!/usr/bin/env python3
"""Patch B -- benchmark_v13.py.  P3 + P4 + P5' + P6.

Six anchors, each asserted to match exactly once. All-or-nothing.

No existing column is renamed or removed, so collect_numbers.py and every
released table keep working. The patch only adds columns and rows.
"""
p = "benchmark_v13.py"
s = open(p).read()

reps = []

# ---- B1 (P3): leave-one-out grand mean, matched to the region comparator ----
reps.append((
"""        # comparator A: grand mean
        g["pred_grand"] = g.survey_mean.mean()
""",
"""        # comparator A: grand mean.
        # AUDIT FIX (P3): the in-sample grand mean is given an advantage that the
        # region comparator is denied, since pred_region below is leave-one-out.
        # The asymmetry favours the comparator over the model, so the released
        # claim was conservative rather than inflated, but the two are not
        # like-for-like. Both are now computed. pred_grand_loo is the matched
        # comparator and is the one to report; pred_grand is retained because
        # every released figure was computed from it.
        g["pred_grand"] = g.survey_mean.mean()
        _gs, _gk = g.survey_mean.sum(), int(g.survey_mean.notna().sum())
        g["pred_grand_loo"] = ((_gs - g.survey_mean) / (_gk - 1)) if _gk > 1 else np.nan
"""))

# ---- B2: include the new predictor in the metric loop ----------------------
reps.append((
'        for name in ["model", "debias", "region", "grand"] + (["gdp"] if gdp is not None else []):\n',
'        for name in ["model", "debias", "region", "grand", "grand_loo"] + \\\n'
'                (["gdp"] if gdp is not None else []):\n'))

# ---- B3 (P4): emit the honest column name alongside the legacy one ---------
reps.append((
"""            rec[f"mae_{name}"] = float(((g.survey_mean - p).abs() / sp)[ok].mean())
""",
"""            # AUDIT FIX (P4): this quantity is already divided by the item range
            # sp, so it is a range-normalised MAE, and the print header below
            # already calls it nMAE. The column name mae_* invites a reader or a
            # downstream script to normalise a second time; dividing by sp again
            # turns the median 0.180 into 0.019. country_profile_and_mae.csv names
            # the same quantity norm_mae. The honest name is emitted alongside.
            # mae_* is retained for compatibility with collect_numbers.py and the
            # released tables, and is DEPRECATED: prefer nmae_*.
            _nmae = float(((g.survey_mean - p).abs() / sp)[ok].mean())
            rec[f"mae_{name}"] = _nmae
            rec[f"nmae_{name}"] = _nmae
"""))

# ---- B4 (P5'): separate exactly-constant from near-constant ---------------
reps.append((
"""            elif p[ok].nunique() <= 1 or p[ok].std() < 1e-10:
                # BUG FIX: testing std() == 0 exactly does not work. A column
                # holding one repeated value returns a variance of order 1e-16
                # rather than exactly zero, so the equality test failed, the
                # constant array reached pearsonr, and the result was NaN. NaN
                # then loses every ">" comparison silently, which understated
                # the head-to-head count against the grand-mean predictor.
                # nunique() is exact for this case; the tolerance covers
                # near-constant predictors as well.
                rec[f"r_{name}"] = 0.0
""",
"""            elif p[ok].nunique() <= 1:
                # BUG FIX (retained): testing std() == 0 exactly does not work. A
                # column holding one repeated value returns a variance of order
                # 1e-16 rather than exactly zero, so the equality test failed, the
                # constant array reached pearsonr, and the result was NaN. NaN then
                # loses every ">" comparison silently, which understated the
                # head-to-head count against the grand-mean predictor. nunique()
                # is exact for this case.
                #
                # An EXACTLY constant predictor has an undefined correlation, and
                # zero is the correct substantive reading: no cross-national
                # discrimination whatsoever. It is kept at 0.0 so the head-to-head
                # counts stay meaningful. The manuscript table must nevertheless
                # print this cell as "undefined", not as 0.00, because a reader
                # scanning the r column will otherwise read 0.00 as an estimate.
                rec[f"r_{name}"] = 0.0
            elif p[ok].std() < 1e-10:
                # AUDIT FIX (P5'): a NEAR-constant predictor is a different case.
                # Its correlation is defined but numerically unstable, so
                # recording 0.0 would invent a number rather than report an
                # undefined one. This branch is dormant for the grand mean, which
                # is exactly constant, but becomes live as soon as --gdp is
                # supplied and a leave-one-out fit degenerates. The n_nan warning
                # below then reports it instead of silently absorbing it.
                rec[f"r_{name}"] = np.nan
"""))

# ---- B5: print the new comparator ----------------------------------------
reps.append((
"""    labels = [("model", "the model, as produced"),
              ("debias", "the model, per-item bias removed (UPPER BOUND, needs truth)"),
              ("region", "leave-one-out region mean (no model)"),
              ("grand", "grand mean (no model, cannot rank at all)")]
""",
"""    labels = [("model", "the model, as produced"),
              ("debias", "the model, per-item bias removed (UPPER BOUND, needs truth)"),
              ("region", "leave-one-out region mean (no model)"),
              ("grand", "grand mean, IN-SAMPLE (no model, cannot rank at all)"),
              ("grand_loo", "grand mean, leave-one-out (matched to the region arm)")]
"""))

# ---- B6 (P6): all four offset-reduction estimators, named ------------------
reps.append((
"""    red = 1 - d.mae_debias.median() / d.mae_model.median()
    print(f"\\n  Removing a single per-item offset cuts the model's error by "
          f"{100*red:.0f}%.")
""",
"""    # AUDIT FIX (P6): the released 64.2% is a RATIO OF MEDIANS, the largest of
    # four defensible estimators. For a claim phrased as "almost all of the level
    # error is a single per-item offset", the median of the per-item reductions is
    # the natural quantity, and it is 59.9%. All four are emitted and named so a
    # quoted figure can be identified.
    _item_red = 1 - d.nmae_debias / d.nmae_model
    _est = [
        ("ratio of medians (as released)",
         1 - d.nmae_debias.median() / d.nmae_model.median()),
        ("ratio of means", 1 - d.nmae_debias.mean() / d.nmae_model.mean()),
        ("MEDIAN of the per-item reductions (recommended)", _item_red.median()),
        ("mean of the per-item reductions", _item_red.mean()),
    ]
    red = _est[2][1]
    print("\\n  Removing a single per-item offset cuts the model's error by:")
    for _lab, _val in _est:
        print(f"    {100 * _val:5.1f}%   {_lab}")
    print(f"  Report {100 * red:.1f}% and name the estimator. Quoting the ratio of")
    print("  medians without naming it invites a reader who recomputes per item to")
    print("  get a figure five points lower.")
    pd.DataFrame([dict(estimator=_l, value=_v) for _l, _v in _est] + [
        dict(estimator="items where removing the offset helps",
             value=int((_item_red > 0).sum())),
        dict(estimator="items where removing the offset hurts",
             value=int((_item_red < 0).sum())),
    ]).to_csv(os.path.join(OUT, "offset_reduction_estimators.csv"), index=False)
"""))

# ---- B7 (P3 corollary): the LOO grand mean's r is -1 by construction -------
reps.append((
"""            if ok.sum() < 10:
                rec[f"r_{name}"] = np.nan
""",
"""            if name == "grand_loo":
                # AUDIT NOTE (P3 corollary). The leave-one-out grand mean is
                # (S - x_c)/(k-1), a strictly DECREASING affine function of the
                # very value it predicts, so its correlation with the truth is
                # exactly -1 for every item, by construction. That is an artefact
                # of leaving one out of a constant predictor, not a finding. It
                # must never be reported as the grand mean inverting country
                # rankings, and "the model beats it on r on 42 of 42 items" is
                # vacuous. This arm is a LEVEL comparator only: quote
                # nmae_grand_loo, and take the ranking comparison from the
                # in-sample arm, whose r of 0 encodes the substantive point that
                # a constant predictor cannot rank countries at all.
                rec[f"r_{name}"] = np.nan
            elif ok.sum() < 10:
                rec[f"r_{name}"] = np.nan
"""))

# ---- B8: do not print a head-to-head r count for an undefined comparator ---
reps.append((
"""    for k, lab in labels[2:]:
        print(f"    model beats {lab.split(' (')[0]:44s} "
              f"on r: {int((d.r_model > d[f'r_{k}']).sum()):2d}   "
              f"on nMAE: {int((d.mae_model < d[f'mae_{k}']).sum()):2d}")
""",
"""    for k, lab in labels[2:]:
        # AUDIT FIX (B8): a comparator whose r is undefined for every item must
        # not be given a win count. NaN loses every ">" comparison silently, so
        # the count would print as 0 and read as "the model never wins".
        _rcol = d[f"r_{k}"]
        _rtxt = "  n/a" if _rcol.isna().all() else f"{int((d.r_model > _rcol).sum()):3d}"
        print(f"    model beats {lab.split(' (')[0]:44s} "
              f"on r: {_rtxt}   "
              f"on nMAE: {int((d.nmae_model < d[f'nmae_{k}']).sum()):2d}")
"""))

for i, (old, new) in enumerate(reps, 1):
    n = s.count(old)
    assert n == 1, f"anchor B{i} matched {n} times, expected 1 -- ABORTED, nothing written"
    s = s.replace(old, new, 1)

open(p, "w").write(s)
print(f"Patch B applied to {p}: {len(reps)} anchors")
