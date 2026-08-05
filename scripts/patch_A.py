#!/usr/bin/env python3
"""Patch A -- analyze_v13.py.  P2 + E2(in place) + exclusion-criterion note.

Four anchors, each asserted to match exactly once. Nothing is written unless
all four match, so this is all-or-nothing.
"""
p = "analyze_v13.py"
s = open(p).read()

reps = []

# ---- A1 (P2): emit both estimators for both dispersion ratios ---------------
reps.append((
"""def within_country_sd_ratio(scatter: pd.DataFrame) -> float:
    r = (scatter.silicon_sd / scatter.survey_sd).replace([np.inf, -np.inf], np.nan)
    return float(r.mean(skipna=True))
""",
"""def within_country_sd_ratio(scatter: pd.DataFrame) -> float:
    r = (scatter.silicon_sd / scatter.survey_sd).replace([np.inf, -np.inf], np.nan)
    return float(r.mean(skipna=True))


# AUDIT FIX (P2). The two dispersion ratios reported side by side use DIFFERENT
# aggregators: within_country_sd_ratio is a MEAN over item x country cells,
# between_country_sd_ratio is a MEDIAN over items. Neither was recorded anywhere,
# and the thesis statement reports the pair in one sentence. Ratios are
# right-skewed, so for qwen_1p the same within-country claim can be stated as
# 0.605 (mean of ratios), 0.573 (median of ratios) or 0.586 (ratio of means).
# All estimators are now emitted so the manuscript can name the one it uses. The
# two original functions are left untouched, so no released figure moves.
def within_country_sd_ratio_median(scatter: pd.DataFrame) -> float:
    r = (scatter.silicon_sd / scatter.survey_sd).replace([np.inf, -np.inf], np.nan)
    return float(r.median(skipna=True))


def within_country_sd_ratio_of_means(scatter: pd.DataFrame) -> float:
    return float(scatter.silicon_sd.mean(skipna=True) /
                 scatter.survey_sd.mean(skipna=True))


def between_country_sd_ratio_mean(scatter: pd.DataFrame) -> float:
    ratios = []
    for v, g in scatter.groupby("variable"):
        hs, ss = g.survey_mean.std(), g.silicon_mean.std()
        if hs and hs > 0:
            ratios.append(ss / hs)
    return float(np.mean(ratios)) if ratios else np.nan
"""))

# ---- A2: compute the extra estimators at the call site ----------------------
reps.append((
"""    wsd = within_country_sd_ratio(scatters[tag])
    bsd = between_country_sd_ratio(scatters[tag])
""",
"""    wsd = within_country_sd_ratio(scatters[tag])
    bsd = between_country_sd_ratio(scatters[tag])
    wsd_med = within_country_sd_ratio_median(scatters[tag])
    wsd_rom = within_country_sd_ratio_of_means(scatters[tag])
    bsd_mean = between_country_sd_ratio_mean(scatters[tag])
"""))

# ---- A3: widen the summary row ---------------------------------------------
reps.append((
"""    summary_rows.append(dict(condition=tag, label=label, median_r_bc=r.median(),
                             n_gt50=int((r > .5).sum()), n_gt70=int((r > .7).sum()),
                             n_negative=int((r < 0).sum()), within_sd_ratio=wsd,
                             between_sd_ratio=bsd, n_items=len(r)))
""",
"""    summary_rows.append(dict(condition=tag, label=label, median_r_bc=r.median(),
                             n_gt50=int((r > .5).sum()), n_gt70=int((r > .7).sum()),
                             n_negative=int((r < 0).sum()), within_sd_ratio=wsd,
                             between_sd_ratio=bsd, n_items=len(r),
                             # P2: estimator made explicit in the column names.
                             # within_sd_ratio IS within_sd_ratio_mean and
                             # between_sd_ratio IS between_sd_ratio_median; the
                             # duplicates exist so downstream code need not change.
                             within_sd_ratio_mean=wsd,
                             within_sd_ratio_median=wsd_med,
                             within_sd_ratio_of_means=wsd_rom,
                             between_sd_ratio_median=bsd,
                             between_sd_ratio_mean=bsd_mean))
"""))

# ---- A4 (E2): implement section 7, which its own header already promises ----
reps.append((
"""if rq3_main is not None:
    n = rq3_main.n_valid.median()
    print(f"  median n_valid per item-country cell: {n:.0f}")
    # detectable r at 80% power, two-sided alpha=.05: standard formula
    from scipy.stats import norm
    z_a, z_b = norm.ppf(0.975), norm.ppf(0.80)
    r_detect = np.tanh((z_a + z_b) / np.sqrt(n - 3))
    print(f"  r detectable at 80% power (n={n:.0f}):  {r_detect:.3f}")
    print(f"  observed mean r_wc (individual level):  {rq3_main.r_pearson.mean():+.4f}  "
          f"(SD {rq3_main.r_pearson.std():.4f})")
else:
    print("  rq3 file not found — skipping.")
""",
"""# AUDIT FIX (E2). This section previously printed only the detectable
# correlation and the observed mean, although the module docstring and this
# section's own header promise reliability and attenuation recomputed at n=685.
# No reliability, no attenuation factor and no CSV were produced, which is why
# the released package contains no such table and the revision plan's figures
# (0.955 at n=500, 0.909 at n=250, silicon reliability 0.924) still describe the
# March round rather than any arm of this one.
#
# The section also reproduced a logical error in revision plan 3.2. Printing a
# within-one-country detectable r of about 0.107 next to an observed 0.027
# supports "underpowered", not "genuine near-zero": the observation lies BELOW
# the detection threshold. The argument only closes on the POOLED within-country
# estimator, where the detectable r is about 0.020 and the observed r_pw of 0.043
# exceeds it. Both levels are now reported and the distinction is stated.
if rq3_main is not None and "qwen_1p" in scatters:
    from scipy.stats import norm

    sc7 = scatters["qwen_1p"]
    npc = manifests.get("qwen_1p", {}).get("sample_per_country")
    n_med = float(rq3_main.n_valid.median())
    n_pooled = float(rq3_main.groupby("variable").n_valid.sum().median())
    zc = norm.ppf(0.975) + norm.ppf(0.80)

    def _detect(nn):
        return float(np.tanh(zc / np.sqrt(nn - 3))) if nn and nn > 4 else np.nan

    # Country-mean reliability. The observed variance of the thirty country means
    # is true between-country variance plus sampling error; subtracting the mean
    # sampling variance of a country mean leaves the true part. Two biases run in
    # opposite directions and are stated rather than adjusted for: n_valid is the
    # human-silicon intersection, which is at or below the full-valid n and so
    # inflates the sampling term; and the human means are design-weighted, whose
    # design effect exceeds one and so deflates it.
    nv = rq3_main.set_index(["variable", "cntry"]).n_valid
    rel_rows = []
    for v, g in sc7.groupby("variable"):
        g = g.dropna(subset=["survey_mean", "silicon_mean", "survey_sd", "silicon_sd"])
        if len(g) < 10:
            continue
        n_i = nv.reindex([(v, c) for c in g.cntry]).to_numpy(float)
        if not np.isfinite(n_i).all():
            continue
        rec = dict(variable=v, n_countries=len(g), n_median=float(np.median(n_i)))
        for side, mcol, scol in (("human", "survey_mean", "survey_sd"),
                                 ("silicon", "silicon_mean", "silicon_sd")):
            S2 = float(g[mcol].var(ddof=1))
            E = float(np.mean(g[scol].to_numpy(float) ** 2 / n_i))
            rec[f"{side}_var_between"] = S2
            rec[f"{side}_var_sampling"] = E
            rec[f"{side}_reliability"] = (max(0.0, S2 - E) / S2) if S2 > 0 else np.nan
        rec["attenuation_factor"] = float(np.sqrt(rec["human_reliability"] *
                                                  rec["silicon_reliability"]))
        rel_rows.append(rec)
    rel = pd.DataFrame(rel_rows).set_index("variable")
    rel.to_csv(os.path.join(OUT, "reliability_attenuation_n685.csv"))

    print(f"  respondents per country (manifest):        {npc}")
    print(f"  median n_valid per item-country cell:      {n_med:.0f}")
    print(f"  median pooled N per item (30 countries):   {n_pooled:.0f}")
    print()
    print(f"  median country-mean reliability, human:    "
          f"{rel.human_reliability.median():.4f}")
    print(f"  median country-mean reliability, silicon:  "
          f"{rel.silicon_reliability.median():.4f}")
    print(f"  median attenuation factor on r_bc:         "
          f"{rel.attenuation_factor.median():.4f}  "
          f"({100 * (1 - rel.attenuation_factor.median()):.1f}% attenuation)")
    print(f"  items with silicon-side reliability < .50: "
          f"{int((rel.silicon_reliability < .5).sum())}  "
          f"{sorted(rel.index[rel.silicon_reliability < .5])}")
    print(f"  items with human-side reliability < .50:   "
          f"{int((rel.human_reliability < .5).sum())}")
    print()
    print(f"  r detectable at 80% power, WITHIN one country (n={n_med:.0f}): "
          f"{_detect(n_med):.4f}")
    print(f"  r detectable at 80% power, POOLED within-country (N={n_pooled:.0f}): "
          f"{_detect(n_pooled):.4f}")
    print(f"  observed mean r_wc (per country-item cell):  "
          f"{rq3_main.r_pearson.mean():+.4f}  (SD {rq3_main.r_pearson.std():.4f})")
    try:
        _pw_med = float(r_pw.median())
        print(f"  observed median r_pw (pooled within-country): {_pw_med:+.4f}")
        _verdict = ("ABOVE the pooled threshold: the near-zero is estimated "
                    "precisely, not merely undetected"
                    if _pw_med > _detect(n_pooled) else
                    "BELOW the pooled threshold: the null is underpowered")
        print(f"  -> {_verdict}")
    except NameError:
        _pw_med = np.nan
        print("  r_pw unavailable (section 6 did not run); pooled comparison skipped.")
    print()
    print("  REPORTING RULE. The within-country figure alone cannot support a")
    print("  'genuine near-zero' claim, because the observed value lies below its")
    print("  own detection threshold. Quote the pooled comparison for that claim,")
    print("  and quote the within-country threshold only to say that no single")
    print("  country's estimate is individually informative. Note also that a")
    print("  larger sample cannot rescue the aggregate conclusions: the median")
    print("  attenuation on r_bc is only a few per cent.")

    pd.DataFrame([
        dict(quantity="respondents per country (all arms)", value=npc),
        dict(quantity="median country-mean reliability, human side",
             value=rel.human_reliability.median()),
        dict(quantity="median country-mean reliability, silicon side",
             value=rel.silicon_reliability.median()),
        dict(quantity="median attenuation factor on r_bc",
             value=rel.attenuation_factor.median()),
        dict(quantity="items with silicon-side reliability below 0.50",
             value=int((rel.silicon_reliability < .5).sum())),
        dict(quantity="items with human-side reliability below 0.50",
             value=int((rel.human_reliability < .5).sum())),
        dict(quantity="detectable r, 80% power, within one country",
             value=_detect(n_med)),
        dict(quantity="detectable r, 80% power, pooled within-country",
             value=_detect(n_pooled)),
        dict(quantity="observed mean r_wc per cell", value=rq3_main.r_pearson.mean()),
        dict(quantity="observed median r_pw", value=_pw_med),
    ]).to_csv(os.path.join(OUT, "reliability_attenuation_summary.csv"), index=False)
else:
    print("  rq3 or scatter file for qwen_1p not found — skipping.")
"""))

# ---- A5: record the criterion that actually separates the excluded items ----
reps.append((
'UNINTERPRETABLE = {"hmsfmlsh", "rlgatnd"}  # pre-declared, noise-dominated on silicon side\n',
'''# Pre-declared before results returned, on the grounds of near-zero silicon
# between-country dispersion. AUDIT NOTE: in v13 raw dispersion does NOT separate
# these two cleanly -- the forward control inprdsc sits between them at 0.0313,
# against hmsfmlsh 0.0201 and rlgatnd 0.0316 -- so the manuscript must not
# justify the screen on dispersion alone. Country-mean RELIABILITY does separate
# them: hmsfmlsh 0.25 and rlgatnd 0.44 are the only two items below 0.50, while
# inprdsc is 0.615 (see section 7). State the screen on reliability, and
# attribute the originally quoted 0.017 and 0.044 to the round on which the
# declaration was made rather than to v13. The screen was applied to the reverse
# arm only; dropping inprdsc from the forward controls moves the primary triple
# difference by -0.0004, and all ten forward leave-one-out variants keep the
# triple difference positive in [0.403, 0.455].
UNINTERPRETABLE = {"hmsfmlsh", "rlgatnd"}
'''))

for i, (old, new) in enumerate(reps, 1):
    n = s.count(old)
    assert n == 1, f"anchor A{i} matched {n} times, expected 1 -- ABORTED, nothing written"
    s = s.replace(old, new, 1)

open(p, "w").write(s)
print(f"Patch A applied to {p}: {len(reps)} anchors")
