# Every number in the manuscript, against the v13 tables

The manuscript is written on the **March round at n = 500**. The data are the
**v13 round at n = 685**. So this is not a list of slips: it is the whole
numeric surface of the paper. Nothing below is a judgement about the argument —
each row is a value in the text set against the value in `results/tables/`, with
the table that holds it.

Verdicts: **STALE** = same quantity, different value. **WRONG NOW** = the claim
as written is no longer true. **CHANGES THE SENTENCE** = the wording has to move,
not just the digits.

---

## 3 Data and Method

| § | manuscript | v13 | verdict |
| --- | --- | --- | --- |
| 3.2 | 500 respondents per country | **685** | STALE |
| 3.2 | 21,000 prompts per country | **28,770** | STALE |
| 3.2 | 630,000 per model–prompt condition | **863,100** | STALE |
| 3.2 | 2,520,000 response attempts | **3,452,400** over the four main arms; **10,398,300** over all fifteen | STALE |
| 3.2 | coverage 98.5 / 98.6 / 98.6 / 76.3 per cent | **97.74 / 97.80 / 97.81 / 75.71** (human–model valid intersection) | STALE |
| 3.2 | "simple random sampling without design weights" | unchanged, but the human benchmark **is** design-weighted by `pspwght`; the weighting check bounds the effect at max item-level \|dr\| = **0.103** (`health`) | CHANGES THE SENTENCE |
| 3.4 | "a four-level cumulative backstory experiment" | the ladder has **five** rungs: demo_only (3), minimal (4), ses (13), **political (14)**, full_clean (20) | WRONG NOW |
| Table 1 | four rows, political folded into "+ Full clean" | the political rung is a separate arm and it is where the curve **dips** (0.287). Folding it hides the non-monotonicity that carries the RQ2 argument | CHANGES THE SENTENCE |
| refs | ESS11 **edition 4.2** cited | every run used **edition 4.1** (`ESS11e04_1.csv`, sha256 `a712a833…`). 4.2 exists and was not used | WRONG NOW |
| 3.5 | "Five correlations are used" | still five, but `r_pool` is now reported only as a labelled counterexample and a sixth, the **pooled-within** `r_pw`, is what the design can actually detect | CHANGES THE SENTENCE |

## 4.1 Aggregate recovery

| manuscript | v13 | table | verdict |
| --- | --- | --- | --- |
| 20 of 42 exceed 0.5, **eight** exceed 0.7 | 20 and **9** | `condition_summary.csv` | STALE |
| trust in police 0.81, **trust in legal system 0.80**, general social trust 0.77 | trstplc **0.819**, **trstprl 0.786**, trstlgl **0.778**, ppltrst **0.776** | `qwen_1p_item_r_bc.csv` | WRONG NOW — the second-ranked item is now trust in **parliament** |
| **Twelve** variables have negative r_bc | **eleven** | `condition_summary.csv` | STALE (also in §5.1 and §5.6) |
| most severe: freehms −0.83, **polintr −0.75**, **hincfel −0.66** | freehms **−0.820**, **hmsacld −0.726**, polintr −0.684, hincfel −0.673 | `qwen_1p_item_r_bc.csv` | WRONG NOW — do not rank beyond the top two; ranks 3 and 4 swap between rounds |
| cross-variable agreement in r_bc: r = 0.86 | **0.843** | recomputed | STALE |
| r_prof 0.65 (BG) to 0.91 (FI), mean 0.82, SD 0.06 | **0.685 to 0.911, mean 0.825, SD 0.055** | `country_profile_and_mae.csv` | STALE |
| highest: FI, CH, IE, NO, AT (0.88–0.91) | **exactly right**, 0.882–0.911 | — | ✓ |
| lowest: BG, RS, UA, LV, IL (0.65–0.74) | **BG, RS, UA, IL, LV**, 0.685–0.760 | — | STALE — IL and LV swap |
| nMAE averages 0.19, range about 0.15 to 0.21 | **0.1947**, range **0.161–0.219** | — | STALE |
| r = 0.78 normalised error; r = 0.98 r_prof | **0.758**; **0.968** | recomputed | STALE |
| all 30 countries positive bias, 0.35 (FI) to 1.38 (BG) | **+0.364 (FI) to +1.398 (BG)**, all positive | recomputed | STALE (direction holds) |
| SD ratios 0.47 (**Serbia**) to 0.63 (Portugal) | **0.535 (Estonia) to 0.675 (Portugal)** | recomputed | WRONG NOW — the low end is Estonia, not Serbia |

## 4.2 The country label

| manuscript | v13 | verdict |
| --- | --- | --- |
| −0.07 → 0.52 | **−0.029 → +0.523** | STALE |
| socioeconomic lowers 0.52 → 0.37 | **0.523 → 0.397** | STALE |
| full profile restores to 0.42 | **0.443** | STALE |
| "the full profile recovers less than the label on its own (0.42 versus 0.52)" | **0.443 versus 0.523** — still true | STALE, claim survives |
| domain changes +0.087 efficacy, +0.048 well-being, −0.069 health, −0.194 income | must be recomputed on the five-rung ladder | STALE |
| — | the **political rung (0.287)** is missing from the manuscript entirely, and it is the minimum of the curve | CHANGES THE SECTION |

**The bigger problem in 4.2 is not a number.** The supervisor's letter asks for the
full backstory **with and without country**, because the cumulative ordering makes
the country effect hard to read. That arm exists and is not in the manuscript:
`full_noregion` minus `full_nocountry`, paired at the item level on the same
respondents, gives a forward-item median of **+0.1352** (27/29 improved,
Wilcoxon p = 2.4e-06) against **−0.1756** on reverse items (1/13). That is the
clean identification the current text lacks.

## 4.3 Individual level

| manuscript | v13 | verdict |
| --- | --- | --- |
| mean r_ind 0.011 (Llama 3P) to 0.047 (**Qwen 3P**) | median r_wc **0.0064** (Llama 3P) to **0.0283** (**Qwen 1P**) | WRONG NOW — the best condition changes |
| 15,000 respondents per condition | **20,550** | STALE |
| five-predictor benchmark median 0.21 | **0.2103** | ✓ |
| model mean within-country 0.023 | **0.0271** | STALE |
| r_pool between 0.64 and 0.83 | **0.642 to 0.833** | ✓ |
| Qwen 1P SD ratio 0.56 | **0.605** (mean of ratios; 0.573 median, 0.586 ratio of means — name the estimator) | STALE |
| Llama 3P expands variance, 1.34 | **1.406** | STALE |
| mean r_wc 0.023 (SD 0.080); **794** positive, **349** at p<0.05 | **0.0271 (SD 0.0792)**; **826** positive, **453** at p<0.05 | STALE |
| Israel highest country mean, 0.071 | **0.0774** | STALE |
| best items: life satisfaction 0.138, happiness 0.136 | **stflife 0.146, happy 0.142** | STALE |
| worst: household income feeling −0.135, **subjective health −0.085** | **hincfel −0.148, freehms −0.085, polintr −0.079** | WRONG NOW — health is no longer in the bottom three |
| eleven of 42 negative r_ind | **11/42** | ✓ |

**Missing from 4.3, and it qualifies the section's headline.** Verbal anchoring
raises median r_wc from **0.0283 to 0.0792** — from 13.5 to **31.9 per cent** of
the demographic ceiling. "Individual recovery is negligible everywhere" is no
longer the whole finding: it is negligible *under numeric scales*, and roughly
triples when the scale is anchored. That is a result, not a caveat.

## 4.4 Aggregate–individual dissociation

| manuscript | v13 | verdict |
| --- | --- | --- |
| r_prof vs r_wc = −0.15, 90% CI [−0.44, 0.16] | **−0.2141**, 90% CI **[−0.488, +0.099]** | STALE |
| range-normalised error version: r = **−0.03** | **+0.0659** | WRONG NOW — the sign flips |
| Finland 0.91 / 0.026 | **0.911 / 0.0321** | STALE |
| Israel 0.74 / 0.071 | **0.755 / 0.0774** | STALE |
| Montenegro 0.83 / **−0.014** | **0.845 / +0.0110** | WRONG NOW — the sign flips, and it was the counterexample |
| Qwen 3P −0.48, Llama 1P −0.08, Llama 3P −0.13 | **−0.515, −0.215, −0.220** | STALE |

The conclusion survives — no positive covariation under any condition — but three
of the illustrations have to be replaced, and the argument is now better carried
by the **counterexample count**: 232 of 435 country pairs invert between the
aggregate and individual orderings (`cross_level_counterexamples.csv`), sharpest
at CH against IL.

## 4.5 Robustness

The section is four sentences and the supervisor asked for much more. There is now
a shipped table for it: `robustness_table.csv`, 16 rows and five columns,
including one that says whether any conclusion changes. Nothing in it is in the
manuscript.

## Appendix

| location | issue |
| --- | --- |
| Table A2 | leakage comparison is from the 27-variable run at ~400/country against the 20-variable run at ~500/country. Both predate v13 |
| Table A3 | domain means at n = 500, all twelve rows |
| Table A4 | paired t-tests at n = 500, all four rows |
| Aggregate robustness | "Qwen 1P mean 0.26, median 0.43, 20 of 42 above 0.5" excluding vote → v13 gives median **0.4595**, 10 negative, direction R² 0.6531 |
| Aggregate robustness | "correlate at r = 0.99" for the full-sample check, and "preserves all twelve negative correlations" → **eleven** |
| Evaluation metrics | five measures defined; the chapter now uses six, and the supervisor asked for **fewer**, not more |

---

## What is in the data and not in the manuscript at all

Ordered by how much it changes the paper rather than by size.

1. **The 2 × 2, country × scale anchoring.** A pre-registered triple difference of
   **+0.4230**, 95% CI [+0.1259, +0.7928], on all ten analysable reverse items,
   surviving at **+0.1636** [+0.0507, +0.2671] when the two largest movers are
   dropped. It closes 107 per cent of the forward-reverse gap. This converts the
   paper's weakest passage — "the inversions are probably a scale artefact" — into
   an experiment. `twoxtwo_*.csv`, `fig_2x2_country_by_scale.pdf`.
2. **The country label, identified cleanly.** Rich baseline with and without the
   country sentence, respondent-matched: **+0.1352** forward, **−0.1756** reverse.
   Exactly the comparison the letter asks for.
3. **What "good" would mean.** Three benchmarks the paper has no answer to:
   the leave-one-out regional average (**+0.554** against the model's +0.443 on r,
   and **0.056** against **0.180** on nMAE — the model loses on both), log GDP plus
   region dummies (**+0.618 / 0.052** — loses again), and the demographic ceiling
   at the individual level (achieved / ceiling = **20.4 per cent** like-for-like).
4. **Calibration, separated from ranking.** Lin's concordance decomposes as
   r × Cb: **0.443 × 0.132**, so **87 per cent** of attainable agreement is lost to
   miscalibration, and 11 of 42 items have *negative* concordance. This is the
   sharpest available answer to "what counts as good", which the letter asks for
   explicitly.
5. **A noise floor.** Two identical unseeded runs differ by a median \|dr\| of
   **0.0602** per item, max 0.1854. Nothing smaller than that is interpretable, and
   several effects the manuscript discusses are smaller.
6. **The leakage bound.** The no-country arm still permits country classification
   at **28.72 per cent** against a 3.33 per cent baseline, replicated at 28.99.
   So that arm's +0.3441 is an **upper bound**, and the paper should say so.
7. **The parse failures, diagnosed.** Llama 3P: 5.75 per cent out of scale,
   8.25 per cent above 1000, 8.71 per cent with no digit. Truncation excluded,
   overshoot excluded, scale confusion excluded, backstory echo excluded
   (2.33 against a 0.94 per cent shuffled null). It is a format-compliance
   failure and belongs in the limitations as one.
8. **Reliability and attenuation at n = 685**, which the current §7 promises and
   does not deliver.

---

## Open items for the Mac side (cloud session, 1 Aug 2026, after revise11)

revise11 was a prose-only pass against the supervisor's letter (ex ante
expectations for RQ2, the framing corollary, the backstory inclusion
criterion, caption fixes, the dispersion point promoted into 4.2). Numbers
vs HEAD: IDENTICAL. Three things could not be done in the cloud:

1. **Stale cross-reference in 3.2.** "the sensitivity of the results to that
   choice is reported in Section 4.5" should point to Section 4.6 (robustness;
   4.5 is now the dissociation section). Left untouched because the fix
   changes a numeric token. One-word edit on the Mac.
2. **Sample-size justification still missing.** The letter asks directly why
   only 500 respondents were sampled; the text (now n = 685) never answers.
   plan.md 3.2 has the n = 500-era answer (median attenuation factor 0.955;
   80% power to detect r ≈ 0.125, so the observed near-zero individual
   recovery is a genuine near-zero). Recompute at n = 685 and insert one or
   two sentences in 3.2 or 3.5. Item 8 above is the same request.
3. **Prompt-wording claim needs verification against the pipeline code.**
   3.2 says prompts include "the ESS question text"; plan.md section 9 records
   that 39 of 42 items were presented as the codebook variable label, with
   full question wording only for the three trust items, and that the scale
   renders as "0-10" while the appendix example shows "Scale: 0 to 10". If
   the plan is right, correct 3.2 and the example prompt, and add the
   instrumentation sentence to 5.1 (it strengthens the measurement-equivalence
   reading). If the code says otherwise, plan.md section 9 is stale instead.
4. Housekeeping: README.md's recorded sha256 and word counts refer to the
   pre-revise11 file (new counts: body 7,572, total 12,639); Figure 2's
   caption now uses (a)/(b)/(c) lettering, so check the figure artwork
   matches; rebuild the docx and re-run verify_manuscript.py.

### Resolution (Mac side, 1 Aug 2026 evening, revise12)

All four items above are closed:

1. Section cross-reference corrected to 4.6.
2. Sample-size justification inserted in 3.2 at n = 685, from
   `reliability_attenuation_summary.csv`: reliability 0.99 human / 0.94
   silicon, median attenuation 0.97, detectable r at 80% power 0.11 per cell
   / 0.02 pooled.
3. Verified against `silicon_sampling_extended_v12.py`: the question is
   `var_info.get('question', var_info['label'])` and only the three trust
   items define 'question', so plan.md section 9 was right. 3.2 rewritten,
   the appendix example aligned with the template verbatim ("Scale: 0-10",
   full closing line), and the instrumentation sentence added to 5.1.
4. README updated (sha256, counts, expected checker line now 447/447);
   Figure 2's caption already matched the artwork's (a)/(b)/(c) labels.

`verify_manuscript.py` after revise12: 447 tested, 447 traced, PASS.

## Open items for the Mac side (cloud session, 1 Aug 2026, after revise13)

1. **Noise-floor contradiction on the ladder steps.** 4.2 says "Every one of
   those steps exceeds the 0.060 noise floor" and 5.2 says "every step change
   exceeds the noise floor", while 4.6 says "The individual steps of the
   backstory ladder do not, so the ladder is read as a shape." The three
   cannot all stand. The median-level steps (0.523 to 0.397 to 0.287 to
   0.443) do exceed 0.060, but 0.060 is a median of per-item run-to-run
   differences, so 4.6 may intend a per-item claim. Decide against the data
   which comparison is meant and reword the losing sentence; if 4.6 is the
   per-item claim, say "at the level of single items" explicitly.
2. Optional: 4.4 cites a package filename in the main text
   ("rq3_cell_summary.csv"). Consider moving the pointer to the appendix;
   the fix removes digit tokens, so it is a Mac-side edit.
3. Housekeeping: README counts and sha256 are pre-revise13 again (new counts:
   body 7,709, total 12,819); rebuild the docx (v17) and re-run
   verify_manuscript.py (no numeric change expected: 442 tokens identical).

### Reviewer comments in v17 (Word comments), resolved 1 Aug 2026 evening

21 comments. Prose items closed by revise14 (progression claim, four-gaps
echo, transition, shorter contributions paragraph per the inline note,
"for example", battery-tied examples, alignment -> correspondence sweep,
topic-sentence audit, Section 5 reviewed no-change). Formatting items closed
in build_docx.py (1.15 line spacing, non-italic H2/H3, justified TNR 10pt
captions, booktabs tables). Figure items closed in the package figure code:
all nine figures rebuilt vertical/full-width, grayscale, two font tiers
(12/10), overlaps removed; Figure 4 reduced to two panels on a shared country
axis; Figure 5 gained the clean with/without-country ablation panel; Figure 6
stacked; figure notes updated by revise15. Rebuild after the change: 63/51
tables, 90 artefacts, MANIFEST 288 PASS; verify_manuscript 447/447 PASS.

### Reviewer comments in v18 (Word comments), resolved 2 Aug 2026

10 comments, all on figures and layout, plus one substantive framing question.
c1 (should the ordering-sensitive ladder or the with/without contrast carry
RQ2): Section 4.2 reordered by revise16 so the paired country-label ablation
is the identifying estimate and leads, with the ladder following as the
dose-response shape; Figure 5 now shows the ablation as panel (a), the ladder
as panel (b), the dispersion panel retired to the figure data csv. Layout
comments: every figure rebuilt compact and document-wide (fig1 one third
smaller with closed boxes and less text; fig2 a/b top row + c full-width
bottom; fig3 items on the x-axis with only the extremes and euftf named;
fig4 one-strip-per-metric with min/median/max labeled; fig6 a full-width +
b/c side by side; A2 side by side; gincdif label shortened). Figure captions
now placed below the artwork in the Word build. All renders inspected for
overlaps. Rebuild: 63/51 tables, 90 artefacts, MANIFEST 288 PASS;
verify_manuscript 447/447 PASS.

### Reviewer comments in v19 (Word comments, 18), resolved 2 Aug 2026

Rendering defects fixed in build_docx.py: every table carried a stray
"[/TABLE]" row (c5, a parser bug); Cb now renders as C with a subscript (c7);
figures embed at natural 300 dpi size, so in-figure type prints at its true
~10 pt (c0), and all figure fonts are Times New Roman first (c0). Table 1
deleted (c4); recovery defined at the top of 3.5 (c6); readable names for the
trust items (c2); the arrow in 3.3 replaced with words (c3); the anchored arm
explained in the Figure 2 note (c9); euftf named (c10); the Figure 4 note
states its purpose (c12); 4.4 points to Appendix Figure A3 (c16). Figure 2 is
(a) full-width over (b)+(c) (c8). Figure 5 rebuilt as the per-item
with-against-without display, every item drawn (c14), and the cumulative
ladder moved to a new Appendix Figure A4 (c13); no arm is re-cut or re-run —
the three blocks are the only ones with with/without arms, which the note now
says. Figure 6's lower row uses the full width (c15). Assessed, no change:
caption placement already follows convention, figure captions below and table
titles above (c1); Figures 3 and 4 both load-bearing, order kept (c11);
appendix figures remain reference material, no swap beyond the ladder (c17).
verify_manuscript: 448/448, 10 figures, 6 tables, PASS.

### Reviewer comments in v20 (Word comments, 11), resolved 2 Aug 2026

Caption format unified (short title line + Note line; Figure 1 split to
match). 3.4 rewritten so the with-against-without contrasts lead and the
cumulative sequence follows as a supplement whose groupings are named as a
property of the arms as run (the political step adds one variable);
regrouping the ladder would require generating new arms on the cluster and
is priced separately for the user, not run. Figure 4: full country names,
extremes only, taller strips with separated points; the note's verdict
sentence removed (main text carries it). Figure 5 taller with separated
points. pspwght glossed; two overlong sentences split; Section 5 read
sentence-by-sentence and its one true repeat of 4.3 compressed to a pointer.
verify_manuscript: 449/449, 10 figures, 6 tables, PASS.

### revise35, the Discussion pass against the letter (2 Aug 2026)

Seven anchored edits, no numeric claim changed. `selfcheck.py` reports seven
added tokens and none removed: `2.2`, `4.1`, `4.2`, `5.3` x2 are section
cross-references, and `2011`, `1995` are the in-text years of Norris and
Diener et al., both already in the reference list and both already cited in
2.1 for the same two domains. Nothing here needs tracing to a shipped file.

What changed: an ex ante expectation pair added to 2.1 ahead of RQ1; the
aggregate standard of adequacy moved forward into 3.5 as a third reference
point beside the noise floor and the individual ceiling; the rind definition
cut from 3.5 (unused in the main text, defined in the appendix); 5.1 given
the calibration discrimination and the rational-cue-use objection; 5.2 given
a post hoc reading of the item heterogeneity direction does not explain.
Body 7,831 to 8,201 words.

Deliberately not done: the global "moderate / limited / negligible"
terminology sweep from plan.md 5.5. "limited" and "negligible" are already
benchmark-anchored at every use, and "moderate" now has a referent fixed in
3.5. Flagged here so the omission is a decision on the record.

### revise36, the three approved fixes from the revise35 audit (2 Aug 2026)

Six anchored edits. RQ4 merged into RQ3 as its third clause (plan.md fixed
three questions; the manuscript had kept four): the freestanding RQ3
statement moves from 2.3 to the end of 2.4, 4.5 opens as the completion of
RQ3, and its closing sentence names the cross-level test. One sentence in
4.5 says why no figure accompanies the bound; one sentence in 4.6 says what
the sixteen checks cover. The selfcheck diff is label tokens only: four RQ4
tokens removed, two RQ3 mentions added. No data claim changed. Body 8,201
to 8,254 words.

Left alone by the same decision: the 4.1 heading, the prose metric
hierarchy in 3.5, and the "moderate / limited / negligible" wording.

### revise37, RQ3 and RQ4 separated again (2 Aug 2026)

User's decision after reconsideration: the letter's complaint was missing
expectations, not the count of questions; the supervisor called the
cross-level question interesting; 2.4's competing predictions deserve their
own question; the dissociation is the title claim. Four anchored edits
revert the revise36 merge, and RQ4 gains the selection clause from plan.md
("and can aggregate performance select countries for individual-level
use?") so the pairwise count in 4.5 answers it directly. The no-figure
sentence in 4.5 and the coverage sentence in 4.6 are kept. The selfcheck
diff is RQ label tokens only, the mirror image of the revise36 entry. Body
8,254 to 8,260 words.

### revise38, the six v29 Word comments (2 Aug 2026, Mac session)

Numbers changed deliberately; this is the Mac side and every new token was
traced before writing. `verify_manuscript.py` reruns PASS: 488 tested, 488
traced, 11 figures, 6 tables (was 450/450/10/6).

- Comment 0: the 42 outcome measures are "items" everywhere; "variables" is
  reserved for the 20 backstory predictors. 22 replacements, no number moved.
- Comment 1: Figure 2(a) draws one marker per item; the direction split is
  Figure 3's job. fig2_recovery.py, regenerated on the cluster.
- Comment 2: Figure 5 kept as is; the three rows are the three ablatable
  blocks (the only with/without arms that exist). One sentence added to the
  note saying why the forward/reverse split cannot be collapsed.
- Comment 3: fig_2x2.py panel (a) taller and full page width, ylabel at the
  left edge. Regenerated on the cluster.
- Comment 4: new Figure 7 (fig_crosslevel.py, fig7_crosslevel_ranks): the
  two country rank orderings joined per country; asserts the drawn 232
  discordant pairs equal the manuscript's number. FIGURE_MAP row 7 in both
  copies; selfcheck figure expectation now 11.
- Comment 5: Table A5 moved into 4.6 as Table 1 with every truncated cell
  completed from results/tables/robustness_table.csv (16 mid-sentence "…"
  cells were shipping). All 68 numeric tokens in Table 1 trace to that CSV
  (checked mechanically). Appendix Table A6 renumbered A5.

Package files edited (fig2_recovery.py, fig_2x2.py, fig_crosslevel.py new,
rebuild.sh, FIGURE_MAP.csv): the manifest self-check is broken until the
revise39 ladder rebuild reseals it, as intended.

### revise39, the v30 review round (2 Aug 2026, Mac session)

- Figure 5 rebuilt in the package (fig_ladder.py): panel (a) is the per-item
  country-label contrast on all 42 items, the display the user asked for;
  panel (b) keeps the three-block medians. The script asserts 27/29 and 1/13
  against the drawn data. Caption rewritten and moved up under the paragraph
  that reports the contrast.
- The robustness table returns to the appendix as Table A5 (completed cells
  kept); the extraction table is A6 again. The selfcheck "removals" are the
  table's tokens crossing the References boundary that selfcheck uses, plus
  label tokens; verify_manuscript covers the whole file and PASSES 489/489.
- Terminology: arm vs condition fixed in 3.2 (two spots), "richer profiles"
  for "richer personas" (5.1), "backstory wording" for "persona wording"
  (5.3), "per-item between-country correlations" for the stray "item-level
  country correlations" (appendix).
- Numbering audit found and fixed two real defects: Figure 6 was never
  referenced in running text (pointer added in 4.3), and Table A4 sat after
  A6 at the end of the file (moved before the Aggregate robustness checks
  section; appendix order is now A1-A6).
- Figures 3 and 4 NOT merged (user question): different units of analysis
  (items vs countries); a merged figure would recreate the letter's
  interpretability complaint.

### revise40, 4.1 slimmed to two figures (2 Aug 2026, Mac session)

The user's point restated: the backstory contrast sat behind three 4.1
figures. The country-profile figure is retired as a separate figure and its
two strips are absorbed into Figure 3 as panels (c) and (d) (fig3_levels.py
regenerated on the cluster; fig_country_profile.py still runs for its data
outputs). Renumbering: contrast = Figure 4, the 2x2 = Figure 5, cross-level
ranks = Figure 6. Ten figures (six main, four appendix), FIGURE_MAP
renumbered in both copies, selfcheck expects 10, verify_manuscript PASSES
489/489, 10 figures, 6 tables. This reconsiders the earlier "do not merge 3
and 4": the user's actual goal (the key experiment one figure earlier) makes
the merge the right trade, and the four panels are one topic (descriptive
recovery: items, then countries).

### revise41 and the six LOGO arms (2 Aug 2026, Mac session)

The user's clarified request for Figure 4(a): compare the COUNTRY LABEL
against every other backstory group's contribution to aggregate recovery,
rows = backstory groups. That needs leave-one-group-out arms that do not
exist yet. Six were prepared and submitted:

- v12.4 pipeline modes (full_noascriptive G2, full_nosocioecon G3,
  full_nohousehold G4, full_nocivic G5, full_nomarkers G7, full_nodomicil
  G1-remainder), using the G1-G7 grouping frozen in the pipeline header.
  Country and the NUTS code stay in every arm; full_nodomicil renders the
  region code without the domicile phrase through a branch gated to that
  mode alone.
- Regression (logo_regression.py, on the cluster, 750 respondents x 30
  countries): all 15 pre-existing modes render byte-identically under the
  new module; each LOGO mode drops exactly its own group's sentences.
- Slurm jobs 30817-30822, queued behind 30816, n=685, ~5-6h each. The
  rebuilt Figure 4 (single panel, 9 block rows: country, region, political
  + the six new; fwd/rev split kept) replaces BOTH current panels when the
  arms land; the per-item country lollipop moves to the appendix then.

revise41 (this pass): 4.1 slimmed at the panel level. Figure 2 drops the
threshold-count panel (counts stay in text, tied to the pre-declared
thresholds); Figure 3 drops the error strip (the metric-dependence numbers
were already prose). 4.1 is five panels, was seven at v32. Token diff is
the retired panel-note tokens (0.50, 0.70, 42). verify_manuscript PASSES
489/489, 10 figures, 6 tables.

### v12.5: the LOGO partition nests in the ladder tiers (2 Aug 2026)

The user asked whether per-variable LOO would be cleaner and whether the
six-group split was arbitrary. Per-variable is NOT cleaner: the profile
holds near-duplicate pairs (age/birth year, education level/years,
household size/children), so single-variable removals measure duplication,
not information. The real defect was that the markers block (union,
discrimination, internet) straddled ladder tiers 4 and 5, giving the paper
two conflicting taxonomies. Fixed before the arm ran: 30821 cancelled,
v12.5 adds full_nomembership (union + internet, tier 4) and
full_nominority (discrimination, tier 5), jobs 30823/30824. The partition
of the 20 profile variables is now exhaustive, exclusive, and nested in
the ladder tiers. Regression: 15 pre-existing modes byte-identical.

### v12.6 and the full overnight queue (2 Aug 2026, ~21:00)

User approved: the two-margin design (leave-one-out AND add-one for every
block, full n), no pilots; then the swapped-label falsification arm and the
Llama replication of the country pair.

- v12.6 modes (regression: 22 pre-existing modes byte-identical across 750
  respondents x 30 countries; every new mode's sentence set checked
  exactly): 7 add-one arms (minimal + one block), the age/birth-year twins
  (the user's surface-form point: an LLM need not treat "60 years old" and
  "born in 1966" as the same fact), country_only (label with no
  demographics), full_noincome (single variable vs its block), and
  full_swapcountry.
- The swap arm: base full_noregion, country sentence mapped through ONE
  fixed derangement of the 30 countries (numpy default_rng(888), first
  no-fixed-point draw, hard-coded as SWAP_MAP in the pipeline). Results
  keep true cntry, so recovery is scored against labeled AND true country.
  Design rationale: a placebo/negative-control test (Eggers et al. 2024
  AJPS); no prior work matches swapped labels against the named country's
  real survey means (closest: Kamruzzaman & Kim 2024 nationality personas;
  counterfactual identity-swap audits).
- Jobs 30825-30843 queued behind the two running LOO arms: 5 LOO rest,
  7 add-one, 4 probes, swap, Llama pair. ~2 days on two GPUs.
- Integration after landing: Figure 4 becomes the two-margin block figure
  (each block: add-one bar + leave-one-out bar), the cumulative ladder
  (Appendix A4) is DELETED as redundant per user decision pending final
  look, swap becomes its own display, Llama pair goes to robustness.

### Integration prep, drafted while arms run (3 Aug 2026, 13:30)

Decision: no manuscript or figure-pixel edits until all arms land (the
manuscript stays self-consistent at every commit). Prepared instead:

- code/analysis/two_margin_table.py: per-block add/remove margins, per item
  and per block, asserts the country remove margin reproduces
  clean_ablations.csv (verified on landed arms: 0.197/-0.176 exact).
  Probe run on 10 landed contrasts confirms the shapes: country add margin
  fwd +0.731 / rev -0.439; politics add -0.122 vs remove -0.005 (the
  dilution contrast, now quantified); every other landed block within or
  near the 0.060 band.
- code/analysis/swap_analysis.py: r_label vs r_true per item, SWAP_MAP
  derangement asserted, human-mean cross-check asserted.
- revise42_DRAFT.py in the repo: 3.4 design prose, refs (Eggers et al.
  2024; Kamruzzaman & Kim 2024), guarded with sys.exit so it cannot run
  before integration.

### v12.7: the membership block split for attribution (3 Aug 2026, ~20:30)

The user's objection: union + internet is a residual category (tier 4 minus
the SES triad), and it is now the one non-country block with a positive add
margin, so its composition is load-bearing. Two single-variable ADD arms
(minimal_union, minimal_internet) were added and submitted (jobs 30844,
30845, queue tail, ~2h each). Single-variable ablation is clean here:
neither variable has a redundant twin. Regression: 34 existing modes
byte-identical; both new modes render exactly minimal + one sentence.
Both country-proxy stories are plausible ex ante (internet penetration and
union density are both strong country gradients); the split decides.

### revise42, the two-margin integration (5 Aug 2026, cloud session)

Arms landed 4 Aug in `chuyao-wang/silicon-replication` (f7ca4b2); the cloud
session verified all 21 v12.6/v12.7 modes complete (42 x 30 x 685) and
applied the approved integration outline. EVERY number below is
cloud-computed from the landed CSVs and needs Mac-side tracing before the
docx is canonical. verify_manuscript.py will fail until the package ships
the corresponding analysis outputs (two_margin_by_block.csv, swap_scores.csv
already exist as scripts in the package; the rest need adding).

New numeric tokens, by section and source:

- 3.2 sample size: 400 (subsample plateau), 0.98 (human split-half median
  r_bc, unweighted; Mac should recompute with pspwght, cloud bound on the
  weighting effect: unweighted vs weighted country means agree at median
  r 0.9951, median abs diff 0.034 scale points). Source: raw qwen_1p.
- 4.1 pooled decomposition: 96.5% (share of pooled covariance from item
  means), 0.515 (pooled r after item centering). Source: scatter qwen_1p.
- 4.2 margins (Fisher z, medians, two_margin_by_block.csv): +0.73/−0.44
  (country add), +0.09/+0.08 (membership add/remove fwd), +0.13/+0.06
  (internet/union single adds fwd), −0.12 (politics add fwd), −0.13/−0.06
  (SES add/remove fwd), −0.055 of −0.061 (noincome vs SES block remove).
- 4.2 swap (swap_scores.csv): 0.40 (r_label fwd median 0.396), 0.56
  (full_noregion fwd 0.561), 0.34 (r_true fwd 0.340), 0.40 (full_nocountry
  fwd 0.403; NOTE both round to 0.40, two tokens), −0.07 (permutation
  baseline: human means labeled vs true, median r).
- 4.2 Llama pair: +0.25/−0.20 (fwd/rev medians, llama noregion vs
  nocountry), 25 (of 29 forward items improving).
- 5.1 variance: 0.69 (country_only within-country silicon SD, median),
  0.88 (full arm), 2.08 (human). Source: raw files.
- 5.2 signal correlation: 0.46, 0.012 (Pearson r and p, forward items,
  r_bc vs scale-normalized human between-country SD; Spearman +0.41
  p=0.026; bootstrap CI touches zero; quoted as supporting evidence only).
- appendix Block ablations: +0.09 (largest non-country forward margin).

Open flags for the Mac side:

1. membership_rep raw AND scatter are byte-identical to minimal_membership
   (md5 match); only manifest tag/timestamp differ. A true unseeded rerun
   cannot produce identical files. Check the cluster outputs; until then
   the rep arm is not evidence of generation-noise stability. The 0.060
   noise floor comes from the earlier repeat pair and is unaffected.
2. Figure 4 pixels are a cloud draft (scratchpad fig4_two_margin.py,
   rendered into figpng/fig4_backstory_ladder.png). Rebuild the canonical
   PDF/PNG from the package figure code and repoint the FIGURE_MAP
   producer. figA4_cumulative_ladder and the stale fig5_country_profile
   PNGs are deleted; FIGURE_MAP row A4 removed; selfcheck expects 9.
3. The 3.2 sentence "13,028,700 across all seventeen arms" is left
   unchanged: the cloud could not reconstruct the 17-arm accounting.
   Update the arm count and response total after integration (21 new
   full-battery modes at 863,100 attempts each).
4. Word-count targets: body grows by the margins/swap paragraphs net of
   the ladder deletion; recheck against the +27.1% baseline.
Exact selfcheck token diff for this pass (vs cb175b0): removed 0.060 x1,
0.32 x1, 0.42 x1, '4' x2 (the two Figure A4 pointers). Added tokens as
listed above plus the incidental section/count/year tokens: 0.028, 0.210,
0.52, 0.06, 1, 20, 25, 29 x2, 30, 400, 2024 x2, 4.2 x3, 5.4 (new
instances of existing values or pointers, no new claims).

### revise43 and the cloud verification pass (5 Aug 2026, cloud session)

1. The 17-arm accounting RECONSTRUCTS EXACTLY: 13 full-battery arms
   (qwen/llama x 1P/3P, demo_only, minimal, ses, political,
   minimal_politics, minimal_politics_econ, full_nocountry, full_noregion,
   full_nopolitical) x 863,100 + 4 22-item arms (anchored x3, repeat run)
   x 452,100 = 13,028,700; 3-country pilots excluded. Adding the 23 landed
   full-battery arms gives 40 arms and 32,880,000. The 3.2 sentence is
   updated by revise43.py; Mac confirms the convention only.
2. cloud_analysis/ added to this repo: two_margin_by_block.csv,
   two_margin_by_item.csv, swap_scores.csv, revise42_numbers.csv, and
   cloud_verify_revise42.py, which recomputes EVERY revise42/43 token from
   the silicon-replication checkout and asserts it against the manuscript.
   Run on 5 Aug: ALL 28 quantities traced. It caught and fixed one
   rounding error (internet add margin is +0.13, not +0.14; median
   +0.1347). Mac side: copy these into the package results tree and
   extend verify_manuscript.py's map so the canonical pass covers them.
3. Duplicate sweep across all 43 raw archives: the ONLY byte-identical
   pair is membership/membership_rep, so the defect is a single copy or
   packaging error on that arm, not systemic. Cluster-side check of the
   rep job's output directory still needed.

### Items 1 and 2 closed cloud-side with the 2 Aug package snapshot (5 Aug)

The user supplied silicon_replication_snapshot_20260802_v33.tar.gz, which
carries code/audit/verify_manuscript.py and code/lib/figstyle.py.

1. CANONICAL NUMBER PASS RUN IN THE CLOUD: with cloud_analysis/*.csv copied
   into the snapshot's results/analysis, verify_manuscript.py on the current
   manuscript reports 518 tested, 518 traced, 9 figures, 6 tables, PASS.
   revise42_numbers.csv gained the derived-quantity rows the checker needs
   as cells (subsample curve, plateau n, Llama improve count). Mac reruns
   the same pass on the live package after copying cloud_analysis/ in.
2. FIGURE 4 REBUILT IN THE PACKAGE STYLE: cloud_analysis/fig_two_margin.py
   (figstyle-based, panel (a) code path identical to fig_ladder.py, panels
   (b, c) the two margins; check_layout clean; asserts 27/29, 1/13, and the
   country remove margin against clean_ablations). figpng PNG replaced with
   this render; figure data CSVs ship in cloud_analysis/. Mac: move the
   script to code/figures/, retire the ladder half of fig_ladder.py, rebuild
   the PDF on the cluster, reseal the manifest.
Remaining Mac-only: the membership_rep cluster check (DISCREPANCIES above).

### revise45 token note (5 Aug)

revise45's commit message says "numbers IDENTICAL"; the actual selfcheck
diff is: added '3.4' x1 and '4.2' x1. Both are section pointers in the two
new cross-reference sentences (3.3 -> Section 3.4; 5.1 -> Section 4.2),
not numeric claims. Nothing to trace.

### membership_rep flag RESOLVED: engine-default determinism, not packaging
(5 Aug, cloud + cluster forensics)

Cluster check: job 30846 ran 1h51 and wrote its own five rep files at
19:08; the raw is byte-identical to the 3 Aug original ON THE CLUSTER
(both decompress to 49b309aa...), so the earlier packaging-error
hypothesis is WITHDRAWN. Cause: since P0-C the pipeline omits the vLLM
seed when --gen_seed is unset, the installed vLLM defaults an omitted
seed to 0, and the rep submission did not vary --gen_seed, so the rerun
was deterministic. The pipeline's own --gen_seed help text warns exactly
this ("a seeded rerun may be near-deterministic and measure nothing").
Consequences: no data are wrong; the 0.060 noise floor (legacy repeat
pair, genuinely stochastic) stands and is conservative for the
deterministic v12 arms; the 3.2 sentence about the unseeded repeat pair
describes the legacy pair and stays. Options recorded for the user:
rerun the rep arm with --gen_seed 889 for a true full-battery noise pair
(recommended; also pins the noise-equality k in the 5.1 decomposition),
or retire the rep arm as evidence. Manuscript untouched pending that
decision; "generation is unseeded" is on the settled-facts list.

### revise46: the true replicate pair, and the 5.1 argument it refutes
(5 Aug 2026, cloud session; job 30847, silicon-replication c59477c)

Job 30847 reran minimal_membership with --gen_seed 889 (1h51, COMPLETED).
Verified on arrival: manifest gen_seed = 889, n = 20,550, and the raw file
now differs from the original arm (cfbc4bc8... against 49b309aa...). This
is the first true replicate pair on the FULL 42-item battery; the shipped
0.060 floor comes from the older 22-item anchored pair.

WHAT IT REFUTED. revise42's 5.1 sentence read the rise in within-country
silicon SD (0.69 country-only, 0.88 full, against 2.08 human) as
profile-induced variation, i.e. as evidence that respondent detail is used.
The replicate pair measures the generation noise directly, from the paired
difference between two runs of the same arm on the same 20,550 respondents:

  implied noise SD (median item)      0.8619
  the arm's within-country SD          0.8646  -> noise share 95.4%
  country_only arm as a noise proxy    0.6889  -> k = 1.280

So the country-only arm UNDERSTATES the noise by 28% (a shorter prompt
generates less variably), and the SD rise is mostly extra generation noise,
not profile-induced signal. The sentence is withdrawn and replaced by the
direct measurement (95% of the arm's within-country variance is noise).
The 5.1 aggregate argument is untouched: "used, and used badly" rests on
the negative add margins in 4.2. The earlier sensitivity band for k
(27-60%) is superseded by the measurement for that arm; the full arm still
has no replicate, so no share is claimed for it.

WHAT IT ADDED. 4.6 now reports the pair as a second stability check:
median |delta r_bc| 0.0289, max 0.2420, 8 of 42 items beyond 0.060, one
sign reversal. The two largest movers are rlgatnd (0.242) and hmsfmlsh
(0.173), which are exactly the two items the Table A5 dispersion screen
already declares uninterpretable (reliability 0.44 and 0.25). The
interpretive band is KEPT at 0.060, the conservative choice.

Also: the membership add margin reproduces across the pair (+0.092 forward
in run A, +0.140 in run B; 23/29 and 22/29 items up), so the 4.2 claim
holds; the arm count becomes 41 arms and 33,743,100 attempts
(13x863,100 + 4x452,100 + 24x863,100), since the replicate is a delivered
arm like the 22-item repeat run already counted.

New numeric tokens: 95% (noise share), 0.029 and 0.242 (pair median and
max), 33,743,100 and forty-one (arm accounting); removed 0.69, 0.88, 2.08,
32,880,000. cloud_verify_revise42.py extended to recompute all of these
and rerun: 28/28 traced. Canonical verify_manuscript.py against the 2 Aug
package snapshot with cloud_analysis merged in: 523 tested, 523 traced, 9
figures, 6 tables, PASS.

THE MEMBERSHIP_REP FLAG IS NOW CLOSED. Cause was the omitted engine seed
defaulting to 0; fix was --gen_seed 889; the arm now measures what it was
designed to measure.

### Data package rebuilt and sealed (5 Aug 2026, cluster)

`build_package.py --with-raw --out ~/silicon_package_20260805` on the
cluster: 41 conditions discovered, every one complete on scatter, rq1,
rq3, raw and manifest; 331 files included, 1 recorded but not included
(the ESS microdata, which the licence does not allow redistributing);
DATA_MANIFEST.csv 333 lines = header + 332 records; No gaps; 2,765.8 MB on
disk, 401 MB as silicon_package_20260805.tar.gz. `fig_two_margin.py` was
added to CODE_FILES so the package carries the source of Figure 4 (code
now 35 files). The 41 conditions match the manuscript's 41 arms exactly,
which is an independent check on the revise43 accounting.

`two_margin_table.py` and `swap_analysis.py` were rerun on the cluster
against the cluster's own results/: every block margin reproduces the
cloud values to the printed precision (country add +0.731/-0.439, remove
+0.197/-0.176; swap 0.396 labelled against 0.340 true), and
`fig_two_margin.py` renders clean under the package's layout checker.

Still outstanding, manuscript package only (`~/silicon_chapter`): merge
cloud_analysis/*.csv into results/analysis, move fig_two_margin.py into
code/figures/, repoint the FIGURE_MAP producer column, rerun rebuild.sh,
reseal the manifest.

### revise50: Figure 4 is backstory blocks only (5 Aug 2026)

The per-item panel had SURVEY ITEMS on its y axis while the block panels had
BACKSTORY VARIABLES, so one figure carried two different objects. The
per-item panel becomes Appendix Figure A4 (new PNG
figA4_country_item_contrast.png; FIGURE_MAP row added; selfcheck now expects
10 figures). Figure 4 keeps the two block panels, retitled "none but this
block" and "all but this block", and they now share ONE x scale: they did
not before, which is the fault the supervisor found in Figure 3.

New numeric tokens, all already traced elsewhere in the manuscript and
recomputed by cloud_verify_revise42.py: 0.73 and 0.20 (the country label's
two forward margins, now compared in one sentence explaining why the add
margin exceeds the remove margin), plus section pointers 4 x2 and 5.4.
Removed: one instance of 42. The new sentence makes a claim the data
support and Section 5.4 already bounds: the full profile carries country
information in the composition of its other variables (classifier recovers
the country 28.7% of the time against a 3.3% baseline), so removing the
label costs less than adding it gains.

Also in this pass: 4.1's three defensive sentences about the direction
coding move to 4.6, next to the direction row in Table A5; 3.3 goes back to
one paragraph; four opening sentences that announced a count now state the
finding.

### revise51: the research questions regrouped; the swap arm gets a figure
(5 Aug 2026)

The supervisor asked for "a small number of well-motivated questions,
analyses that directly answer those questions". Four questions were already
there, but RQ1 and RQ2 each carried two, and Section 4.3 answered the tail
of both, so no results section answered exactly one question. The count and
the order are unchanged; each question loses its tail and the tails become a
question of their own:

  RQ1  how accurate, how uneven across items                  -> 4.1
  RQ2  where the signal comes from, does richer help          -> 4.2
  RQ3  country knowledge or response scale (NEW, from the
       tails of the old RQ1 and RQ2)                          -> 4.3
  RQ4  individual recovery, framing, and whether it tracks
       the aggregate (old RQ3 + old RQ4)                      -> 4.4

Sections 2.3 and 2.4 merge (both motivate RQ4); 4.4 and 4.5 merge (both
answer it); 2.2 gains RQ3's ex ante prediction, which is what makes the
scale account falsifiable before the 2x2 is run. The introduction's four
tests now run in the order of the four questions.

FIGURES RENUMBERED. The swapped-label arm gets Figure 5 (new,
fig5_swap_country.png, producer cloud_analysis/fig_swap.py); the 2x2 becomes
Figure 6 and the cross-level ranks Figure 7. FIGURE_MAP updated; selfcheck
now expects 11 figures.

New numeric tokens, all recomputed by cloud_verify_revise42.py and already
in the text elsewhere: 0.56, 0.40 x2, 0.34 (the four swap conditions, now
also in the Figure 5 note) and the figure numbers 5, 6, 7. Removed: the
section numbers 2.4 and 4.5, which no longer exist as headings.
verify_manuscript against the package snapshot: 528 tested, 528 traced, 11
figures, 6 tables, PASS.

### revise52: three questions, three results sections; the swap result read
from the paired test (5 Aug 2026)

RQ2 and RQ3 merge. The scale format supplies no cross-national signal of its
own, so "label or scale" misdescribed the pair. What the 2x2 shows is that
the label supplies the signal and the format decides the sign it takes: one
mechanism, one question. RQ2's enrichment clause goes with it, because if
the signal is the label then richer profiles not helping is the same finding
from the other side, and Figure 4 still shows it. The old RQ4 becomes RQ3.

  RQ1  how accurate, how uneven across items                  -> 4.1
  RQ2  where the signal comes from, and what sets its sign    -> 4.2
  RQ3  individual recovery, framing, tracking the aggregate   -> 4.3

Sections 4.2 and 4.3 merge under "The country label supplies the signal; the
response scale sets its sign"; 4.4 becomes 4.3 and 4.6 becomes 4.4. All
cross-references follow. No table, figure, or result moves.

SWAP RESULT CORRECTED. The old sentence read a transfer off the medians:
scored against the named country 0.396, against the respondents' own country
0.340, no label 0.403. Medians are not additive. The paired per-item test on
the 29 forward items says:

  swap-own minus no-label      -0.096  [-0.142, -0.003]   excludes zero
  swap-named minus swap-own    +0.012  [-0.068, +0.122]   includes zero

So the arm shows the label is read and read as a country: a wrong name costs
the respondents' own country more than no name does. It does not show
recovery handed to the country named. The text and the Figure 5 note now
make the weaker claim, which is the one the data support and still separates
the label account from a compositional one. The summing-up sentence in 4.2
changes with it ("replacing the name degrades the placement it was
carrying").

NEW NUMERIC TOKENS: -0.096, -0.142, -0.003, +0.012, -0.068, +0.122. All six
are recomputed from data/summary by cloud_verify_revise42.py (now 34
quantities), bootstrap over items, seed 888, B = 10000. REMOVED: the section
numbers 4.3 and 4.6, which no longer exist as headings, and repeated copies
of 0.34/0.40/0.56 in the Figure 5 note.

INTERVALS, 3.5. One sentence merged into the paragraph that defines the two
correlations: the countries are the full ESS set and the items a fixed
battery, so these correlations estimate no wider parameter and carry no
interval; uncertainty is reported where it exists, in the sampling error
inside each country mean and in the run-to-run variation of Section 4.4. No
confidence intervals were added to Figures 4 or 5, by decision: the item set
is fixed and the bars are not estimates of a population quantity.
