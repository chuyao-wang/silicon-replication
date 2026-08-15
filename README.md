# Silicon Sampling Captures Country-Level Assumptions, Not Individual Attitudes

Replication package for the cross-national evaluation of silicon sampling
against European Social Survey Round 11: 30 countries, 42 items, 685
respondents per country, 43 generation arms, 34,647,300 response attempts.

Download the whole thing with the green **Code → Download ZIP** button, or
`git clone https://github.com/chuyao-wang/silicon-replication.git`.

## What is here

| directory | contents |
|---|---|
| `manuscript/` | the manuscript (`paper_current.md`), the Word build, and `DISCREPANCIES.md`, which traces every number to the file it comes from |
| `data/raw/` | every model response, one gzipped csv per arm: respondent id, country, item, human answer, model answer, raw model output |
| `data/summary/` | per arm, the country-mean scatter (`*_country_scatter_*`), the pooled item table (`*_rq1_*`), the per country-item individual table (`*_rq3_*`), and the run manifest recording model, seeds, weighting and library versions |
| `results/tables/` | the shipped result tables, including `manuscript_numbers.csv`, which lists each quoted number with its source |
| `results/analysis/` | the analysis outputs behind the figures and the appendix, including the two-margin table and the swapped-label scores |
| `results/figures/` | the figures as PDF and PNG, with `FIGURE_MAP.csv` mapping manuscript figure numbers to file names and producers |
| `code/figures/` | the figure scripts |
| `code/analysis/`, `scripts/` | the analysis scripts and the generation pipeline, including the submission scripts that produced each arm |
| `code/audit/` | the verification scripts (see below) |

## What is not here

The ESS microdata. The European Social Survey requires users to register
before downloading, and redistributing the file conflicts with those terms
even though access is free. The run manifests record the edition
(`10.21338/ess11e04_1`), so a user can confirm they hold the same file.
The per-response files in `data/raw/` do carry each sampled respondent's
human answer alongside the model answer, so every benchmark value the
analyses use is included; only the ESS source file itself is excluded.

## Verifying

Every number in the manuscript is traceable to a shipped file:

```bash
python3 code/audit/verify_manuscript.py --manuscript manuscript/paper_current.md
```

Expected: 523 numbers tested, 523 traced, 9 figures present, 6 tables laid
out, PASS.

The quantities added in the August 2026 revision are recomputed from the
data, rather than checked against a table, by:

```bash
python3 code/audit/cloud_verify_revise42.py --data data/summary
```

Expected: 28 of 28 quantities traced.

## The design in one paragraph

Two open-weight models (Qwen 2.5-7B-Instruct, Llama 3.1-8B-Instruct) are
prompted in first and third person to answer 42 ESS items as 685 sampled
respondents per country, conditioned on a natural-language backstory built
from 20 demographic variables. The backstory experiment observes every
block of that profile at two margins: added alone to a sparse base, and
removed alone from the complete profile. A falsification arm renders every
backstory with a wrong country under a fixed derangement and scores
recovery against both the named and the true country. Subsampling is
seeded (888); generation is unseeded except in the replicate arm, which
uses `--gen_seed 889` so that it measures run-to-run variation instead of
reproducing its twin.
