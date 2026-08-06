# Replicating the 2 x 2 in Llama — what to run, and where

Two new generation arms. Everything else already exists.

Set one variable first, on the Mac, so nothing below hard-codes the cluster.
Type the alias itself, with no angle brackets: `<` and `>` are redirection in
the shell and the line will not parse.

```bash
export CLUSTER=my_cluster_alias           # the name you already type after ssh
ssh "$CLUSTER" true && echo "alias works"
```

---

## 0. What is missing, and what is not

The Llama 2 x 2 is half done already. The numeric cells are the 42-item runs
and the analysis subsets them to the same 22 items:

|                  | numeric scales             | anchored scales                       |
|------------------|----------------------------|---------------------------------------|
| with country     | `llama_1p_full_noregion` ✅ | `llama_1p_full_noregion_anchored` ❌   |
| without country  | `llama_1p_full_nocountry` ✅| `llama_1p_full_nocountry_anchored` ❌  |

So: **two jobs, 904,200 prompts, about 2.5 hours of wall clock** with the two
jobs side by side on one GPU each. That is the same prompt count as the Qwen
contrast and about 2.7% of everything generated for this chapter so far.

Every setting matches the Qwen 2 x 2 and the existing Llama arms: n = 685 per
country, sampling seed 888 (the same respondents), temperature 0.7,
`max_model_len` 1024, `missing_rule range`, `human_weight pspwght`, and the
same 22 items. The only differences from `submit_anchored_contrast.sh` are
`--model llama` and the job names.

---

## 1. Mac: send the submission script up

Two files go up: the submission script, and the updated `analyze_2x2.py`,
which is the one that takes `--model`. The copy already on the cluster does
not.

```bash
cd ~/silicon_pkg/silicon_chapter          # or wherever this package lives
scp scripts/submit_anchored_contrast_llama.sh "$CLUSTER":~/Winston_Code/
scp scripts/analyze_2x2.py                  "$CLUSTER":~/Winston_Code/
```

If this package is not the checkout that has them, take them from wherever
you saved them, for example `~/Downloads`:

```bash
scp ~/Downloads/submit_anchored_contrast_llama.sh "$CLUSTER":~/Winston_Code/
scp ~/Downloads/analyze_2x2.py                    "$CLUSTER":~/Winston_Code/
```

## 2. Cluster: submit

```bash
ssh "$CLUSTER"
cd ~/Winston_Code
bash submit_anchored_contrast_llama.sh
```

It prints two job IDs. If `sbatch` rejects the partition, set it and rerun:

```bash
PARTITION=<partition> bash submit_anchored_contrast_llama.sh
```

## 3. Watch — unattended, from the Mac

Steps 3, 4 and 5 are one command. Run it on the Mac, in the package
directory, and leave it:

```bash
nohup caffeinate -i bash scripts/watch_llama_anchored.sh \
      > ~/llama_anchored_watch.out 2>&1 &
```

It checks `squeue` every ten minutes over a fresh ssh each time, so a dropped
VPN costs one check and not the run. When both jobs leave the queue it
verifies the four output files, runs `analyze_2x2.py --model llama`, copies
everything down to `data/summary/`, and raises a macOS notification. If the
jobs die without output it copies the tail of the error logs down instead.
Everything it prints is also in `~/llama_anchored_watch.log`.

`caffeinate -i` stops the Mac sleeping while it waits. Closing the lid still
sleeps the machine; the watcher picks up where it left off on wake, because
the cluster jobs do not depend on it. `INTERVAL=3600` makes it hourly.
Stopping it (`pkill -f watch_llama_anchored`) does not touch the jobs.

By hand instead:

```bash
squeue --me
tail -f logs/anch_llama_noregion_*.out
```

Done when these four files exist:

```bash
ls -la results/silicon_full_country_scatter_llama_1p_full_noregion_anchored.csv \
       results/silicon_full_country_scatter_llama_1p_full_nocountry_anchored.csv \
       results/manifest_llama_1p_full_noregion_anchored.json \
       results/manifest_llama_1p_full_nocountry_anchored.json
```

Sanity check before trusting them, on the cluster:

```bash
python - <<'PY'
import json, pandas as pd
for tag in ("llama_1p_full_noregion_anchored", "llama_1p_full_nocountry_anchored"):
    m = json.load(open(f"results/manifest_{tag}.json"))
    d = pd.read_csv(f"results/silicon_full_country_scatter_{tag}.csv")
    print(tag)
    print("   items", m["variables_n"], "| n/country", m["sample_per_country"],
          "| seed", m["sampling_seed"], "| scale", m["scale_labels"])
    print("   rows", len(d), "| countries", d.cntry.nunique(),
          "| variables", d.variable.nunique())
PY
```

Expect `items 22`, `n/country 685`, `seed 888`, `scale anchored`, and
22 x 30 = 660 rows per file.

## 4. Cluster: the number that matters

```bash
cd ~/Winston_Code
python analyze_2x2.py --model llama
```

It prints the triple difference: the change in the country-label effect under
anchoring, contrasted between reverse and forward items, with a bootstrap
interval. That single number is the replication.

## 5. Mac: pull the results down

```bash
cd ~/silicon_pkg/silicon_chapter
for t in llama_1p_full_noregion_anchored llama_1p_full_nocountry_anchored; do
  scp "$CLUSTER":"~/Winston_Code/results/silicon_full_country_scatter_$t.csv" data/summary/
  scp "$CLUSTER":"~/Winston_Code/results/manifest_$t.json"                     data/summary/
done
scp "$CLUSTER":"~/Winston_Code/results/analysis/twoxtwo_*llama*" results/tables/ 2>/dev/null || true
```

## 6. Mac: optional figure, and the rebuild

The replication does not need its own figure; a sentence in 4.2 carries it.
If you want one for the appendix:

```bash
python3 code/figures/fig_2x2.py --model llama \
        --data data/summary --figdir results/figures \
        --name fig_2x2_country_by_scale_llama
```

Then rebuild and verify as usual:

```bash
PYTHON=$HOME/miniconda3/bin/python3 bash code/rebuild.sh 2>&1 | tail -25
python3 code/audit/verify_manuscript.py --manuscript ~/Downloads/paper_current.md 2>&1 | tail -6
```

## 7. What to write, under each outcome

The prediction is already in the manuscript, in 2.2, written before this run:
supplying verbal endpoints should remove the inversion without touching the
model's country knowledge. So this is an out-of-sample test, not a search.

| Outcome | 4.2 | 5.4 |
|---|---|---|
| Triple difference positive, interval excludes zero | add one sentence: the same contrast in Llama gives +X [lo, hi] | limitation four is deleted, and its next step with it |
| Same sign, interval includes zero | "the same contrast points the same way in Llama, at +X [lo, hi]" | limitation four becomes "the effect size differs between models" |
| Opposite sign, or flat | "the contrast does not replicate in Llama" | 5 and 5.2 say the instrument effect is model-specific, which is a finding in itself |

Only the third outcome touches the discussion, and it is better found now than
at the viva.

---

## Troubleshooting, from the last cluster session

**`verify_manuscript.py` says `No such file or directory: /tmp/paper_current.md`.**
The manuscript lives on the Mac, not on the cluster. Either run the verifier on
the Mac against the local file, or copy the manuscript up first:

```bash
scp ~/Downloads/paper_current.md "$CLUSTER":/tmp/paper_current.md
```

**`NO PRODUCER: fig4_backstory_ladder.*` and `figA4_cumulative_ladder.*`.**
Two different causes.
`fig4_backstory_ladder.*` is the current Figure 4, drawn by
`code/figures/fig_two_margin.py`; the cluster copy of `code/figures/` is behind
the Mac copy, which is why the audit cannot attribute it. Sync the directory:

```bash
rsync -av code/figures/ "$CLUSTER":~/silicon_chapter/code/figures/
```

`figA4_cumulative_ladder.*` is the retired cumulative ladder, deleted from the
manuscript. Delete the leftovers on the cluster:

```bash
ssh "$CLUSTER" 'rm -f ~/silicon_chapter/results/figures/figA4_cumulative_ladder.*'
```

**`cd: ~/silicon_chapter/.stage: No such file or directory`.**
`.stage` is created by `code/rebuild.sh`. Run the rebuild on the cluster before
`verify_patches.py`, or run both on the Mac, where the last full pass already
reported 523 tested, 523 traced, 9 figures, 6 tables, PASS.
