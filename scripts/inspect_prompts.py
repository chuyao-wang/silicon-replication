#!/usr/bin/env python3
"""
inspect_prompts.py — CPU-only audit of what will actually be sent to the model.

WHY RUN THIS BEFORE ANY GPU JOB
-------------------------------
Three of the four pilot gates are properties of prompt construction, not of the
model, and therefore need no GPU, no queue slot and no model load:

  * that full_noregion and full_nocountry differ by exactly one sentence, which
    is the entire identification claim of the primary contrast;
  * that the anchored scale line renders the ESS showcard labels correctly for
    every item in the anchored arm, and that the treatment and control items
    receive comparable label dosage;
  * that prompt token length stays well inside max_model_len, since silent
    truncation would corrupt a run without any error.

Only the parse-rate gate needs the model. Running this first means a GPU slot is
never spent discovering a text-formatting mistake.

Usage:
  python inspect_prompts.py --ess <path to ESS11e04_2.csv>
  python inspect_prompts.py --ess ... --countries AT DE FI --show 2
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys

PIPE = "silicon_sampling_extended_v12.py"

REVERSE = "polintr imsmetn impcntr imdfetn gincdif freehms hmsfmlsh hmsacld " \
          "health rlgatnd aesfdrk hincfel".split()
FWD_CTRL = "actrolga cptppola inprdsc psppipla psppsgva sclmeet".split()
FWD_PLACEBO = "trstplc stflife happy stfdem".split()
ANCHOR_SET = REVERSE + FWD_CTRL + FWD_PLACEBO


def anchored_scale(m, var: str) -> str:
    """Reproduce exactly what main() does under --scale_labels anchored."""
    base = m.ESS_VARIABLES[var]["scale"]
    labs = m.ANCHOR_LABELS.get(var, {})
    if not labs:
        return base
    return f"{base} (" + "; ".join(f"{v} = {t}" for v, t in sorted(labs.items())) + ")"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ess", required=True)
    ap.add_argument("--countries", nargs="*", default=["AT", "DE", "FI"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--show", type=int, default=1, help="full prompts to print")
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args()

    logging.disable(logging.INFO)
    if not os.path.exists(PIPE):
        sys.exit(f"{PIPE} not found in {os.getcwd()}")
    spec = importlib.util.spec_from_file_location("v12", PIPE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)          # does NOT import vllm

    print("=" * 78)
    print("1. BACKSTORY VARIANTS — the identification claim")
    print("=" * 78)
    df = m.load_data(ess_file=args.ess, countries=args.countries,
                     sample_per_country=args.n, seed=888,
                     backstory_mode="v5_clean", missing_rule="range")
    row = df.iloc[0]
    modes = ["demo_only", "minimal", "ses", "political", "v5_clean",
             "full_noregion", "full_nocountry", "full_nopolitical"]
    bs = {mo: m.generate_backstory(row, mode=mo) for mo in modes}
    for mo in modes:
        n_sent = len([s for s in bs[mo].split(". ") if s.strip()])
        print(f"  {mo:17s} {n_sent:2d} sentences, {len(bs[mo]):4d} chars")

    a = [s.strip() for s in bs["full_noregion"].split(". ") if s.strip()]
    b = [s.strip() for s in bs["full_nocountry"].split(". ") if s.strip()]
    only_a, only_b = [s for s in a if s not in b], [s for s in b if s not in a]
    print(f"\n  full_noregion minus full_nocountry : {only_a}")
    print(f"  full_nocountry minus full_noregion : {only_b}")
    verdict = (len(only_a) == 1 and not only_b and only_a[0].lower().startswith("i live in"))
    print(f"  --> PRIMARY CONTRAST CLEAN: {verdict}"
          + ("" if verdict else "   <-- STOP. Do not submit noregion_pair."))

    reg = [s for s in bs["v5_clean"].split(". ") if "region" in s]
    print(f"\n  region sentence in the main condition: {reg}")
    print("  (a raw NUTS code; its first two characters are the country ISO code,")
    print("   which is why both no-* arms must drop it — see revision plan section 6)")

    print("\n" + "=" * 78)
    print("2. ANCHORED SCALE LINES — dosage of the treatment")
    print("=" * 78)
    print(f"  {'item':10s} {'grp':9s} {'numeric':8s} anchored")
    for var in ANCHOR_SET:
        grp = ("reverse" if var in REVERSE else
               "ctrl" if var in FWD_CTRL else "placebo")
        base = m.ESS_VARIABLES[var]["scale"]
        anc = anchored_scale(m, var)
        print(f"  {var:10s} {grp:9s} {base:8s} {anc}")
    n_lab = {g: [] for g in ("reverse", "ctrl", "placebo")}
    for var in ANCHOR_SET:
        g = ("reverse" if var in REVERSE else "ctrl" if var in FWD_CTRL else "placebo")
        n_lab[g].append(len(m.ANCHOR_LABELS.get(var, {})))
    for g, v in n_lab.items():
        print(f"  mean labelled categories, {g:8s}: {sum(v)/len(v):.2f}  (n={len(v)})")
    print("  reverse and ctrl should be comparable (dosage-matched); placebo is")
    print("  endpoint-only by construction and is a placebo, not a matched control.")

    print("\n" + "=" * 78)
    print("3. PROMPT LENGTH vs max_model_len")
    print("=" * 78)
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(m.MODELS["qwen"])
        print("  tokenizer: Qwen2.5-7B-Instruct")
    except Exception as e:
        print(f"  tokenizer unavailable ({type(e).__name__}); using chars/3.5 as a"
              f" rough upper bound")

    def toklen(s: str) -> int:
        return len(tok(s)["input_ids"]) if tok else int(len(s) / 3.5)

    tmpl = {"1p": m.PROMPT_1P_STANDARD, "3p": m.PROMPT_3P}
    for cond, scales in (("numeric", False), ("anchored", True)):
        lens = []
        for pm, t in tmpl.items():
            for _, r in df.iterrows():
                for var in ANCHOR_SET:
                    info = m.ESS_VARIABLES[var]
                    lens.append(toklen(t.format(
                        backstory=r["backstory"],
                        question=info.get("question", info["label"]),
                        scale=anchored_scale(m, var) if scales else info["scale"])))
        lens.sort()
        p99 = lens[int(0.99 * (len(lens) - 1))]
        flag = "" if max(lens) < 0.8 * args.max_model_len else "   <-- RAISE max_model_len"
        print(f"  {cond:9s} median {lens[len(lens)//2]:5d}  p99 {p99:5d}  "
              f"max {max(lens):5d}  budget {args.max_model_len}{flag}")

    print("\n" + "=" * 78)
    print("4. VERBATIM PROMPTS — check these against the appendix example")
    print("=" * 78)
    for var in ("freehms", "trstplc"):
        for cond in ("numeric", "anchored"):
            info = m.ESS_VARIABLES[var]
            p = m.PROMPT_1P_STANDARD.format(
                backstory=df.iloc[0]["backstory"],
                question=info.get("question", info["label"]),
                scale=anchored_scale(m, var) if cond == "anchored" else info["scale"])
            print(f"\n----- {var} / 1p / {cond} " + "-" * 40)
            print(p)
    print("\n  NOTE: for 39 of the 42 items the question line is the ESS codebook")
    print("  LABEL, not the survey question text. For the agree-disagree items this")
    print("  removes the agree-disagree framing entirely, which is why part of the")
    print("  direction failure is instrumentation. The manuscript currently says")
    print("  'the ESS question text'; see revision plan section 9.")


if __name__ == "__main__":
    main()
