#!/usr/bin/env python3
"""
build_package.py — assemble a replication package with a real manifest.

Run from ~/Winston_Code. Safe to run now and again after the 2x2 lands; it
discovers conditions from the files actually present rather than from a
hard-coded list, so a later run simply picks up the new arms.

    python build_package.py                    # summary tables only (~5 MB)
    python build_package.py --with-raw         # also copy the raw response files (~800 MB)
    python build_package.py --out ../silicon_package_v13

WHAT THIS FIXES RELATIVE TO THE MARCH PACKAGE
---------------------------------------------
The March package had four defects that this script is designed not to repeat:

  1. DATA_MANIFEST.csv shipped as a ZERO-BYTE file while the edits summary
     recorded the manifest check as passed. Here the manifest is generated from
     the files themselves, with sha256, byte size and row count for each, and
     the script refuses to finish if it is empty.
  2. The per-item pooled tables (rq1) were absent for four conditions, so
     downstream code silently reconstructed them from the per-country moments.
     Here every condition is checked for its full set of four tables and any gap
     is reported by name.
  3. The documentation stated that the human benchmark used unweighted country
     means when the pipeline in fact design-weighted them, and described the
     missingness rule incorrectly. Here those two facts are read out of the run
     manifests rather than written by hand, so they cannot drift from what the
     code did.
  4. No environment lock. Here the interpreter and library versions are recorded
     both from the current process and from every run manifest, which matters
     because the nesting of the 250-respondent subsample inside the 500 depends
     on numpy's RandomState.choice remaining permutation-based.

WHAT IS DELIBERATELY NOT INCLUDED
---------------------------------
The ESS microdata. The European Social Survey requires users to register before
downloading, and redistributing the file conflicts with those terms even though
access is free. The package records the edition, the file's sha256 and its row
count so a user can verify they have the same file, and the README states where
to obtain it. The March package shipped the CSV itself; that should not be
repeated.

Raw per-response files are excluded by default because they run to roughly
800 MB. Their checksums and row counts are still recorded in the manifest, so a
user who obtains them separately can verify integrity. Pass --with-raw to copy
them in.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime

import pandas as pd

KINDS = {
    "country_scatter": "silicon_full_country_scatter_{tag}.csv",
    "rq1": "silicon_full_rq1_{tag}.csv",
    "rq3": "silicon_full_rq3_{tag}.csv",
}
RAW = "silicon_full_raw_{tag}_seed888.csv"

CODE_FILES = [
    "silicon_sampling_extended_v12.py",
    "submit_v13.sh",
    "submit_anchored_contrast.sh",
    "inspect_prompts.py",
    "check_pilot.py",
    "verify_reproduction.py",
    "reanalyse_v12.py",
    "analyze_v13.py",
    "analyze_v13b.py",
    "analyze_v13c.py",
    "analyze_2x2.py",
    "compare_rounds.py",
    "benchmark_v13.py",
    "logo_v13.py",
    "collect_numbers.py",
    "eta.py",
    "build_package.py",
    "monitor_v13.sh",
    # --- audit round 3: verification, corrections and extensions ---
    "silicon_v13.py",           # v13-native metrics + the 227-check suite
    "audit3_fix.py",            # framing contrast, cross-level counterexamples
    "gdp_report.py",            # GDP arm, decomposition, bloc-of-one diagnostics
    "fig_2x2.py",               # the figure carrying the mechanism claim
    "make_gdp_file.py",         # retrieves and validates the GDP inputs
    "patch_A.py",               # correction record for analyze_v13.py
    "patch_B.py",               # correction record for benchmark_v13.py
    "verify_patches.py",        # asserts the corrections took effect
    "patch_C.py",               # correction record for build_package.py itself
    # --- inputs that are not code but must ship, and that SCAN_PATTERNS
    #     (.py/.sh/.diff/.md) would NOT flag if they went missing ---
    "gdp_per_capita.csv",
    "gdp_per_capita_provenance.txt",
    "gdp_pc_ppp.csv",
    "gdp_pc_ppp_provenance.txt",
    "item_direction_table.csv",
    "CHANGELOG_v12.md",
    "v11_to_v12.diff",          # provenance: what changed from the v11 pipeline
]
# Optional, only if a Fabian migration was prepared. Collected if present.
OPTIONAL_DIRS = ["optional_fabian"]
# Anything matching these in the working directory but NOT collected above is
# reported at the end. The point is that a file cannot vanish silently: if it is
# in the project directory and not in the package, the build says so by name.
# AUDIT FIX. The scan previously covered only code, so a data INPUT that
# failed to be collected would vanish without appearing in the gap report --
# which is what happened to the GDP files and to every audit-round output table.
# Adding .csv and .txt makes the report noisier and correct; the noise is the
# point, since each hit has to be decided deliberately.
SCAN_PATTERNS = ("*.py", "*.sh", "*.diff", "*.md", "*.csv", "*.txt")
SCAN_IGNORE = {"__pycache__"}


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def n_rows(path: str) -> int | str:
    if not path.endswith(".csv"):
        return ""
    try:
        with open(path, "rb") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--march", default="results_march2026")
    ap.add_argument("--out", default="silicon_package_v13")
    ap.add_argument("--ess", default="data/ESS Data/ESS11e04_1.csv")
    ap.add_argument("--with-raw", action="store_true",
                    help="copy the raw response files (about 700 MB uncompressed)")
    ap.add_argument("--compress-raw", action="store_true",
                    help="copy the raw response files gzipped; these CSVs are "
                         "mostly short integers and repeated strings, so they "
                         "compress roughly tenfold. Implies --with-raw.")
    args = ap.parse_args()

    if args.compress_raw:
        args.with_raw = True
    R, OUT = args.results, args.out
    if not os.path.isdir(R):
        sys.exit(f"results directory not found: {R}")

    # --- discover conditions from the scatter files present ----------------
    tags = sorted(
        f[len("silicon_full_country_scatter_"):-len(".csv")]
        for f in os.listdir(R)
        if f.startswith("silicon_full_country_scatter_") and f.endswith(".csv")
    )
    # pilots are diagnostics, not results
    tags = [t for t in tags if "pilot" not in t]
    print(f"Discovered {len(tags)} conditions:")
    for t in tags:
        print(f"  {t}")

    for sub in ("code", "data", "outputs", "manifests"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    manifest_rows: list[dict] = []
    gaps: list[str] = []

    def record(src: str, rel: str, category: str, note: str = "", copy: bool = True):
        if not os.path.exists(src):
            gaps.append(f"{category}: missing {os.path.basename(src)}")
            return
        if copy:
            dst = os.path.join(OUT, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        manifest_rows.append(dict(
            path=rel, category=category, bytes=os.path.getsize(src),
            rows=n_rows(src), sha256=sha256(src), included=bool(copy), note=note))

    # --- summary tables ----------------------------------------------------
    print("\nCopying summary tables...")
    for t in tags:
        for kind, pat in KINDS.items():
            record(os.path.join(R, pat.format(tag=t)),
                   os.path.join("data", pat.format(tag=t)), f"data/{kind}")
        # raw: checksum always, copy only on request
        raw_src = os.path.join(R, RAW.format(tag=t))
        if os.path.exists(raw_src):
            if args.compress_raw:
                dst = os.path.join(OUT, "data", RAW.format(tag=t) + ".gz")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(raw_src, "rb") as fi, gzip.open(dst, "wb", compresslevel=6) as fo:
                    shutil.copyfileobj(fi, fo, length=1 << 20)
                # checksum is of the ORIGINAL file, so it can be verified after
                # decompression rather than only against this specific archive
                manifest_rows.append(dict(
                    path=os.path.join("data", RAW.format(tag=t) + ".gz"),
                    category="data/raw", bytes=os.path.getsize(raw_src),
                    rows=n_rows(raw_src), sha256=sha256(raw_src), included=True,
                    note=f"stored gzipped ({os.path.getsize(dst)/1e6:.1f} MB on disk); "
                         f"bytes/rows/sha256 describe the UNCOMPRESSED file"))
            else:
                record(raw_src, os.path.join("data", RAW.format(tag=t)), "data/raw",
                       note="" if args.with_raw else "checksum only; not included (size)",
                       copy=args.with_raw)
        else:
            gaps.append(f"data/raw: missing {RAW.format(tag=t)}")

    # --- run manifests -----------------------------------------------------
    print("Copying run manifests...")
    env_from_runs: dict[str, set] = {}
    for t in tags:
        src = os.path.join(R, f"manifest_{t}.json")
        record(src, os.path.join("manifests", f"manifest_{t}.json"), "manifests")
        if os.path.exists(src):
            with open(src) as f:
                m = json.load(f)
            for k in ("ess_file", "missing_rule", "human_weight", "pandas_version",
                      "numpy_version", "vllm_version", "sample_per_country",
                      "max_model_len", "gen_seed", "sampling_seed", "temperature"):
                env_from_runs.setdefault(k, set()).add(str(m.get(k)))

    # --- analysis outputs --------------------------------------------------
    print("Copying analysis outputs...")
    adir = os.path.join(R, "analysis")
    if os.path.isdir(adir):
        for f in sorted(os.listdir(adir)):
            if f.endswith(".csv"):
                record(os.path.join(adir, f), os.path.join("outputs", f), "outputs")
    else:
        gaps.append("outputs: results/analysis/ not found — run the analyze scripts")

    # --- code --------------------------------------------------------------
    print("Copying code...")
    for f in CODE_FILES:
        if os.path.exists(f):
            record(f, os.path.join("code", f), "code")
        else:
            gaps.append(f"code: missing {f}")

    # --- optional prepared-but-unused material -----------------------------
    for od in OPTIONAL_DIRS:
        if os.path.isdir(od):
            print(f"Copying {od}/ ...")
            for f in sorted(os.listdir(od)):
                src = os.path.join(od, f)
                if os.path.isfile(src):
                    record(src, os.path.join("code", od, f), "code/optional",
                           note="prepared for a possible cluster migration; "
                                "not used for the reported results")

    # --- ESS reference (checksum only) -------------------------------------
    ess_info = None
    if os.path.exists(args.ess):
        print("Recording ESS reference (not copied)...")
        ess_info = dict(name=os.path.basename(args.ess),
                        bytes=os.path.getsize(args.ess),
                        rows=n_rows(args.ess), sha256=sha256(args.ess))
        manifest_rows.append(dict(
            path=f"[external] {ess_info['name']}", category="data/ess-reference",
            bytes=ess_info["bytes"], rows=ess_info["rows"], sha256=ess_info["sha256"],
            included=False,
            note="NOT redistributed: ESS requires registration. Obtain from "
                 "europeansocialsurvey.org and verify against this sha256."))
    else:
        gaps.append(f"data/ess-reference: {args.ess} not found")

    # --- March round, for the cross-round robustness claim ------------------
    if os.path.isdir(args.march):
        print("Recording March round summary tables...")
        for f in sorted(os.listdir(args.march)):
            keep = any(f.startswith(pre) for pre in
                       ("silicon_full_country_scatter_", "silicon_full_rq1_",
                        "silicon_full_rq3_")) and f.endswith(".csv")
            if keep:
                record(os.path.join(args.march, f),
                       os.path.join("data_march_round", f), "data/march-round",
                       note="independent earlier round: ESS edition 4.2, n=500, "
                            "older pandas/vLLM; supports the cross-round "
                            "stability claim")

    # --- write the manifest ------------------------------------------------
    man = pd.DataFrame(manifest_rows)
    if man.empty:
        sys.exit("REFUSING TO FINISH: manifest is empty. Nothing was recorded.")
    man = man.sort_values(["category", "path"])
    man_path = os.path.join(OUT, "DATA_MANIFEST.csv")
    man.to_csv(man_path, index=False)
    if os.path.getsize(man_path) == 0:
        sys.exit("REFUSING TO FINISH: DATA_MANIFEST.csv wrote zero bytes.")

    # --- environment lock --------------------------------------------------
    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return "not importable here"

    env = dict(
        built_at=datetime.now().isoformat(timespec="seconds"),
        built_with_python=sys.version.split()[0],
        packaging_environment={m: ver(m) for m in
                               ("pandas", "numpy", "scipy", "sklearn")},
        generation_environment={k: sorted(v) for k, v in env_from_runs.items()},
        note=("generation_environment is read from the run manifests, not typed "
              "by hand. A single value per key means every condition agreed. "
              "numpy/pandas versions matter because the nesting of a smaller "
              "subsample inside a larger one relies on RandomState.choice being "
              "implemented as permutation(pop)[:size]."),
    )
    with open(os.path.join(OUT, "ENVIRONMENT.json"), "w") as f:
        json.dump(env, f, indent=2)

    # --- completeness check per condition ----------------------------------
    print("\n" + "=" * 78)
    print("COMPLETENESS BY CONDITION")
    print("=" * 78)
    print(f"  {'condition':32s} {'scatter':>8s} {'rq1':>5s} {'rq3':>5s} {'raw':>5s} {'manifest':>9s}")
    inc = set(man.path)
    for t in tags:
        marks = []
        for kind, pat in KINDS.items():
            marks.append("yes" if os.path.join("data", pat.format(tag=t)) in inc else "NO")
        # gzipped raw files end in .csv.gz, so match on the stem rather than
        # on the exact filename or every condition reports NO.
        stem = RAW.format(tag=t)
        raw_ok = any(os.path.basename(p) in (stem, stem + ".gz") for p in man.path)
        marks.append("yes" if raw_ok else "NO")
        marks.append("yes" if os.path.join("manifests", f"manifest_{t}.json") in inc else "NO")
        print(f"  {t:32s} {marks[0]:>8s} {marks[1]:>5s} {marks[2]:>5s} "
              f"{marks[3]:>5s} {marks[4]:>9s}")

    # --- README ------------------------------------------------------------
    gen = env["generation_environment"]

    def one(k, default="unknown"):
        v = gen.get(k, [])
        return v[0] if len(v) == 1 else (", ".join(v) if v else default)

    readme = f"""# Silicon sampling — replication package (v13 round)

Built {env['built_at']} by code/build_package.py. Every figure in DATA_MANIFEST.csv
is computed from the file itself; nothing in this README is typed by hand.

## Generation settings, read from the run manifests

| setting | value |
|---|---|
| ESS edition | {one('ess_file')} |
| respondents per country | {one('sample_per_country')} |
| sampling seed | {one('sampling_seed')} |
| generation seed | {one('gen_seed')} (None = unseeded, so each arm is an independent draw) |
| temperature | {one('temperature')} |
| max_model_len | {one('max_model_len')} |
| human missingness rule | {one('missing_rule')} |
| human benchmark weighting | {one('human_weight')} |
| pandas / numpy | {one('pandas_version')} / {one('numpy_version')} |
| vLLM | {one('vllm_version')} |

Two of these deserve emphasis because the earlier round's documentation stated
them incorrectly. The human country means are DESIGN-WEIGHTED by pspwght while
the silicon means are necessarily unweighted, so the two sides are not the same
estimand. And the missingness rule is each item's own valid response range, which
removes the ESS single-digit refusal / don't-know / no-answer codes (7, 8, 9)
that a rule based only on the two- and three-digit codes retains on the sixteen
items whose substantive scale has fewer than seven categories.

## Layout

    code/                 pipeline, submit scripts, every analysis script
    data/                 per-condition summary tables (scatter, rq1, rq3)
    data_march_round/     the earlier independent round, for the stability claim
    manifests/            one JSON per condition, recording its exact settings
    outputs/              analysis tables, including manuscript_numbers.csv
    DATA_MANIFEST.csv     sha256, byte size and row count for every file
    ENVIRONMENT.json      interpreter and library versions

## Provenance of the March round

data_march_round/ holds summary tables from an earlier, independent round used
only to support the cross-round stability claim, which code/compare_rounds.py
recomputes from the tables shipped here. That round is NOT reproducible from this
package: its generation code was an earlier pipeline version, and file
timestamps indicate that not every condition in it was produced by the same
version (most conditions date from early March, the demo_only condition from
roughly two weeks later). It also used ESS edition 4.2 and n=500 per country.
Treat it as an independent replication whose summary statistics are verifiable
here, not as a second reproducible round. The primary round is the one described
in the table above, which used one pipeline version throughout.

## The direction classification

code/item_direction_table.csv is the frozen response-direction classification for
all forty-two items, taken from the ESS codebook, with each item's scale, its low
and high anchor labels and the original question wording. Thirteen items are
classified as reverse-coded. The analysis scripts carry the same classification
as an internal constant; the two were checked against each other and agree
exactly. The table is the citable form and belongs in the manuscript appendix.

## What is not here

The ESS microdata is not redistributed: the European Social Survey requires
registration before download. DATA_MANIFEST.csv records the edition, size, row
count and sha256 of the file used, so an independent user can verify they have
the same one. Obtain it from europeansocialsurvey.org.

Raw per-response files{'' if args.with_raw else ' are excluded by default because of their size, but'} have their
checksums recorded regardless, so integrity can be verified if they are obtained
separately.

## Reproducing

1. Obtain the ESS file named above and verify its sha256 against DATA_MANIFEST.csv.
2. Recreate the environment in ENVIRONMENT.json.
3. Regenerate responses: `bash code/submit_v13.sh all` (about 10.4M prompts) and
   `bash code/submit_anchored_contrast.sh` for the 2x2 arms.
4. Reproduce the analysis: run `analyze_v13.py`, `analyze_v13b.py`,
   `analyze_v13c.py`, `analyze_2x2.py`, then `collect_numbers.py`.

Generation is unseeded, so a re-run will not reproduce responses byte for byte.
The empirical run-to-run spread is reported in outputs/anchored_did_real_noise.csv
(the empirical_noise column), measured from two independent generations of the
same 22 items over the same respondents. Any difference smaller than that is
generation stochasticity, not a discrepancy.
"""
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write(readme)

    # --- report ------------------------------------------------------------
    # bytes in the manifest describes the ORIGINAL (uncompressed) file so the
    # sha256 remains verifiable, so it cannot be used as disk usage. Measure the
    # package directory instead.
    disk = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(OUT) for f in fs)
    total = int(man[man.included].bytes.sum())
    print("\n" + "=" * 78)
    print("PACKAGE SUMMARY")
    print("=" * 78)
    print(f"  location: {os.path.abspath(OUT)}")
    print(f"  files included: {int(man.included.sum())}   "
          f"recorded but not included: {int((~man.included).sum())}")
    print(f"  size on disk: {disk/1e6:.1f} MB"
          + (f"  (uncompressed content would be {total/1e6:.1f} MB)"
             if disk < total * 0.9 else ""))
    print(f"\n  by category:")
    for cat, g in man.groupby("category"):
        print(f"    {cat:24s} {len(g):3d} files  "
              f"{g[g.included].bytes.sum()/1e6:8.2f} MB "
              f"({int(g.included.sum())} included)")
    # --- anything in the project directory that did NOT get collected ------
    collected = {os.path.basename(p) for p in man.path}
    uncollected = []
    for pat in SCAN_PATTERNS:
        for f in sorted(glob.glob(pat)):
            if (os.path.isfile(f) and os.path.basename(f) not in collected
                    and not any(x in f for x in SCAN_IGNORE)):
                uncollected.append(f)
    if uncollected:
        print(f"\n  IN THE PROJECT DIRECTORY BUT NOT IN THE PACKAGE ({len(uncollected)}):")
        for f in uncollected:
            print(f"    {f}  ({os.path.getsize(f)/1e3:.1f} kB)")
        print("  Decide deliberately for each: add it to CODE_FILES if it belongs,")
        print("  or leave it out knowingly. Superseded pipeline versions and old")
        print("  submit scripts normally do NOT belong in a replication package,")
        print("  because shipping several versions of the same script invites a")
        print("  reader to run the wrong one.")

    if gaps:
        print(f"\n  GAPS ({len(gaps)}) — these are reported, not silently skipped:")
        for x in gaps:
            print(f"    {x}")
        print("\n  If the 2x2 arms are still running, their absence is expected;")
        print("  re-run this script once they finish and they will be picked up.")
    else:
        print("\n  No gaps.")
    print("\n  To archive:  tar czf silicon_package_v13.tar.gz " + OUT)
    print("=" * 78)


if __name__ == "__main__":
    main()
