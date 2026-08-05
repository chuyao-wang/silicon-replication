#!/usr/bin/env python3
"""Patch C -- build_package.py.  Adds the files that CODE_FILES does not collect.

Two anchors, both asserted. All-or-nothing.

WHY THESE FILES
  silicon_v13.py       the v13-native metric library and the 227-check reproduction
                       suite. A third party cannot verify the package without it.
  audit3_fix.py        produces the framing contrast and the cross-level
                       counterexamples, both of which the chapter cites.
  gdp_report.py        produces the GDP-arm tables, the decomposition and the
                       bloc-of-one diagnostics, all cited in section 5.6 and 4.5.
  fig_2x2.py           the figure carrying the central mechanism claim.
  make_gdp_file.py     retrieves and validates the GDP inputs. Without it the
                       macro data in the package cannot be re-derived.
  patch_A/B.py         the record of the corrections applied to analyze_v13.py and
                       benchmark_v13.py. These are not superseded scripts, which
                       the build's own gap report warns against shipping; they are
                       the audit trail for two files that ARE shipped, and each
                       carries the pre-patch expression in a comment.
  verify_patches.py    asserts that those corrections took effect and that no
                       pre-existing column moved.
  gdp_per_capita*.csv  the two GDP vintages and their provenance records. These are
  gdp_pc_ppp*          inputs rather than code, but CODE_FILES already carries
                       item_direction_table.csv on the same logic, and SCAN_PATTERNS
                       covers only .py/.sh/.diff/.md, so a missing .csv input would
                       NOT be caught by the gap report. Shipping them here is what
                       makes `benchmark_v13.py --gdp` reproducible.
"""
p = "build_package.py"
s = open(p).read()

reps = [
    # ---- 1. the code list itself
    ('''    "item_direction_table.csv",
    "CHANGELOG_v12.md",
    "v11_to_v12.diff",          # provenance: what changed from the v11 pipeline
]
''',
     '''    # --- audit round 3: verification, corrections and extensions ---
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
'''),
    # ---- 2. widen the scan so a missing input file cannot go unreported
    ('SCAN_PATTERNS = ("*.py", "*.sh", "*.diff", "*.md")\n',
     '''# AUDIT FIX. The scan previously covered only code, so a data INPUT that
# failed to be collected would vanish without appearing in the gap report --
# which is what happened to the GDP files and to every audit-round output table.
# Adding .csv and .txt makes the report noisier and correct; the noise is the
# point, since each hit has to be decided deliberately.
SCAN_PATTERNS = ("*.py", "*.sh", "*.diff", "*.md", "*.csv", "*.txt")
'''),
]

for i, (old, new) in enumerate(reps, 1):
    n = s.count(old)
    assert n == 1, f"anchor C{i} matched {n} times, expected 1 -- ABORTED, nothing written"
    s = s.replace(old, new, 1)

open(p, "w").write(s)
print(f"Patch C applied to {p}: {len(reps)} anchors")
print("\nNow rerun:  python3 build_package.py --compress-raw")
print("Expect the gap report to list the remaining uncollected files by name.")
print("Then check:  tar tzf silicon_package_v13.tar.gz | grep -cE "
      "'silicon_v13|audit3_fix|gdp_report|fig_2x2'   -> should be 4")
