#!/usr/bin/env python3
"""
Silicon Sampling Framework - VERSION 12 (Jul 2026)
==================================================
Single authoritative pipeline. Supersedes v4-v11. Runs every condition
used in the manuscript and the arms required by the July 2026 supervisor
review.

v12 CHANGES (all analysis-side; prompts and sampling are untouched)
------------------------------------------------------------------
  P0-A  --missing_rule {range,legacy}, default 'range'. Pre-v12 runs kept
        the ESS single-digit refusal / don't-know / no-answer codes (7/8/9)
        as valid HUMAN data on the sixteen items whose substantive scale has
        fewer than seven categories, because ESS_MISSING_CODES held only the
        two- and three-digit variants. The silicon side was never affected
        (parse_response validates against the same range), so the resulting
        bias was one-sided and concentrated on the coarse-scale, mostly
        reverse-coded items. See ITEM_VALID_RANGE.
  P0-B  --human_weight {pspwght,none}, default 'pspwght'. This preserves the
        behaviour of every released run, but the choice was undocumented and
        is described the wrong way round in the replication package. It is
        now explicit and written to the run manifest.
  P0-C  The vLLM engine seed is no longer passed as an explicit None; some
        releases default LLM.seed to 0, so passing None is not equivalent to
        omitting the argument.
  Also: run_full_variables writes silicon_full_rq3_<tag>.csv directly, so
        per-country individual-level recovery no longer depends on a separate
        post-processing script; token-length diagnostics; corrected variable
        counts in --help and BACKSTORY_DESC (ses 13 not 14, political 14 not
        16); pandas / numpy versions recorded in the manifest because nested
        subsampling depends on RandomState.choice remaining permutation-based.

REPRODUCING PRE-v12 OUTPUT EXACTLY
----------------------------------
  --missing_rule legacy --human_weight pspwght --max_model_len 1024
  (and do not set --gen_seed). Any other combination changes the human side
  of the comparison and must not be pooled with pre-v12 files.

CONDITION DESIGN (v10; order-free identification)
-------------------------------------------------
  demo_only        3 vars, NO country           } sparse pair:
  minimal          4 vars = demo_only + country } country effect vs SPARSE baseline
  ses              13 vars (cumulative)         } DESCRIPTIVE ONLY: the cumulative
  political        14 vars (= ses + lrscale)    } curve is order-dependent (Type-I-
  full_clean       20 vars, WITH country        } style path dependence); its non-
                                                  monotonicity is reported as
                                                  evidence, not as identification.
  full_noregion    full_clean - NUTS region     } PRIMARY country contrast:
  full_nocountry   - NUTS region - country      } both arms region-free, so the
                                                  ONLY difference is the country
                                                  label. (A full_clean vs
                                                  full_nocountry comparison would
                                                  bundle country + region and
                                                  UPPER-bound the label effect.)
  minimal_politics 6 vars = minimal + lrscale   } v12.2 (2 Aug 2026): the
                   + clsprty                    } regrouped ladder's political
                                                } rung, the COMPLETE political-
                                                } identity block over the
                                                } sparse country baseline
  full_nopolitical full_clean - lrscale/clsprty } optional LOGO of the quasi-
                                                  attitudinal group (the observed
                                                  interference point on the curve).

  Order-free logic: primary inference comes from pairwise
  leave-one-group-out contrasts against the rich baseline, never from
  the position of a group in a cumulative sequence. Theory-grouping of
  the 20 profile variables (for any further LOGO work):
  G1 contextual (country, region, domicil) | G2 ascriptive (gndr, agea,
  yrbrn) | G3 socioeconomic (eisced, eduyrs, mainact, hinctnta) |
  G4 household (maritalb, hhmmb, chldhhe) | G5 civic status (brncntr,
  ctzcntr) | G6 political identity (lrscale, clsprty; quasi-attitudinal,
  conceptually adjacent to the outcome battery) | G7 social markers
  (mbtru, dscrgrp, netusoft). Country is the only group-level cue, so
  the decisive test for RQ2 is the single G1-label contrast above;
  bonus: full_noregion(n) vs legacy full_clean(500) estimates the
  region-code effect.

SCALE ANCHORING (mechanism-two experiment; # v9 CHANGE)
-------------------------------------------------------
  --scale_labels numeric   (default) bare scale range, e.g.
                           "Scale: 1-4" - EXACT reproduction of all
                           existing runs.
  --scale_labels anchored  scale line reproduces the ESS showcard
                           labels from the codebook, e.g.
                           "Scale: 1-4 (1 = Very interested; ...;
                           4 = Not at all interested)".
                           Question wording is deliberately NOT
                           changed: ESS question text contains
                           "[country]" placeholders, which would both
                           confound the anchor manipulation and leak
                           the country cue.
  Output files gain an "_anchored" tag. If the reverse-coded items
  flip from negative to positive r_bc under anchoring, the
  scale-direction-heuristic mechanism is demonstrated experimentally.

OTHER v9 CHANGES
----------------
  --variables v1 v2 ...    optional subset of the 42 items (e.g. the
                           13 reverse-coded items for a cheap anchored
                           run); default = all 42.
  full_nocountry mode      from v8: removes the country sentence AND
                           the NUTS region code (its two-letter prefix
                           identifies the country); verified on real
                           ESS rows across all 30 countries that these
                           are the only two differences vs full_clean.
  ANCHOR_LABELS            all 42 items' verbal labels extracted from
                           the ESS11 codebook (authoritative source
                           for the reverse-coding classification; see
                           item_direction_table.csv).

Sampling: identical respondents for a given seed and
sample_per_country; sample_per_country=250 is nested within 500
(verified 30/30 countries). Note that DEFAULT flags no longer reproduce
pre-v12 output: max_model_len is 2048 (was 1024) and the human-side
missingness rule has changed. See "REPRODUCING PRE-v12 OUTPUT" above.
"""

import pandas as pd
import numpy as np
import time
import random
import re
import os
import json
import argparse
import logging
from datetime import datetime
from scipy import stats
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configurations
MODELS = {
    'qwen': 'Qwen/Qwen2.5-7B-Instruct',
    'llama': 'meta-llama/Llama-3.1-8B-Instruct'
}

# Data paths
DATA_DIR = os.path.expanduser("~/Winston_Code/data")
ESS11_FILE = os.path.join(DATA_DIR, "ESS11e04_2.csv")  # v10 CHANGE (B2): idno-verified: existing 500pp runs match e04_2 row order exactly (30/30 countries, seed 888)
ESS_ROUNDS_DIR = os.path.join(DATA_DIR, "ESS Data")

# ESS Round to fieldwork year mapping
ESS_ROUND_YEAR = {
    1: 2002, 2: 2004, 3: 2006, 4: 2008, 5: 2010,
    6: 2012, 7: 2014, 8: 2016, 9: 2018, 10: 2020, 11: 2023
}

# Output directories
RESULTS_DIR = "results"
LOGS_DIR = "logs"
BACKUP_DIR = os.path.join(RESULTS_DIR, "backups")


def auto_backup_results(tag):
    """
    Auto-backup existing results before a new run to prevent overwrites.
    Creates timestamped backup of any files matching the current tag.

    v6 feature: prevents the 1P/3P overwrite issue from v5.
    """
    if not os.path.exists(RESULTS_DIR):
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_subdir = os.path.join(BACKUP_DIR, f"{tag}_{timestamp}")
    backed_up = []

    for f in os.listdir(RESULTS_DIR):
        if not f.endswith('.csv'):
            continue
        # Backup files that would be overwritten by this run
        # Match files containing the tag OR the model name (for legacy compat)
        model_name = tag.split('_')[0]  # e.g., 'qwen' from 'qwen_1p'
        if tag in f or (model_name in f and '_1p' not in f and '_3p' not in f):
            src = os.path.join(RESULTS_DIR, f)
            if not os.path.exists(backup_subdir):
                os.makedirs(backup_subdir, exist_ok=True)
            dst = os.path.join(backup_subdir, f)
            import shutil
            shutil.copy2(src, dst)
            backed_up.append(f)

    if backed_up:
        logging.info(f"Auto-backed up {len(backed_up)} files to {backup_subdir}/")
        for f in backed_up:
            logging.info(f"  {f}")
    else:
        logging.info(f"No existing files to backup for tag '{tag}'")

    return backup_subdir if backed_up else None

# Country code to full name mapping (ESS11 countries)
# v12.6 (2 Aug 2026): the swapped-label falsification arm. One fixed
# derangement of the 30 ESS11 countries: numpy default_rng(888), third
# permutation drawn is the first with no fixed point. Every respondent in
# full_swapcountry mode is described as living in the mapped country; the
# results files keep the TRUE cntry, so recovery can be scored against
# both the labeled and the true country.
SWAP_MAP = {
    'AT': 'ME', 'BE': 'SE', 'BG': 'DE', 'CH': 'SI', 'CY': 'FI',
    'DE': 'NL', 'EE': 'CH', 'ES': 'EE', 'FI': 'LT', 'FR': 'IS',
    'GB': 'ES', 'GR': 'RS', 'HR': 'PT', 'HU': 'PL', 'IE': 'CY',
    'IL': 'HU', 'IS': 'UA', 'IT': 'SK', 'LT': 'NO', 'LV': 'BG',
    'ME': 'GR', 'NL': 'LV', 'NO': 'FR', 'PL': 'IL', 'PT': 'IT',
    'RS': 'IE', 'SE': 'GB', 'SI': 'HR', 'SK': 'BE', 'UA': 'AT',
}

COUNTRY_NAMES = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CH': 'Switzerland',
    'CY': 'Cyprus', 'DE': 'Germany', 'EE': 'Estonia', 'ES': 'Spain',
    'FI': 'Finland', 'FR': 'France', 'GB': 'United Kingdom', 'GR': 'Greece',
    'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland', 'IL': 'Israel',
    'IS': 'Iceland', 'IT': 'Italy', 'LT': 'Lithuania', 'LV': 'Latvia',
    'ME': 'Montenegro', 'NL': 'Netherlands', 'NO': 'Norway', 'PL': 'Poland',
    'PT': 'Portugal', 'RS': 'Serbia', 'SE': 'Sweden', 'SI': 'Slovenia',
    'SK': 'Slovakia', 'UA': 'Ukraine'
}

# ESS missing value codes to filter out
# Note: ESS uses different codes by variable (77=Refusal, 88=DK, 99=NA,
# 999=NA for age, 7777/8888/9999 for year of birth)
ESS_MISSING_CODES = {55, 66, 77, 88, 99, 666, 777, 888, 999,
                     6666, 7777, 8888, 9999}

# ============================================================================
# SOCIAL TRUST VARIABLES (Breyer Social Trust Scale, ESS)
# Ref: https://zis.gesis.org/skala/Breyer-Social-Trust-Scale-(ESS)
# ============================================================================

SOCIAL_TRUST_VARIABLES = {
    'ppltrst': {
        'scale': '0-10',
        'label': 'Most people can be trusted, or you can\'t be too careful',
        'question': 'Using this card, generally speaking, would you say that most people can be trusted, or that you can\'t be too careful in dealing with people? Please tell me on a score of 0 to 10, where 0 means you can\'t be too careful and 10 means that most people can be trusted.',
        'domain': 'social_trust',
        'low_label': 'You can\'t be too careful',
        'high_label': 'Most people can be trusted'
    },
    'pplfair': {
        'scale': '0-10',
        'label': 'Most people try to be fair, or try to take advantage',
        'question': 'Using this card, do you think that most people would try to take advantage of you if they got the chance, or would they try to be fair? Please tell me on a score of 0 to 10, where 0 means most people would try to take advantage and 10 means most people would try to be fair.',
        'domain': 'social_trust',
        'low_label': 'Most people try to take advantage',
        'high_label': 'Most people try to be fair'
    },
    'pplhlp': {
        'scale': '0-10',
        'label': 'Most of the time people helpful or mostly looking out for themselves',
        'question': 'Would you say that most of the time people try to be helpful or that they are mostly looking out for themselves? Please tell me on a score of 0 to 10, where 0 means people mostly look out for themselves and 10 means people mostly try to be helpful.',
        'domain': 'social_trust',
        'low_label': 'People mostly look out for themselves',
        'high_label': 'People mostly try to be helpful'
    },
}

# ============================================================================
# FULL VARIABLE SET (42 variables available in ESS11)
# Excludes: 21 Schwartz Personal Values + 3 Justice items (not in ESS11)
# ============================================================================

ESS_VARIABLES = {
    # --- Social Trust (3) - CORE FOCUS ---
    **SOCIAL_TRUST_VARIABLES,
    'sclmeet': {'scale': '1-7', 'label': 'How often socially meet with friends, relatives or colleagues', 'domain': 'social_trust'},
    'inprdsc': {'scale': '0-6', 'label': 'Anyone to discuss intimate and personal matters with', 'domain': 'social_trust'},

    # --- Political Attitudes (5) ---
    'stfdem': {'scale': '0-10', 'label': 'How satisfied with the way democracy works in country', 'domain': 'political'},
    'polintr': {'scale': '1-4', 'label': 'How interested in politics', 'domain': 'political'},
    'psppipla': {'scale': '1-5', 'label': 'Political system allows people to have influence on politics', 'domain': 'political'},
    'psppsgva': {'scale': '1-5', 'label': 'Political system allows people to have a say in what government does', 'domain': 'political'},
    'vote': {'scale': '1-3', 'label': 'Voted last national election', 'domain': 'political'},

    # --- Political Efficacy (2) ---
    'actrolga': {'scale': '1-5', 'label': 'Able to take active role in a group involved with political issues', 'domain': 'efficacy'},
    'cptppola': {'scale': '1-5', 'label': 'Confident in own ability to participate in politics', 'domain': 'efficacy'},

    # --- Institutional Trust (7) ---
    'trstprl': {'scale': '0-10', 'label': 'Trust in country\'s parliament', 'domain': 'inst_trust'},
    'trstplt': {'scale': '0-10', 'label': 'Trust in politicians', 'domain': 'inst_trust'},
    'trstlgl': {'scale': '0-10', 'label': 'Trust in the legal system', 'domain': 'inst_trust'},
    'trstprt': {'scale': '0-10', 'label': 'Trust in political parties', 'domain': 'inst_trust'},
    'trstplc': {'scale': '0-10', 'label': 'Trust in the police', 'domain': 'inst_trust'},
    'trstep':  {'scale': '0-10', 'label': 'Trust in the European Parliament', 'domain': 'inst_trust'},
    'trstun':  {'scale': '0-10', 'label': 'Trust in the United Nations', 'domain': 'inst_trust'},

    # --- Subjective Well-being (6) ---
    'happy':   {'scale': '0-10', 'label': 'How happy are you', 'domain': 'wellbeing'},
    'stflife': {'scale': '0-10', 'label': 'How satisfied with life as a whole', 'domain': 'wellbeing'},
    'stfeco':  {'scale': '0-10', 'label': 'How satisfied with present state of economy in country', 'domain': 'wellbeing'},
    'stfgov':  {'scale': '0-10', 'label': 'How satisfied with the national government', 'domain': 'wellbeing'},
    'stfhlth': {'scale': '0-10', 'label': 'State of health services in country nowadays', 'domain': 'wellbeing'},
    'stfedu':  {'scale': '0-10', 'label': 'State of education in country nowadays', 'domain': 'wellbeing'},

    # --- Immigration Attitudes (6) ---
    'imueclt': {'scale': '0-10', 'label': 'Country\'s cultural life undermined or enriched by immigrants', 'domain': 'immigration'},
    'imwbcnt': {'scale': '0-10', 'label': 'Immigrants make country worse or better place to live', 'domain': 'immigration'},
    'imbgeco': {'scale': '0-10', 'label': 'Immigration bad or good for country\'s economy', 'domain': 'immigration'},
    'imsmetn': {'scale': '1-4', 'label': 'Allow many/few immigrants of same race/ethnic group as majority', 'domain': 'immigration'},
    'impcntr': {'scale': '1-4', 'label': 'Allow many/few immigrants from poorer countries outside Europe', 'domain': 'immigration'},
    'imdfetn': {'scale': '1-4', 'label': 'Allow many/few immigrants of different race/ethnic group from majority', 'domain': 'immigration'},

    # --- Social Values (5) ---
    'gincdif':  {'scale': '1-5', 'label': 'Government should reduce differences in income levels', 'domain': 'values'},
    'freehms':  {'scale': '1-5', 'label': 'Gay men and lesbians free to live life as they wish', 'domain': 'values'},
    'hmsfmlsh': {'scale': '1-5', 'label': 'Ashamed if close family member gay or lesbian', 'domain': 'values'},
    'hmsacld':  {'scale': '1-5', 'label': 'Gay male or lesbian couple right to adopt children', 'domain': 'values'},
    'euftf':    {'scale': '0-10', 'label': 'European Union: European unification go further or gone too far', 'domain': 'values'},

    # --- National Attachment (2) ---
    'atchctr': {'scale': '0-10', 'label': 'How emotionally attached to country', 'domain': 'attachment'},
    'atcherp': {'scale': '0-10', 'label': 'How emotionally attached to Europe', 'domain': 'attachment'},

    # --- Health, Religion, Safety (3) ---
    'health':  {'scale': '1-5', 'label': 'Subjective general health', 'domain': 'health'},
    'rlgatnd': {'scale': '1-7', 'label': 'How often attend religious services apart from special occasions', 'domain': 'religion'},
    'aesfdrk': {'scale': '1-4', 'label': 'Feeling of safety walking alone in local area after dark', 'domain': 'safety'},

    # --- Income Perception (1) ---
    'hincfel': {'scale': '1-4', 'label': 'Feeling about household\'s income nowadays', 'domain': 'income'},
}

# ---------------------------------------------------------------------------
# v12 FIX (P0-A): per-item valid response range, derived from the very scale
# string that is shown to the model in the prompt.
#
# WHY THIS EXISTS. ESS codes Refusal / Don't know / No answer as 7, 8 and 9 on
# every item whose substantive scale has fewer than seven categories, and as
# 77 / 88 / 99 on the wider scales. ESS_MISSING_CODES contains only the two-
# and three-digit variants, so sixteen of the forty-two outcome items retained
# non-substantive codes as valid HUMAN data: hmsfmlsh (4.4% of cases),
# hmsacld (3.7), freehms (2.7), impcntr (2.5), imdfetn (2.3), imsmetn (2.2),
# psppsgva (2.1), cptppola (1.9), gincdif (1.9), psppipla (1.8), actrolga
# (1.8), hincfel (1.4), aesfdrk (1.1), vote (1.0), polintr (0.2), health (0.2).
# The silicon side was never affected, because parse_response() validates every
# generated value against the same range. The contamination was therefore
# one-sided: it inflated human country means and human standard deviations on
# exactly the coarse-scale, predominantly reverse-coded items on which the
# scale-direction mechanism is identified.
#
# The range is built here, at import time, from the pristine ESS_VARIABLES, so
# that (i) --variables cannot shrink it and (ii) --scale_labels anchored cannot
# corrupt it by rewriting info['scale'].
# ---------------------------------------------------------------------------
def _parse_scale_range(scale_str):
    """Return (lo, hi) from a scale string such as '0-10'; None if unparsable."""
    m = re.match(r'\s*(-?\d+\.?\d*)\s*-\s*(-?\d+\.?\d*)', str(scale_str))
    if not m:
        return None
    lo, hi = float(m.group(1)), float(m.group(2))
    return (min(lo, hi), max(lo, hi))


ITEM_VALID_RANGE = {k: _parse_scale_range(v['scale'])
                    for k, v in ESS_VARIABLES.items()
                    if _parse_scale_range(v['scale']) is not None}

# Verbal anchor labels extracted from the ESS11 codebook (ESS11e04_2).  # v9 CHANGE
# Every value whose codebook label is verbal (non-numeric) is listed,
# reproducing the response card shown to human respondents.
ANCHOR_LABELS = {
    'ppltrst': {0: "You can't be too careful", 10: 'Most people can be trusted'},
    'pplfair': {0: 'Most people try to take advantage of me', 10: 'Most people try to be fair'},
    'pplhlp': {0: 'People mostly look out for themselves', 10: 'People mostly try to be helpful'},
    'sclmeet': {1: 'Never', 2: 'Less than once a month', 3: 'Once a month', 4: 'Several times a month', 5: 'Once a week', 6: 'Several times a week', 7: 'Every day'},
    'inprdsc': {0: 'None', 1: '1', 2: '2', 3: '3', 4: '4-6', 5: '7-9', 6: '10 or more'},  # v11 FIX: gap at 1-3 would reduce treatment dosage on a DiD CONTROL item
    'stfdem': {0: 'Extremely dissatisfied', 10: 'Extremely satisfied'},
    'polintr': {1: 'Very interested', 2: 'Quite interested', 3: 'Hardly interested', 4: 'Not at all interested'},
    'psppipla': {1: 'Not at all', 2: 'Very little', 3: 'Some', 4: 'A lot', 5: 'A great deal'},
    'psppsgva': {1: 'Not at all', 2: 'Very little', 3: 'Some', 4: 'A lot', 5: 'A great deal'},
    'vote': {1: 'Yes', 2: 'No', 3: 'Not eligible to vote'},
    'actrolga': {1: 'Not at all able', 2: 'A little able', 3: 'Quite able', 4: 'Very able', 5: 'Completely able'},
    'cptppola': {1: 'Not at all confident', 2: 'A little confident', 3: 'Quite confident', 4: 'Very confident', 5: 'Completely confident'},
    'trstprl': {0: 'No trust at all', 10: 'Complete trust'},
    'trstplt': {0: 'No trust at all', 10: 'Complete trust'},
    'trstlgl': {0: 'No trust at all', 10: 'Complete trust'},
    'trstprt': {0: 'No trust at all', 10: 'Complete trust'},
    'trstplc': {0: 'No trust at all', 10: 'Complete trust'},
    'trstep': {0: 'No trust at all', 10: 'Complete trust'},
    'trstun': {0: 'No trust at all', 10: 'Complete trust'},
    'happy': {0: 'Extremely unhappy', 10: 'Extremely happy'},
    'stflife': {0: 'Extremely dissatisfied', 10: 'Extremely satisfied'},
    'stfeco': {0: 'Extremely dissatisfied', 10: 'Extremely satisfied'},
    'stfgov': {0: 'Extremely dissatisfied', 10: 'Extremely satisfied'},
    'stfhlth': {0: 'Extremely bad', 10: 'Extremely good'},
    'stfedu': {0: 'Extremely bad', 10: 'Extremely good'},
    'imueclt': {0: 'Cultural life undermined', 10: 'Cultural life enriched'},
    'imwbcnt': {0: 'Worse place to live', 10: 'Better place to live'},
    'imbgeco': {0: 'Bad for the economy', 10: 'Good for the economy'},
    'imsmetn': {1: 'Allow many to come and live here', 2: 'Allow some', 3: 'Allow a few', 4: 'Allow none'},
    'impcntr': {1: 'Allow many to come and live here', 2: 'Allow some', 3: 'Allow a few', 4: 'Allow none'},
    'imdfetn': {1: 'Allow many to come and live here', 2: 'Allow some', 3: 'Allow a few', 4: 'Allow none'},
    'gincdif': {1: 'Agree strongly', 2: 'Agree', 3: 'Neither agree nor disagree', 4: 'Disagree', 5: 'Disagree strongly'},
    'freehms': {1: 'Agree strongly', 2: 'Agree', 3: 'Neither agree nor disagree', 4: 'Disagree', 5: 'Disagree strongly'},
    'hmsfmlsh': {1: 'Agree strongly', 2: 'Agree', 3: 'Neither agree nor disagree', 4: 'Disagree', 5: 'Disagree strongly'},
    'hmsacld': {1: 'Agree strongly', 2: 'Agree', 3: 'Neither agree nor disagree', 4: 'Disagree', 5: 'Disagree strongly'},
    'euftf': {0: 'Unification already gone too far', 10: 'Unification go further'},
    'atchctr': {0: 'Not at all emotionally attached', 10: 'Very emotionally attached'},
    'atcherp': {0: 'Not at all emotionally attached', 10: 'Very emotionally attached'},
    'health': {1: 'Very good', 2: 'Good', 3: 'Fair', 4: 'Bad', 5: 'Very bad'},
    'rlgatnd': {1: 'Every day', 2: 'More than once a week', 3: 'Once a week', 4: 'At least once a month', 5: 'Only on special holy days', 6: 'Less often', 7: 'Never'},
    'aesfdrk': {1: 'Very safe', 2: 'Safe', 3: 'Unsafe', 4: 'Very unsafe'},
    'hincfel': {1: 'Living comfortably on present income', 2: 'Coping on present income', 3: 'Difficult on present income', 4: 'Very difficult on present income'},
}

ENABLE_EDUYRS_GUARD = False  # v11: OFF by design - see note in generate_backstory()

# Demographic variables used for backstory generation
# EXPANDED per supervisor feedback (Patrick Sturgis, Feb 2025):
#   Added: sclmeet, inprdsc, netusoft, polintr, clsprty, mbtru, dscrgrp,
#          chldhhe, agea/yrbrn remain, edulvlb for earlier rounds
BACKSTORY_DEMOGRAPHICS = [
    # Core demographics
    'cntry', 'gndr', 'agea', 'yrbrn', 'maritalb', 'eduyrs', 'eisced',
    'hinctnta', 'hincfel', 'brncntr', 'ctzcntr', 'region', 'domicil',
    'mainact', 'health', 'rlgatnd', 'lrscale',
    # Social integration (NEW)
    'sclmeet',    # How often socially meet friends/relatives (1-7)
    'inprdsc',    # Anyone to discuss intimate matters with (0-6)
    # Political engagement (NEW)
    'polintr',    # Interest in politics (1-4)
    'clsprty',    # Feel closer to a particular party (1-2)
    'vote',       # Voted in last election (1-3)
    # Household composition (NEW)
    'chldhhe',    # Children living at home (1-2)
    'hhmmb',      # Number of people in household
    # Union and discrimination (NEW)
    'mbtru',      # Member of trade union (1-3)
    'dscrgrp',    # Member of discriminated group (1-2)
    # Internet use (NEW)
    'netusoft',   # Internet use frequency (1-5)
]

# Variables for similarity matching (RQ11)
SIMILARITY_VARS = ['agea', 'gndr', 'eduyrs', 'hinctnta', 'cntry']

# ============================================================================
# VALUE LABEL MAPPINGS (ESS codebook)
# ============================================================================

GENDER_MAP = {1: 'male', 2: 'female'}

MARITAL_MAP = {
    1: 'legally married',
    2: 'in a legally registered civil union',
    3: 'legally separated',
    4: 'legally divorced or civil union dissolved',
    5: 'widowed or civil partner died',
    6: 'never married and never in a civil union'
}

# ESS R1-R5 use 'marital' or 'maritala' with slightly different codes
MARITAL_MAP_EARLY = {
    1: 'legally married',
    2: 'legally separated',
    3: 'legally divorced',
    4: 'widowed',
    5: 'never married',
    6: 'never married',  # R3 maritala code
    9: 'in a legally registered civil union'
}

EISCED_MAP = {
    0: 'not completed primary education',
    1: 'primary or first stage of basic education',
    2: 'lower secondary or second stage of basic education',
    3: 'lower tier upper secondary education',
    4: 'upper tier upper secondary education',
    5: 'advanced vocational or sub-degree education',
    6: 'lower tertiary education (BA level)',
    7: 'higher tertiary education (MA level or above)',
    55: None  # other
}

INCOME_FEELING_MAP = {
    1: 'living comfortably on present income',
    2: 'coping on present income',
    3: 'finding it difficult on present income',
    4: 'finding it very difficult on present income'
}

DOMICIL_MAP = {
    1: 'a big city',
    2: 'the suburbs or outskirts of a big city',
    3: 'a town or a small city',
    4: 'a country village',
    5: 'a farm or home in the countryside'
}

MAINACT_MAP = {
    1: 'paid work',
    2: 'education',
    3: 'unemployed and actively looking for a job',
    4: 'unemployed and not actively looking for a job',
    5: 'permanently sick or disabled',
    6: 'retired',
    7: 'community or military service',
    8: 'doing housework, looking after children or other persons',
    9: 'other'
}

HEALTH_MAP = {
    1: 'very good',
    2: 'good',
    3: 'fair',
    4: 'bad',
    5: 'very bad'
}

RLGATND_MAP = {
    1: 'every day',
    2: 'more than once a week',
    3: 'once a week',
    4: 'at least once a month',
    5: 'only on special holy days',
    6: 'less often',
    7: 'never'
}

# --- NEW VALUE MAPS for expanded backstory (v5) ---

SCLMEET_MAP = {
    1: 'never',
    2: 'less than once a month',
    3: 'once a month',
    4: 'several times a month',
    5: 'once a week',
    6: 'several times a week',
    7: 'every day'
}

INPRDSC_MAP = {
    0: 'none',
    1: '1 person',
    2: '2 people',
    3: '3 people',
    4: '4-6 people',
    5: '7-9 people',
    6: '10 or more people'
}

POLINTR_MAP = {
    1: 'very interested',
    2: 'quite interested',
    3: 'hardly interested',
    4: 'not at all interested'
}

NETUSOFT_MAP = {
    1: 'never',
    2: 'only occasionally',
    3: 'a few times a week',
    4: 'most days',
    5: 'every day'
}

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

PROMPT_1P_STANDARD = """Adopt the following persona and respond as if you were this person.

{backstory}

Question: {question}
Scale: {scale}

Respond with ONLY a single number within the scale range. No explanation."""

PROMPT_3P = """Consider a person with the following background:

{backstory}

If asked "{question}", how would this person respond on a scale of {scale}?

Respond with ONLY a single number within the scale range. No explanation."""

PROMPT_1P_IDIOSYNCRATIC = """Adopt the following persona. Feel free to give unusual responses that might differ from what's typical - express your individual perspective.

{backstory}

Question: {question}
Scale: {scale}

Respond with ONLY a single number within the scale range. No explanation."""

PROMPT_FEWSHOT = """Here are responses from people with similar backgrounds:

{examples}

Now consider this person:
{backstory}

Question: {question}
Scale: {scale}

Based on how similar people responded, what would this person answer?
Respond with ONLY a single number within the scale range. No explanation."""

PROMPT_DIST_ANCHOR = """In {country}, responses to the question "{question}" typically follow this pattern:
- Mean: {pop_mean:.1f}
- Standard deviation: {pop_sd:.1f}
- Scale: {scale}

Now consider this specific person:
{backstory}

Given their specific background and life circumstances, where would they personally fall on this scale?
Respond with ONLY a single number within the scale range. No explanation."""


def select_prompt_template(prompt_mode):
    """Select prompt template by mode name."""
    templates = {
        '1p': PROMPT_1P_STANDARD,
        '3p': PROMPT_3P,
        '1p_idiosyncratic': PROMPT_1P_IDIOSYNCRATIC,
    }
    if prompt_mode not in templates:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}. Choose from {list(templates.keys())}")
    return templates[prompt_mode]

# ============================================================================
# DYNAMIC BACKSTORY GENERATION
# ============================================================================

def _safe_int(val):
    """Convert a value to int, returning None if missing/invalid."""
    if pd.isna(val):
        return None
    try:
        v = int(float(val))
        if v in ESS_MISSING_CODES:
            return None
        return v
    except (ValueError, TypeError):
        return None


def harmonize_ess_variables(df):
    """
    Harmonize variable names across ESS rounds.

    R1-R5 use different variable names for some demographics:
      - marital / maritala  → maritalb
      - hinctnt             → hinctnta
      - region{cc}          → region

    This function renames columns in-place so downstream code works uniformly.
    """
    # Marital status: marital (R1) or maritala (R3-R5) → maritalb
    if 'maritalb' not in df.columns:
        if 'maritala' in df.columns:
            df['maritalb'] = df['maritala']
            df['_marital_early'] = True  # flag for backstory mapping
            logging.info("  Harmonized: maritala → maritalb (early-round codes)")
        elif 'marital' in df.columns:
            df['maritalb'] = df['marital']
            df['_marital_early'] = True
            logging.info("  Harmonized: marital → maritalb (early-round codes)")
        else:
            df['_marital_early'] = False
    else:
        df['_marital_early'] = False

    # Income decile: hinctnt (R1-R5, 12 categories) → hinctnta (R6+, 10 deciles)
    if 'hinctnta' not in df.columns:
        if 'hinctnt' in df.columns:
            # hinctnt is 1-12 in R1-R5; roughly map to 1-10 deciles
            # R1-R3: 1=lowest...12=highest → compress to 1-10
            hinctnt = pd.to_numeric(df['hinctnt'], errors='coerce')
            # Linear rescale: 1-12 → 1-10
            df['hinctnta'] = np.round(1 + (hinctnt - 1) * 9 / 11).astype('Int64')
            logging.info("  Harmonized: hinctnt (1-12) → hinctnta (1-10)")

    # Region: R1-R5 use country-specific columns (regiongb, regionfr, etc.)
    if 'region' not in df.columns:
        region_cols = [c for c in df.columns if c.startswith('region') or c.startswith('regio')]
        if region_cols:
            # Coalesce all region columns into one (use first non-null)
            df['region'] = pd.Series([np.nan] * len(df), dtype='object')
            for col in region_cols:
                col_vals = df[col].astype(str).replace('nan', np.nan)
                mask = df['region'].isna() & col_vals.notna()
                df.loc[mask, 'region'] = col_vals[mask]
            logging.info(f"  Harmonized: {len(region_cols)} region columns → region")

    # Add survey year from essround
    if 'essround' in df.columns and 'survey_year' not in df.columns:
        df['survey_year'] = df['essround'].map(ESS_ROUND_YEAR)

    return df


def generate_backstory(row, mode='v5_clean'):
    """
    Dynamically generate a natural-language backstory from ESS demographic variables.

    Parameters
    ----------
    row : pd.Series
        A single row from the ESS dataframe containing demographic variables.
    mode : str
        'v4'         = original 17 demographic variables (includes leaked)
        'v5'         = expanded 27 variables (includes leaked)
        'v5_clean'   = 20 variables (v5 minus 7 leaked outcome variables) [DEFAULT]
        'demo_only'  = 3 core demographics WITHOUT country: gender, age, birth year
        'minimal'    = 4 core variables: country, gender, age, birth year
        'ses'        = minimal + family, geography, education, economic (13 vars)  # v10 FIX (M1)
        'political'  = ses + left-right scale ONLY (14 vars; clsprty is gated
                       behind v5_extras and appears only in full_clean)  # v10 FIX (M1)
        'full_clean' = same as v5_clean (20 vars)
        'full_nocountry' = full_clean MINUS country sentence and NUTS region  # v8 CHANGE
        'full_noregion'  = full_clean MINUS NUTS region only (country kept)   # v10 CHANGE
                           PRIMARY CONTRAST: full_noregion vs full_nocountry
                           (both region-free; only difference = country label)
        'full_nopolitical' = full_clean MINUS lrscale and clsprty              # v10 CHANGE
        'minimal_politics' = minimal PLUS lrscale and clsprty (6 vars)          # v12.2
                           (optional LOGO of the quasi-attitudinal group)

    The 7 leaked variables (excluded in v5_clean/full_clean/minimal/ses/political):
        health, hincfel, rlgatnd, sclmeet, inprdsc, polintr, vote
    These overlap with the 42 outcome variables and create information leakage.

    Returns
    -------
    str
        A natural-language backstory paragraph.
    """
    parts = []

    # Determine inclusion flags
    include_leaked = mode in ('v4', 'v5')
    include_country = mode not in ('demo_only', 'full_nocountry')   # v8 CHANGE
    include_ses = mode not in ('minimal', 'demo_only', 'minimal_politics', 'minimal_ageonly', 'minimal_yrbrnonly', 'country_only')  # v12.2: minimal_politics = minimal + political block
    # v12.3 (2 Aug 2026): two finer cumulative rungs over minimal_politics.
    # 'minimal_politics_ses' adds only the SES triad sentences (education,
    # main activity, income); 'minimal_politics_ses_geo' adds geography and
    # migration/civic status (domicile+region, born in country, citizenship).
    # Family, household and membership sentences stay out of both.
    _ses_tier = mode in ('minimal_politics_ses', 'minimal_politics_econ')  # v12.3: econ = SES triad + union + internet
    _geo_tier = mode == 'minimal_politics_ses_geo'
    render_family = include_ses and not (_ses_tier or _geo_tier)
    render_geo = include_ses and not _ses_tier
    include_political_vars = mode not in ('minimal', 'ses', 'demo_only', 'full_nopolitical')  # v10 CHANGE: LOGO of political-identity group (lrscale + clsprty)
    # v12.4 (2 Aug 2026): six leave-one-group-out arms against full_clean,
    # one per theory group from the G1-G7 grouping in the header (G6
    # political already exists as full_nopolitical; the country and region
    # rows come from the existing region-free pair). Country and the NUTS
    # code stay IN for all six, so each arm differs from full_clean by
    # exactly one group of sentences.
    # v12.5 (2 Aug 2026): the LOGO blocks are aligned to NEST inside the
    # regrouped ladder tiers, so the paper carries ONE taxonomy of the 20
    # profile variables. The v12.4 markers block straddled tiers 4 and 5
    # and was replaced, before it ran, by the membership block (union +
    # internet, the tier-4 remainder) and the minority block
    # (discrimination-group membership, tier 5's minority-status class).
    # The partition is now exhaustive and exclusive over all 20 variables.
    _LOGO_MODES = ('full_noascriptive',   # ladder level 1 content: gender, age, birth year
                   'full_nosocioecon',    # tier 4 SES triad: education, main activity, income
                   'full_nomembership',   # tier 4 remainder: union membership, internet use
                   'full_nohousehold',    # tier 5: marital status, household size, children
                   'full_nocivic',        # tier 5: born in country, citizenship
                   'full_nominority',     # tier 5: discrimination-group membership
                   'full_nodomicil')      # tier 5: the domicile phrase
    _logo = mode if mode in _LOGO_MODES else None
    # v12.6: add-one arms (minimal + exactly one block), the age/birth-year
    # twins, the country-only probe, the income probe, and the swap arm.
    _ADD_MODES = {'minimal_ses': 'ses3',           # + education, activity, income
                  'minimal_membership': 'member',  # + union, internet
                  'minimal_household': 'house',    # + marital, size, children
                  'minimal_civic': 'civic',        # + born here, citizen
                  'minimal_minority': 'minor',     # + discriminated group
                  'minimal_domicil': 'domic',      # + the domicile phrase
                  'minimal_region': 'region',     # + the NUTS code alone
                  # v12.7 (3 Aug 2026): the membership block was the one
                  # non-country block with a positive add margin, and it is
                  # a residual category (tier 4 minus the SES triad), so its
                  # two variables get single-variable add arms. Clean here:
                  # neither has a redundant twin elsewhere in the profile.
                  'minimal_union': 'union1',       # + union membership alone
                  'minimal_internet': 'net1'}      # + internet use alone
    _add = _ADD_MODES.get(mode)

    def _ok(g):
        # under an add-one mode, only the target block's sentences render
        return _add is None or _add == g
    if _add is not None or mode in ('minimal_ageonly', 'minimal_yrbrnonly', 'country_only'):
        include_political_vars = False
    include_v5_extras = mode in (('v5', 'v5_clean', 'full_clean', 'full_nocountry', 'full_noregion', 'full_nopolitical', 'full_noincome', 'full_swapcountry', 'minimal_household', 'minimal_membership', 'minimal_minority', 'minimal_union', 'minimal_internet') + _LOGO_MODES)  # v10 CHANGE; v12.4 adds the LOGO arms
    include_region = mode not in ('full_nocountry', 'full_noregion', 'full_swapcountry')  # v10 CHANGE (B1): both no-* arms drop the NUTS code, so full_noregion vs full_nocountry differ ONLY in the country sentence

    # --- MINIMAL LEVEL: country, gender, age, birth year (all modes) ---

    # Survey year (for overtime analysis)
    survey_year = _safe_int(row.get('survey_year'))
    if survey_year:
        parts.append(f"This survey was conducted in {survey_year}.")

    # Country
    cntry = row.get('cntry', None)
    if mode == 'full_swapcountry' and cntry:
        cntry = SWAP_MAP.get(cntry, cntry)   # v12.6: the wrong-label arm
    country_name = COUNTRY_NAMES.get(cntry, cntry) if cntry else None
    if include_country and country_name:
        parts.append(f"I live in {country_name}.")

    # Gender
    gndr = _safe_int(row.get('gndr')) if _logo != 'full_noascriptive' and mode != 'country_only' else None  # v12.4/12.6
    gender_str = GENDER_MAP.get(gndr)
    if gender_str:
        parts.append(f"I am {gender_str}.")

    # Age and birth year
    agea = _safe_int(row.get('agea')) if _logo != 'full_noascriptive' and mode not in ('minimal_yrbrnonly', 'country_only') else None  # v12.4/12.6
    yrbrn = _safe_int(row.get('yrbrn')) if _logo != 'full_noascriptive' and mode not in ('minimal_ageonly', 'country_only') else None  # v12.4/12.6
    if agea and not (0 <= agea <= 120):
        agea = None
    if yrbrn and not (1880 <= yrbrn <= 2010):
        yrbrn = None
    if agea and yrbrn:
        parts.append(f"I am {agea} years old and I was born in {yrbrn}.")
    elif agea:
        parts.append(f"I am {agea} years old.")
    elif yrbrn and mode == 'minimal_yrbrnonly':
        # v12.6: the birth-year twin. Gated to this mode so respondents with
        # a missing age render identically in every pre-existing arm.
        parts.append(f"I was born in {yrbrn}.")

    # --- SES LEVEL: family, geography, education, economic ---
    if include_ses:
        # Marital status
        marital = _safe_int(row.get('maritalb')) if render_family and _logo != 'full_nohousehold' and _ok('house') else None  # v12.4/12.6
        use_early_map = row.get('_marital_early', False)
        if use_early_map:
            marital_str = MARITAL_MAP_EARLY.get(marital)
        else:
            marital_str = MARITAL_MAP.get(marital)
        if marital_str:
            parts.append(f"I am {marital_str}.")

        # Region and domicile
        domicil = _safe_int(row.get('domicil')) if render_geo and _logo != 'full_nodomicil' and _ok('domic') else None  # v12.4/12.6
        domicil_str = DOMICIL_MAP.get(domicil)
        region = row.get('region', None)
        if domicil_str and region and not pd.isna(region) and include_region and _ok('region'):  # v8 CHANGE; v12.6 _ok
            parts.append(f"I live in {domicil_str} in the {region} region.")
        elif domicil_str:
            parts.append(f"I live in {domicil_str}.")
        elif (_logo == 'full_nodomicil' or _add == 'region') and region and not pd.isna(region) and include_region:
            # v12.4: the domicile phrase is the ablated content; the NUTS code
            # must survive its removal or the arm would also ablate the region.
            # Gated to this mode so every existing arm renders byte-identically.
            parts.append(f"I live in the {region} region.")

        # Born in country / citizenship
        brncntr = _safe_int(row.get('brncntr')) if render_geo and _logo != 'full_nocivic' and _ok('civic') else None  # v12.4/12.6
        if brncntr == 1:
            parts.append(f"I was born in this country.")
        elif brncntr == 2:
            parts.append(f"I was not born in this country.")

        ctzcntr = _safe_int(row.get('ctzcntr')) if render_geo and _logo != 'full_nocivic' and _ok('civic') else None  # v12.4/12.6
        if ctzcntr == 1:
            parts.append(f"I am a citizen of this country.")
        elif ctzcntr == 2:
            parts.append(f"I am not a citizen of this country.")

        # Education
        eisced = _safe_int(row.get('eisced')) if _logo != 'full_nosocioecon' and _ok('ses3') else None  # v12.4/12.6
        eisced_str = EISCED_MAP.get(eisced)
        eduyrs = _safe_int(row.get('eduyrs')) if _logo != 'full_nosocioecon' and _ok('ses3') else None  # v12.4/12.6
        # v11 NOTE (verified empirically, not inferred from code): eduyrs has no
        # explicit range check, but _safe_int() already drops ESS_MISSING_CODES
        # {55,66,77,88,99,...}, so the 901 respondents coded 77/88/99 (1.8%) are NOT
        # prompted with "88 years of education" - they simply lose the years clause.
        # What does survive is 15 respondents (0.03%) with genuine-but-implausible
        # values 41-69 (AT, CY, ES; NOT Montenegro or Israel). The guard below is
        # therefore cosmetic; enabling it would make new arms prompt-incomparable
        # with every completed run, so it stays OFF. Handle analytically instead
        # (exclusion robustness), and report the true magnitude, not the code-reading
        # estimate.
        if ENABLE_EDUYRS_GUARD and eduyrs is not None and not (0 <= eduyrs <= 40):
            eduyrs = None
        if eisced_str and eduyrs:
            parts.append(f"My highest level of education is {eisced_str}, with {eduyrs} years of full-time education.")
        elif eisced_str:
            parts.append(f"My highest level of education is {eisced_str}.")
        elif eduyrs:
            parts.append(f"I have {eduyrs} years of full-time education.")

        # Main activity
        mainact = _safe_int(row.get('mainact')) if _logo != 'full_nosocioecon' and _ok('ses3') else None  # v12.4/12.6
        mainact_str = MAINACT_MAP.get(mainact)
        if mainact_str:
            parts.append(f"My main activity is {mainact_str}.")

        # Income decile (NOT leaked; hinctnta is not an outcome variable)
        hinctnta = _safe_int(row.get('hinctnta')) if _logo != 'full_nosocioecon' and _ok('ses3') and mode != 'full_noincome' else None  # v12.4/12.6
        if hinctnta and 1 <= hinctnta <= 10:
            decile_desc = {1: 'the lowest', 2: 'the 2nd', 3: 'the 3rd', 4: 'the 4th', 5: 'the 5th',
                           6: 'the 6th', 7: 'the 7th', 8: 'the 8th', 9: 'the 9th', 10: 'the highest'}
            parts.append(f"My household income is in {decile_desc[hinctnta]} decile for this country.")

        # Income feeling (LEAKED — only include in v4/v5)
        if include_leaked:
            hincfel = _safe_int(row.get('hincfel'))
            hincfel_str = INCOME_FEELING_MAP.get(hincfel)
            if hincfel_str:
                parts.append(f"I would say I am {hincfel_str}.")

        # Health (LEAKED — only include in v4/v5)
        if include_leaked:
            health = _safe_int(row.get('health'))
            health_str = HEALTH_MAP.get(health)
            if health_str:
                parts.append(f"My general health is {health_str}.")

        # Religious attendance (LEAKED — only include in v4/v5)
        if include_leaked:
            rlgatnd = _safe_int(row.get('rlgatnd'))
            rlgatnd_str = RLGATND_MAP.get(rlgatnd)
            if rlgatnd_str:
                parts.append(f"I attend religious services {rlgatnd_str}.")

        # Household composition (v5 extra, but NOT leaked)
        if include_v5_extras and _logo != 'full_nohousehold':  # v12.4
            hhmmb = _safe_int(row.get('hhmmb')) if _ok('house') else None  # v12.6
            chldhhe = _safe_int(row.get('chldhhe'))
            if hhmmb and 1 <= hhmmb <= 20:
                if chldhhe == 1:
                    parts.append(f"I live in a household of {hhmmb} people, with children living at home.")
                elif chldhhe == 2:
                    parts.append(f"I live in a household of {hhmmb} people, with no children at home.")
                else:
                    parts.append(f"I live in a household of {hhmmb} people.")

    # --- POLITICAL LEVEL: political orientation, party ---
    if include_political_vars:
        lrscale = _safe_int(row.get('lrscale'))
        if lrscale is not None and 0 <= lrscale <= 10:
            parts.append(f"On a left (0) to right (10) political scale, I would place myself at {lrscale}.")

        if include_v5_extras or mode in ('minimal_politics', 'minimal_politics_ses', 'minimal_politics_ses_geo', 'minimal_politics_econ'):  # v12.2/v12.3
            # Party affiliation (NOT leaked)
            clsprty = _safe_int(row.get('clsprty'))
            if clsprty == 1:
                parts.append("I feel close to a particular political party.")
            elif clsprty == 2:
                parts.append("I do not feel close to any particular political party.")

        # Political interest (LEAKED — only include in v4/v5)
        if include_leaked and include_v5_extras:
            polintr = _safe_int(row.get('polintr'))
            polintr_str = POLINTR_MAP.get(polintr)
            if polintr_str:
                parts.append(f"I am {polintr_str} in politics.")

        # Voting (LEAKED — only include in v4/v5)
        if include_leaked and include_v5_extras:
            vote = _safe_int(row.get('vote'))
            if vote == 1:
                parts.append("I voted in the last national election.")
            elif vote == 2:
                parts.append("I did not vote in the last national election.")

    # --- FULL CLEAN LEVEL: social extras (v5 non-leaked) ---
    if include_v5_extras or mode == 'minimal_politics_econ':  # v12.3
        # Social meeting frequency (LEAKED — only include in v4/v5)
        if include_leaked:
            sclmeet = _safe_int(row.get('sclmeet'))
            sclmeet_str = SCLMEET_MAP.get(sclmeet)
            if sclmeet_str:
                parts.append(f"I socially meet with friends, relatives or colleagues {sclmeet_str}.")

        # Confidants (LEAKED — only include in v4/v5)
        if include_leaked:
            inprdsc = _safe_int(row.get('inprdsc'))
            inprdsc_str = INPRDSC_MAP.get(inprdsc)
            if inprdsc_str:
                parts.append(f"I have {inprdsc_str} with whom I can discuss intimate and personal matters.")

        # Trade union membership (NOT leaked)
        mbtru = _safe_int(row.get('mbtru')) if _logo != 'full_nomembership' and _add in (None, 'member', 'union1') else None  # v12.5-7
        if mbtru == 1:
            parts.append("I am currently a member of a trade union.")
        elif mbtru == 2:
            parts.append("I have previously been a member of a trade union.")
        elif mbtru == 3:
            parts.append("I have never been a member of a trade union.")

        # Discrimination (NOT leaked; background tier, so not in the econ rung)
        dscrgrp = _safe_int(row.get('dscrgrp')) if include_v5_extras and _logo != 'full_nominority' and _ok('minor') else None  # v12.5/12.6
        if dscrgrp == 1:
            parts.append("I consider myself a member of a group that is discriminated against in this country.")
        elif dscrgrp == 2:
            parts.append("I do not consider myself a member of a discriminated group.")

        # Internet use (NOT leaked)
        netusoft = _safe_int(row.get('netusoft')) if _logo != 'full_nomembership' and _add in (None, 'member', 'net1') else None  # v12.5-7
        netusoft_str = NETUSOFT_MAP.get(netusoft)
        if netusoft_str:
            parts.append(f"I use the internet {netusoft_str}.")

    return " ".join(parts)


def generate_backstories_for_df(df, mode='v5_clean'):
    """
    Generate backstories for all rows in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        ESS dataframe with demographic columns.
    mode : str
        'v4' = original 17 variables, 'v5' = expanded 27 variables.

    Returns
    -------
    pd.Series
        Series of backstory strings, indexed to match df.
    """
    logging.info(f"Generating backstories for {len(df)} respondents (mode={mode})...")
    backstories = df.apply(lambda row: generate_backstory(row, mode=mode), axis=1)
    logging.info(f"Generated {len(backstories)} backstories. "
                 f"Mean length: {backstories.str.len().mean():.0f} chars")
    return backstories


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(ess_file=None, countries=None, sample_per_country=None, seed=888,
              backstory_mode='v5_clean', missing_rule='range'):  # v12
    """
    Load ESS11 data and generate backstories dynamically.

    Parameters
    ----------
    ess_file : str, optional
        Path to ESS CSV file. Defaults to ESS11_FILE.
    countries : list of str, optional
        ISO country codes to filter (e.g., ['GB', 'FR', 'DE']).
        If None, loads all countries.
    sample_per_country : int, optional
        If specified, randomly sample this many respondents per country.
        Countries with fewer respondents than this value are kept in full.
    seed : int
        Random seed for reproducible sampling.
    backstory_mode : str
        'v5_clean' (20 vars, default), 'demo_only' (3, no country),
        'minimal' (4), 'ses' (14),
        'political' (14), 'full_clean' (=v5_clean),
        'v4' (17, includes leaked), 'v5' (27, includes leaked).
    missing_rule : str
        'range'  = drop values outside each item's valid range (v12 default;
                   removes ESS 7/8/9 refusal/DK/no-answer codes on coarse scales)
        'legacy' = drop only ESS_MISSING_CODES (pre-v12 behaviour)

    Returns
    -------
    pd.DataFrame
        Merged dataframe with backstories and all ESS variables.
    """
    if ess_file is None:
        ess_file = ESS11_FILE

    logging.info(f"Loading ESS data from {ess_file}...")

    # Determine columns to load
    target_vars = list(ESS_VARIABLES.keys())
    demo_vars = BACKSTORY_DEMOGRAPHICS
    meta_vars = ['idno', 'essround', 'pspwght', 'dweight', 'anweight', 'pweight']
    # Include alternate column names from earlier ESS rounds
    alt_vars = ['marital', 'maritala', 'hinctnt', 'edulvlb',
                'chldhm', 'chldhhe', 'hhmmb']  # v5: added household vars
    all_cols = list(set(meta_vars + demo_vars + target_vars + alt_vars))

    # Read CSV, selecting only needed columns (memory efficient)
    available_cols = pd.read_csv(ess_file, nrows=0).columns.tolist()
    # Also include any region* columns from earlier rounds
    region_cols = [c for c in available_cols if c.startswith('region') or c.startswith('regio')]
    cols_to_load = [c for c in all_cols if c in available_cols] + region_cols
    cols_to_load = list(set(cols_to_load))  # deduplicate
    missing_cols = [c for c in all_cols if c not in available_cols and c not in alt_vars]
    if missing_cols:
        logging.warning(f"Columns not found in data: {missing_cols}")

    df = pd.read_csv(ess_file, usecols=cols_to_load)
    logging.info(f"Loaded {len(df)} total respondents from {df['cntry'].nunique()} countries")

    # Filter by country if specified
    if countries:
        countries_upper = [c.upper() for c in countries]
        df = df[df['cntry'].isin(countries_upper)].copy()
        logging.info(f"Filtered to {len(df)} respondents from {countries_upper}")

    # Random sample per country if specified
    if sample_per_country is not None:
        sampled_parts = []
        for cntry in sorted(df['cntry'].unique()):
            cntry_df = df[df['cntry'] == cntry]
            if len(cntry_df) >= sample_per_country:
                # v12 FIX: was `>`. At `>`, a country whose full sample size
                # exactly equals sample_per_country (e.g. Cyprus at n=685, the
                # smallest of the 30 ESS11 edition 4.1 countries) fell through
                # to the "keep all" branch below and was used in raw file
                # order rather than randomly permuted. With `>=`,
                # .sample(n=len(cntry_df)) draws every row but in a seeded
                # random order, matching every other country's treatment and
                # removing any dependence on incidental file ordering.
                cntry_df = cntry_df.sample(n=sample_per_country, random_state=seed)
                logging.info(f"  {cntry}: sampled {sample_per_country} from {len(df[df['cntry'] == cntry])}")
            else:
                logging.info(f"  {cntry}: kept all {len(cntry_df)} (< {sample_per_country})")
            sampled_parts.append(cntry_df)
        df = pd.concat(sampled_parts, ignore_index=True)
        logging.info(f"After sampling: {len(df)} total respondents")

    # Clean ESS missing codes from target variables
    for var in target_vars:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors='coerce')
            df.loc[df[var].isin(ESS_MISSING_CODES), var] = np.nan

    # v12 FIX (P0-A): apply each item's own valid range. See ITEM_VALID_RANGE.
    # missing_rule='legacy' reproduces the pre-v12 behaviour exactly and exists
    # only so that a legacy raw file can be regenerated for gate comparisons.
    if missing_rule == 'range':
        n_dropped_total = 0
        for var in target_vars:
            rng = ITEM_VALID_RANGE.get(var)
            if var in df.columns and rng is not None:
                lo, hi = rng
                out_of_range = (df[var] < lo) | (df[var] > hi)
                n_dropped = int(out_of_range.sum())
                if n_dropped:
                    df.loc[out_of_range, var] = np.nan
                    n_dropped_total += n_dropped
                    logging.info(f"  {var}: {n_dropped} non-substantive values "
                                 f"set missing (valid range {lo:g}-{hi:g})")
        logging.info(f"Valid-range rule: {n_dropped_total} human values set missing "
                     f"across {len(target_vars)} items")
    elif missing_rule == 'legacy':
        logging.warning("missing_rule='legacy': ESS single-digit refusal/DK/no-answer "
                        "codes (7/8/9) are RETAINED as valid human data on coarse-scale "
                        "items. Use only to regenerate a legacy comparator.")
    else:
        raise ValueError(f"Unknown missing_rule: {missing_rule}")

    # Harmonize variable names across ESS rounds
    df = harmonize_ess_variables(df)

    # Generate backstories
    df['backstory'] = generate_backstories_for_df(df, mode=backstory_mode)

    logging.info(f"Final dataset: {len(df)} respondents, "
                 f"{df['cntry'].nunique()} countries")

    return df


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_logging(block_name, model_name):
    """Setup logging configuration."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"silicon_{block_name}_{model_name}_{timestamp}.log")

    # Reset handlers to avoid duplicate logging
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_response(text, scale_str=None):
    """
    Extract numeric response from model output.

    Parameters
    ----------
    text : str
        Raw model output text.
    scale_str : str, optional
        Scale string (e.g., '0-10', '1-5'). If provided, values outside
        the scale range are rejected as invalid.
    """
    if text is None:
        return None
    text = str(text).strip()
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        try:
            val = float(numbers[0])
            # Reject astronomical values (parsing artifacts)
            if abs(val) > 1000:
                return None
            # Validate against scale bounds if provided
            # v9 FIX: extract the numeric range from the HEAD of the scale
            # string with a regex, so validation also works for anchored
            # scale strings like "1-4 (1 = Very interested; ...)". The old
            # split('-') silently skipped validation on anchored strings,
            # which would have made validity filtering asymmetric between
            # numeric and anchored conditions.
            if scale_str:
                m = re.match(r'\s*(-?\d+\.?\d*)\s*-\s*(-?\d+\.?\d*)', str(scale_str))
                if m:
                    lo, hi = float(m.group(1)), float(m.group(2))
                    if lo > hi:
                        lo, hi = hi, lo
                    if val < lo or val > hi:
                        return None
            return val
        except (ValueError, TypeError):
            return None
    return None


def bootstrap_correlation_ci(x, y, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for Pearson correlation."""
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    if len(x_valid) < 10:
        return np.nan, np.nan, np.nan

    correlations = []
    n = len(x_valid)

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        r, _ = stats.pearsonr(x_valid[indices], y_valid[indices])
        correlations.append(r)

    correlations = np.array(correlations)
    alpha = 1 - ci
    lower = np.percentile(correlations, alpha / 2 * 100)
    upper = np.percentile(correlations, (1 - alpha / 2) * 100)

    return np.mean(correlations), lower, upper


def compute_variance_metrics(human_vals, silicon_vals):
    """Compute variance compression metrics."""
    human_var = np.nanvar(human_vals)
    silicon_var = np.nanvar(silicon_vals)

    var_ratio = silicon_var / human_var if human_var > 0 else np.nan

    # Tail mass (|z| > 1.5)
    h_std = np.nanstd(human_vals)
    s_std = np.nanstd(silicon_vals)

    if h_std > 0:
        human_z = (human_vals - np.nanmean(human_vals)) / h_std
        human_tail = np.nanmean(np.abs(human_z) > 1.5)
    else:
        human_tail = np.nan

    if s_std > 0:
        silicon_z = (silicon_vals - np.nanmean(silicon_vals)) / s_std
        silicon_tail = np.nanmean(np.abs(silicon_z) > 1.5)
    else:
        silicon_tail = np.nan

    return {
        'var_ratio': var_ratio,
        'human_var': human_var,
        'silicon_var': silicon_var,
        'human_mean': np.nanmean(human_vals),
        'silicon_mean': np.nanmean(silicon_vals),
        'human_tail_mass': human_tail,
        'silicon_tail_mass': silicon_tail
    }


def compute_weighted_stats(df, var_name, weight_col='pspwght'):
    """Compute weighted mean and SD for a variable.

    weight_col=None gives the unweighted mean and SD.  # v12
    """
    values = pd.to_numeric(df[var_name], errors='coerce')

    if weight_col is not None and weight_col in df.columns:
        weights = pd.to_numeric(df[weight_col], errors='coerce').fillna(1)
    else:
        weights = pd.Series(np.ones(len(df)), index=df.index)

    valid_mask = values.notna() & weights.notna()
    values = values[valid_mask]
    weights = weights[valid_mask]

    if len(values) == 0:
        return np.nan, np.nan

    weighted_mean = np.average(values, weights=weights)
    weighted_var = np.average((values - weighted_mean) ** 2, weights=weights)
    weighted_sd = np.sqrt(weighted_var)

    return weighted_mean, weighted_sd


def find_similar_respondents(target_idx, df, n_similar=3, exclude_self=True):
    """Find n most similar respondents based on demographics."""
    sim_cols = [c for c in SIMILARITY_VARS if c in df.columns]

    if not sim_cols:
        candidates = [i for i in range(len(df)) if i != target_idx]
        return random.sample(candidates, min(n_similar, len(candidates)))

    features = df[sim_cols].copy()

    for col in sim_cols:
        if col == 'cntry':
            # Encode country as numeric
            features[col] = pd.Categorical(features[col]).codes
        else:
            features[col] = pd.to_numeric(features[col], errors='coerce')
        features[col] = features[col].fillna(features[col].median())

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    target_features = features_scaled[target_idx].reshape(1, -1)
    distances = euclidean_distances(target_features, features_scaled)[0]

    sorted_indices = np.argsort(distances)

    similar_indices = []
    for idx in sorted_indices:
        if exclude_self and idx == target_idx:
            continue
        similar_indices.append(idx)
        if len(similar_indices) >= n_similar:
            break

    return similar_indices


# ============================================================================
# VLLM GENERATION (Optimized)
# ============================================================================

# Module-level model cache to avoid reloading between calls
_VLLM_CACHE = {'model_name': None, 'llm': None}


def generate_with_vllm(prompts, model_name='qwen', temperature=0.7, max_tokens=16, gen_seed=None, max_model_len=2048):  # v11
    """
    Generate responses using VLLM.

    Optimizations (v5):
      - temperature=0.7 (stochastic sampling)
      - max_tokens=16 (output is a single number, 1-2 tokens; 16 is safe margin)
      - max_model_len: 2048 by default since v11 (legacy runs used 1024)
      - gpu_memory_utilization=0.92 (inference-only, safe to push higher)
      - Model caching: reuses loaded model across multiple calls within same process
    """
    from vllm import LLM, SamplingParams

    model_path = MODELS.get(model_name, MODELS['qwen'])

    # Reuse model if already loaded (avoids ~30s reload in ablation, mechanisms, etc.)
    logging.info(f"Max prompt length: {max(len(p) for p in prompts)} chars")  # v10 CHANGE (H3)
    cache_key = (model_name, gen_seed, max_model_len)                           # v11
    if _VLLM_CACHE['model_name'] == cache_key and _VLLM_CACHE['llm'] is not None:
        llm = _VLLM_CACHE['llm']
        logging.info(f"Reusing cached model: {model_path}")
    else:
        # Clear previous model from GPU if switching models (e.g., rq8 block)
        if _VLLM_CACHE['llm'] is not None:
            logging.info(f"Clearing cached model: {_VLLM_CACHE['model_name']}")
            del _VLLM_CACHE['llm']
            _VLLM_CACHE['llm'] = None
            _VLLM_CACHE['model_name'] = None
            import gc; gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass

        logging.info(f"Loading model: {model_path}")
        # v12 FIX (P0-C): do NOT pass seed=None explicitly. Some vLLM releases
        # default LLM.seed to 0 rather than None, so passing None is not
        # guaranteed to reproduce a legacy run that omitted the argument. An
        # engine-seed mismatch would surface as a reproduction-gate failure and
        # be misattributed to max_model_len.
        _llm_kwargs = dict(model=model_path, trust_remote_code=True,
                           max_model_len=max_model_len,   # v11: configurable; legacy = 1024
                           gpu_memory_utilization=0.92)   # v5: was 0.85; safe for inference
        if gen_seed is not None:
            _llm_kwargs['seed'] = gen_seed
        logging.info(f"vLLM engine kwargs: "
                     f"{ {k: v for k, v in _llm_kwargs.items() if k != 'model'} }")
        llm = LLM(**_llm_kwargs)
        _VLLM_CACHE['model_name'] = cache_key
        _VLLM_CACHE['llm'] = llm

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,            # v5: was 150; output is just a number
        top_p=1.0 if temperature == 0 else 0.95  # disable nucleus sampling at temp=0
    )

    # v12: max_model_len is a TOKEN budget, but the only length ever logged was
    # in characters. Anchored scale cards lengthen prompts; log token lengths so
    # silent truncation cannot pass unnoticed.
    try:
        _tok = llm.get_tokenizer()
        _sample = prompts[::max(1, len(prompts) // 500)]
        _lens = [len(_tok(p)['input_ids']) for _sample_i, p in enumerate(_sample)]
        logging.info(f"Prompt tokens (n={len(_lens)} sample): max={max(_lens)} "
                     f"p99={int(np.percentile(_lens, 99))} budget={max_model_len}")
        if max(_lens) > 0.8 * max_model_len:
            logging.warning("Prompt tokens exceed 80% of max_model_len; raise it.")
    except Exception as _e:
        logging.warning(f"Token-length diagnostic skipped: {_e}")

    logging.info(f"Generating {len(prompts)} responses...")
    start_time = time.time()

    outputs = llm.generate(prompts, sampling_params)

    elapsed = time.time() - start_time
    logging.info(f"Generated {len(outputs)} responses in {elapsed:.1f}s "
                 f"({len(outputs) / elapsed:.1f} resp/sec)")

    responses = [output.outputs[0].text.strip() for output in outputs]
    return responses


# ============================================================================
# BLOCK: SOCIAL TRUST (Core — run first)
# ============================================================================

def run_social_trust(df, model_name='qwen', seed=888, prompt_mode='1p', temperature=0.7):
    """
    Run social trust analysis: individual-level recovery + country-level scatter.

    This is the FIRST block to run per supervisor's direction.
    Uses the Breyer Social Trust Scale (ppltrst, pplfair, pplhlp).

    Outputs:
      - Individual-level correlation (r) per variable per country
      - Country-level scatter data: survey mean vs silicon mean
      - Variance compression per variable per country
    """
    # Tag for output filenames: e.g., "qwen_1p" or "llama_3p"
    tag = f"{model_name}_{prompt_mode}"

    logger = setup_logging('social_trust', tag)
    logger.info(f"=== SOCIAL TRUST ANALYSIS (Model: {model_name}, Prompt: {prompt_mode}, Temp: {temperature}) ===")
    logger.info(f"Countries: {sorted(df['cntry'].unique())}")
    logger.info(f"Sample size: {len(df)}")

    random.seed(seed)
    np.random.seed(seed)

    trust_vars = SOCIAL_TRUST_VARIABLES
    prompt_template = select_prompt_template(prompt_mode)

    # ---- Step 1: Generate silicon responses ----
    prompts = []
    prompt_metadata = []

    for var_name, var_info in trust_vars.items():
        for idx, row in df.iterrows():
            backstory = row['backstory']

            prompt = prompt_template.format(
                backstory=backstory,
                question=var_info['question'],
                scale=var_info['scale']
            )

            prompts.append(prompt)
            prompt_metadata.append({
                'idno': row['idno'],
                'cntry': row['cntry'],
                'variable': var_name,
                'human_response': row.get(var_name, np.nan),
                'domain': var_info['domain'],
                'scale': var_info['scale']  # FIX: pass scale for validation
            })

    logger.info(f"Total prompts for social trust: {len(prompts)}")
    logger.info(f"  ({len(trust_vars)} variables × {len(df)} respondents)")

    # Generate responses
    responses = generate_with_vllm(prompts, model_name=model_name, temperature=temperature)

    # Parse
    results_data = []
    for meta, response in zip(prompt_metadata, responses):
        silicon_val = parse_response(response, scale_str=meta['scale'])  # FIX: validate against scale
        results_data.append({
            **meta,
            'silicon_response': silicon_val,
            'raw_response': response,
            'prompt_mode': prompt_mode
        })

    results_df = pd.DataFrame(results_data)

    # Save raw data
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_file = os.path.join(RESULTS_DIR, f"silicon_trust_raw_{tag}_seed{seed}.csv")
    results_df.to_csv(raw_file, index=False)
    logger.info(f"Raw data saved to {raw_file}")

    # ---- Step 2: Individual-level analysis (per variable, per country) ----
    indiv_results = []

    for var_name in trust_vars:
        # Overall (pooled)
        var_data = results_df[results_df['variable'] == var_name]
        human = pd.to_numeric(var_data['human_response'], errors='coerce').values
        silicon = pd.to_numeric(var_data['silicon_response'], errors='coerce').values
        valid = ~(np.isnan(human) | np.isnan(silicon))

        if valid.sum() >= 10:
            r_p, p_p = stats.pearsonr(human[valid], silicon[valid])
            r_s, p_s = stats.spearmanr(human[valid], silicon[valid])
            r_mean, r_lo, r_hi = bootstrap_correlation_ci(human, silicon)
            metrics = compute_variance_metrics(human, silicon)

            indiv_results.append({
                'variable': var_name,
                'cntry': 'ALL',
                'r_pearson': r_p, 'p_pearson': p_p,
                'r_spearman': r_s, 'p_spearman': p_s,
                'r_boot_mean': r_mean, 'r_ci_lo': r_lo, 'r_ci_hi': r_hi,
                'var_ratio': metrics['var_ratio'],
                'human_mean': metrics['human_mean'],
                'silicon_mean': metrics['silicon_mean'],
                'n_valid': valid.sum(),
                'model': model_name
            })

        # Per country
        for cntry in sorted(df['cntry'].unique()):
            cntry_data = var_data[var_data['cntry'] == cntry]
            human_c = pd.to_numeric(cntry_data['human_response'], errors='coerce').values
            silicon_c = pd.to_numeric(cntry_data['silicon_response'], errors='coerce').values
            valid_c = ~(np.isnan(human_c) | np.isnan(silicon_c))

            if valid_c.sum() < 10:
                continue

            r_p, p_p = stats.pearsonr(human_c[valid_c], silicon_c[valid_c])
            r_s, p_s = stats.spearmanr(human_c[valid_c], silicon_c[valid_c])
            metrics = compute_variance_metrics(human_c, silicon_c)

            indiv_results.append({
                'variable': var_name,
                'cntry': cntry,
                'r_pearson': r_p, 'p_pearson': p_p,
                'r_spearman': r_s, 'p_spearman': p_s,
                'r_boot_mean': np.nan, 'r_ci_lo': np.nan, 'r_ci_hi': np.nan,
                'var_ratio': metrics['var_ratio'],
                'human_mean': metrics['human_mean'],
                'silicon_mean': metrics['silicon_mean'],
                'n_valid': valid_c.sum(),
                'model': model_name
            })

    indiv_df = pd.DataFrame(indiv_results)
    indiv_file = os.path.join(RESULTS_DIR, f"silicon_trust_individual_{tag}.csv")
    indiv_df.to_csv(indiv_file, index=False)
    logger.info(f"Individual-level results saved to {indiv_file}")

    # ---- Step 3: Country-level scatter data ----
    # For each variable: survey mean per country vs silicon mean per country
    scatter_results = []

    for var_name in trust_vars:
        var_data = results_df[results_df['variable'] == var_name]

        for cntry in sorted(df['cntry'].unique()):
            cntry_data = var_data[var_data['cntry'] == cntry]

            human_vals = pd.to_numeric(cntry_data['human_response'], errors='coerce')
            silicon_vals = pd.to_numeric(cntry_data['silicon_response'], errors='coerce')

            # Weighted survey mean
            cntry_df = df[df['cntry'] == cntry]
            # v12: the human benchmark is DESIGN-WEIGHTED by default (pspwght),
            # which is what every released run did; it was previously undocumented
            # and is described the wrong way round in the replication package.
            # The silicon side is necessarily unweighted. Recorded in the manifest.
            survey_mean, survey_sd = compute_weighted_stats(
                cntry_df, var_name,
                weight_col=(None if human_weight == 'none' else human_weight))

            # Silicon mean (unweighted, as each respondent got one prompt)
            silicon_mean = silicon_vals.mean()
            silicon_sd = silicon_vals.std()

            scatter_results.append({
                'variable': var_name,
                'cntry': cntry,
                'country_name': COUNTRY_NAMES.get(cntry, cntry),
                'survey_mean': survey_mean,
                'survey_sd': survey_sd,
                'silicon_mean': silicon_mean,
                'silicon_sd': silicon_sd,
                'n_survey': human_vals.notna().sum(),
                'n_silicon': silicon_vals.notna().sum(),
                'model': model_name
            })

    scatter_df = pd.DataFrame(scatter_results)
    scatter_file = os.path.join(RESULTS_DIR, f"silicon_trust_country_scatter_{tag}.csv")
    scatter_df.to_csv(scatter_file, index=False)
    logger.info(f"Country-level scatter data saved to {scatter_file}")

    # ---- Step 4: Composite social trust (average of 3 items) ----
    # Compute per-respondent composite trust score (0-10 average)
    composite_results = []

    for cntry in sorted(df['cntry'].unique()):
        cntry_resp_data = results_df[results_df['cntry'] == cntry].copy()

        # Pivot to get one row per respondent
        pivot = cntry_resp_data.pivot_table(
            index='idno',
            columns='variable',
            values=['human_response', 'silicon_response'],
            aggfunc='first'
        )

        trust_items = list(trust_vars.keys())
        human_cols = [('human_response', v) for v in trust_items]
        silicon_cols = [('silicon_response', v) for v in trust_items]

        # Compute composite (average across 3 items)
        human_composite = pivot[human_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
        silicon_composite = pivot[silicon_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)

        # Weighted survey composite mean
        cntry_df = df[df['cntry'] == cntry]
        trust_vals_survey = cntry_df[trust_items].apply(pd.to_numeric, errors='coerce').mean(axis=1)
        if 'pspwght' in cntry_df.columns:
            w = pd.to_numeric(cntry_df['pspwght'], errors='coerce').fillna(1)
            survey_composite_mean = np.average(trust_vals_survey.dropna(), weights=w[trust_vals_survey.notna()])
        else:
            survey_composite_mean = trust_vals_survey.mean()

        composite_results.append({
            'cntry': cntry,
            'country_name': COUNTRY_NAMES.get(cntry, cntry),
            'survey_composite_mean': survey_composite_mean,
            'silicon_composite_mean': silicon_composite.mean(),
            'survey_composite_sd': trust_vals_survey.std(),
            'silicon_composite_sd': silicon_composite.std(),
            'n': len(human_composite.dropna()),
            'model': model_name
        })

    composite_df = pd.DataFrame(composite_results)
    composite_file = os.path.join(RESULTS_DIR, f"silicon_trust_composite_scatter_{tag}.csv")
    composite_df.to_csv(composite_file, index=False)
    logger.info(f"Composite trust scatter saved to {composite_file}")

    # ---- Summary ----
    logger.info("=" * 70)
    logger.info("SOCIAL TRUST ANALYSIS SUMMARY")
    logger.info("=" * 70)

    pooled = indiv_df[indiv_df['cntry'] == 'ALL']
    for _, row in pooled.iterrows():
        logger.info(f"  {row['variable']}: r={row['r_pearson']:.3f}, "
                     f"VR={row['var_ratio']:.3f}, n={row['n_valid']}")

    # Country-level correlation (survey mean vs silicon mean)
    for var_name in trust_vars:
        var_scatter = scatter_df[scatter_df['variable'] == var_name].dropna(
            subset=['survey_mean', 'silicon_mean'])
        if len(var_scatter) >= 3:
            r_country, p_country = stats.pearsonr(
                var_scatter['survey_mean'], var_scatter['silicon_mean'])
            logger.info(f"  {var_name} country-level r: {r_country:.3f} (p={p_country:.4f}), "
                        f"n_countries={len(var_scatter)}")

    # Composite
    comp_valid = composite_df.dropna(subset=['survey_composite_mean', 'silicon_composite_mean'])
    if len(comp_valid) >= 3:
        r_comp, p_comp = stats.pearsonr(
            comp_valid['survey_composite_mean'], comp_valid['silicon_composite_mean'])
        logger.info(f"  COMPOSITE country-level r: {r_comp:.3f} (p={p_comp:.4f})")

    logger.info("=" * 70)

    return results_df, indiv_df, scatter_df, composite_df


# ============================================================================
# BLOCK: FULL VARIABLE SET (All 42 ESS11 variables)
# ============================================================================

def run_full_variables(df, model_name='qwen', seed=888, prompt_mode='1p', temperature=0.7, backstory_tag='', gen_seed=None, max_model_len=2048, human_weight='pspwght'):  # v12
    """
    Run full variable set analysis (all 42 ESS11 variables).
    Individual-level recovery + variance compression + country-level aggregation.
    """
    tag = f"{model_name}_{prompt_mode}" + (f"_{backstory_tag}" if backstory_tag else "")
    logger = setup_logging('full_vars', tag)
    logger.info(f"=== FULL VARIABLE SET (Model: {model_name}, Prompt: {prompt_mode}, Temp: {temperature}) ===")
    logger.info(f"Variables: {len(ESS_VARIABLES)}")
    logger.info(f"Sample: {len(df)} respondents, {df['cntry'].nunique()} countries")

    random.seed(seed)
    np.random.seed(seed)

    prompt_template = select_prompt_template(prompt_mode)

    # Prepare all prompts
    prompts = []
    prompt_metadata = []

    for var_name, var_info in ESS_VARIABLES.items():
        question_text = var_info.get('question', var_info['label'])
        for idx, row in df.iterrows():
            prompt = prompt_template.format(
                backstory=row['backstory'],
                question=question_text,
                scale=var_info['scale']
            )
            prompts.append(prompt)
            prompt_metadata.append({
                'idno': row['idno'],
                'cntry': row['cntry'],
                'variable': var_name,
                'human_response': row.get(var_name, np.nan),
                'domain': var_info['domain'],
                'scale': var_info['scale']  # FIX: pass scale for validation
            })

    logger.info(f"Total prompts: {len(prompts)}")

    # Generate
    logging.info(f"vLLM engine seed: {gen_seed}")  # v11 FIX: make the effective seed auditable in logs
    responses = generate_with_vllm(prompts, model_name=model_name, temperature=temperature,
                                   gen_seed=gen_seed, max_model_len=max_model_len)  # v11 FIX (P0): was accepted but never forwarded

    results_data = []
    for meta, response in zip(prompt_metadata, responses):
        results_data.append({
            **meta,
            'silicon_response': parse_response(response, scale_str=meta['scale']),  # FIX: validate against scale
            'raw_response': response
        })

    results_df = pd.DataFrame(results_data)
    raw_file = os.path.join(RESULTS_DIR, f"silicon_full_raw_{tag}_seed{seed}.csv")
    results_df.to_csv(raw_file, index=False)
    logger.info(f"Raw data saved to {raw_file}")

    # ---- Individual-level analysis ----
    rq1_results = []
    for var_name in ESS_VARIABLES:
        var_data = results_df[results_df['variable'] == var_name]
        human = pd.to_numeric(var_data['human_response'], errors='coerce').values
        silicon = pd.to_numeric(var_data['silicon_response'], errors='coerce').values
        valid = ~(np.isnan(human) | np.isnan(silicon))

        if valid.sum() < 10:
            continue

        r_p, p_p = stats.pearsonr(human[valid], silicon[valid])
        r_s, p_s = stats.spearmanr(human[valid], silicon[valid])
        r_mean, r_lo, r_hi = bootstrap_correlation_ci(human, silicon)
        metrics = compute_variance_metrics(human, silicon)

        rq1_results.append({
            'variable': var_name,
            'domain': ESS_VARIABLES[var_name]['domain'],
            'r_pearson': r_p, 'p_pearson': p_p,
            'r_spearman': r_s, 'p_spearman': p_s,
            'r_boot_mean': r_mean, 'r_ci_lo': r_lo, 'r_ci_hi': r_hi,
            'var_ratio': metrics['var_ratio'],
            'human_mean': metrics['human_mean'],
            'silicon_mean': metrics['silicon_mean'],
            'n_valid': valid.sum(),
            'model': model_name
        })

    rq1_df = pd.DataFrame(rq1_results)
    rq1_df.to_csv(os.path.join(RESULTS_DIR, f"silicon_full_rq1_{tag}.csv"), index=False)

    # ---- Country-level scatter (all variables) ----
    scatter_results = []
    for var_name in ESS_VARIABLES:
        var_data = results_df[results_df['variable'] == var_name]
        for cntry in sorted(df['cntry'].unique()):
            cntry_data = var_data[var_data['cntry'] == cntry]
            cntry_df = df[df['cntry'] == cntry]

            survey_mean, survey_sd = compute_weighted_stats(cntry_df, var_name)
            silicon_vals = pd.to_numeric(cntry_data['silicon_response'], errors='coerce')

            scatter_results.append({
                'variable': var_name,
                'domain': ESS_VARIABLES[var_name]['domain'],
                'cntry': cntry,
                'country_name': COUNTRY_NAMES.get(cntry, cntry),
                'survey_mean': survey_mean,
                'survey_sd': survey_sd,
                'silicon_mean': silicon_vals.mean(),
                'silicon_sd': silicon_vals.std(),
                'model': model_name
            })

    scatter_df = pd.DataFrame(scatter_results)
    scatter_df.to_csv(os.path.join(RESULTS_DIR, f"silicon_full_country_scatter_{tag}.csv"),
                      index=False)

    # ---- v12: per-country individual-level correlations (r_wc) ----
    # Previously produced only by the separate replication script compute_rq3.py,
    # which was easy to forget and reintroduced the pre-v12 missingness rule.
    # Schema is identical to that script's output so downstream code is unchanged.
    rq3_results = []
    for var_name in ESS_VARIABLES:
        var_data = results_df[results_df['variable'] == var_name]
        for cntry in sorted(df['cntry'].unique()):
            cd = var_data[var_data['cntry'] == cntry]
            h = pd.to_numeric(cd['human_response'], errors='coerce').values
            s_ = pd.to_numeric(cd['silicon_response'], errors='coerce').values
            ok = ~(np.isnan(h) | np.isnan(s_))
            n_ok = int(ok.sum())
            if n_ok < 10:
                r_p = p_p = r_s = p_s = np.nan
            else:
                r_p, p_p = stats.pearsonr(h[ok], s_[ok])
                r_s, p_s = stats.spearmanr(h[ok], s_[ok])
            rq3_results.append({
                'variable': var_name,
                'domain': ESS_VARIABLES[var_name]['domain'],
                'cntry': cntry,
                'r_pearson': r_p, 'p_pearson': p_p,
                'r_spearman': r_s, 'p_spearman': p_s,
                'human_mean': np.nanmean(h[ok]) if n_ok else np.nan,
                'silicon_mean': np.nanmean(s_[ok]) if n_ok else np.nan,
                'human_sd': np.nanstd(h[ok], ddof=1) if n_ok > 1 else np.nan,
                'silicon_sd': np.nanstd(s_[ok], ddof=1) if n_ok > 1 else np.nan,
                'n_valid': n_ok,
            })
    pd.DataFrame(rq3_results).to_csv(
        os.path.join(RESULTS_DIR, f"silicon_full_rq3_{tag}.csv"), index=False)

    # Summary
    logger.info("=" * 70)
    logger.info("FULL VARIABLE ANALYSIS SUMMARY")
    logger.info("=" * 70)
    # v12 FIX: at very small sample_per_country (e.g. a pilot run), every item
    # can fail the "valid.sum() < 10" individual-level gate above, leaving
    # rq1_df with zero rows AND zero columns. rq1_df['r_pearson'] then raises
    # KeyError rather than returning NaN, which crashed this summary print
    # after the raw, scatter and rq3 files had already been written correctly.
    # This never triggers at the real sample_per_country=500 (500 respondents
    # per country is far over the threshold for every item); it only affects
    # small diagnostic runs, so it is guarded rather than the threshold changed.
    if len(rq1_df) and "r_pearson" in rq1_df.columns:
        logger.info(f"Mean Pearson r:  {rq1_df['r_pearson'].mean():.3f}")
        logger.info(f"Mean VR:         {rq1_df['var_ratio'].mean():.3f}")
    else:
        logger.warning("rq1_df is empty (every item had fewer than 10 valid "
                       "human-silicon pairs — expected only at small "
                       "sample_per_country). Raw, scatter and rq3 files were "
                       "still written; skipping the per-item summary.")

    if len(rq1_df) and "domain" in rq1_df.columns:
        for domain in rq1_df['domain'].unique():
            domain_data = rq1_df[rq1_df['domain'] == domain]
            logger.info(f"  {domain}: mean r={domain_data['r_pearson'].mean():.3f}, "
                         f"mean VR={domain_data['var_ratio'].mean():.3f}")

    # Country-level correlation across all variables
    scatter_valid = scatter_df.dropna(subset=['survey_mean', 'silicon_mean'])
    if len(scatter_valid) >= 10:
        r_all, p_all = stats.pearsonr(scatter_valid['survey_mean'],
                                       scatter_valid['silicon_mean'])
        logger.info(f"Overall country-level r (all vars): {r_all:.3f} (p={p_all:.6f})")
    logger.info("=" * 70)

    return results_df, rq1_df, scatter_df


# ============================================================================
# BLOCK: MECHANISMS (Prompt framing RQ3, Format sensitivity RQ6)
# ============================================================================

def run_mechanisms(df, model_name='qwen', seed=888):
    """
    Run mechanism tests on social trust variables.
    RQ3: Prompt framing (1P standard vs 3P vs idiosyncratic)
    RQ6: Format sensitivity (standard vs reversed scale)
    """
    logger = setup_logging('mechanisms', model_name)
    logger.info(f"=== MECHANISM TESTS (Model: {model_name}) ===")

    random.seed(seed)
    np.random.seed(seed)

    trust_vars = SOCIAL_TRUST_VARIABLES
    all_results = []

    # ---- RQ3: Prompt framing ----
    logger.info("Running RQ3: Prompt framing on social trust...")
    prompt_conditions = {
        '1p_standard': PROMPT_1P_STANDARD,
        '3p': PROMPT_3P,
        '1p_idiosyncratic': PROMPT_1P_IDIOSYNCRATIC
    }

    for condition, template in prompt_conditions.items():
        prompts = []
        metadata = []

        for var_name, var_info in trust_vars.items():
            question_text = var_info.get('question', var_info['label'])
            for idx, row in df.iterrows():
                prompt = template.format(
                    backstory=row['backstory'],
                    question=question_text,
                    scale=var_info['scale']
                )
                prompts.append(prompt)
                metadata.append({
                    'idno': row['idno'],
                    'cntry': row['cntry'],
                    'variable': var_name,
                    'human_response': row.get(var_name, np.nan),
                    'condition': condition,
                    'rq': 'RQ3'
                })

        responses = generate_with_vllm(prompts, model_name=model_name)

        for meta, response in zip(metadata, responses):
            all_results.append({
                **meta,
                'silicon_response': parse_response(response),
                'model': model_name
            })

    # ---- RQ6: Format sensitivity ----
    logger.info("Running RQ6: Format sensitivity on social trust...")
    for var_name, var_info in trust_vars.items():
        prompts_std = []
        prompts_rev = []
        metadata_pairs = []

        question_text = var_info.get('question', var_info['label'])

        for idx, row in df.iterrows():
            prompt_std = PROMPT_1P_STANDARD.format(
                backstory=row['backstory'],
                question=question_text,
                scale=var_info['scale']
            )

            scale_parts = var_info['scale'].split('-')
            if len(scale_parts) == 2:
                rev_scale = f"{scale_parts[1]}-{scale_parts[0]}"
            else:
                rev_scale = var_info['scale']

            prompt_rev = PROMPT_1P_STANDARD.format(
                backstory=row['backstory'],
                question=question_text + " (reversed scale)",
                scale=rev_scale
            )

            prompts_std.append(prompt_std)
            prompts_rev.append(prompt_rev)
            metadata_pairs.append({
                'idno': row['idno'],
                'cntry': row['cntry'],
                'variable': var_name,
                'human_response': row.get(var_name, np.nan)
            })

        responses_std = generate_with_vllm(prompts_std, model_name=model_name)
        responses_rev = generate_with_vllm(prompts_rev, model_name=model_name)

        for meta, resp_std, resp_rev in zip(metadata_pairs, responses_std, responses_rev):
            all_results.append({
                **meta, 'silicon_response': parse_response(resp_std),
                'condition': 'standard', 'rq': 'RQ6', 'model': model_name
            })
            all_results.append({
                **meta, 'silicon_response': parse_response(resp_rev),
                'condition': 'reversed', 'rq': 'RQ6', 'model': model_name
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, f"silicon_mechanisms_data_{model_name}.csv"),
                      index=False)

    # ---- Analyze RQ3 ----
    rq3_results = []
    for condition in prompt_conditions:
        for var_name in trust_vars:
            cond_var = results_df[
                (results_df['rq'] == 'RQ3') &
                (results_df['condition'] == condition) &
                (results_df['variable'] == var_name)
            ]
            human = pd.to_numeric(cond_var['human_response'], errors='coerce').values
            silicon = pd.to_numeric(cond_var['silicon_response'], errors='coerce').values
            valid = ~(np.isnan(human) | np.isnan(silicon))

            if valid.sum() < 10:
                continue

            r, p = stats.pearsonr(human[valid], silicon[valid])
            metrics = compute_variance_metrics(human, silicon)

            rq3_results.append({
                'variable': var_name, 'condition': condition,
                'r_pearson': r, 'p_pearson': p,
                'var_ratio': metrics['var_ratio'],
                'model': model_name
            })

    rq3_df = pd.DataFrame(rq3_results)
    rq3_df.to_csv(os.path.join(RESULTS_DIR, f"silicon_mechanisms_rq3_{model_name}.csv"), index=False)

    # ---- Analyze RQ6 ----
    rq6_results = []
    rq6_data = results_df[results_df['rq'] == 'RQ6']
    for var_name in trust_vars:
        std = rq6_data[(rq6_data['variable'] == var_name) & (rq6_data['condition'] == 'standard')]
        rev = rq6_data[(rq6_data['variable'] == var_name) & (rq6_data['condition'] == 'reversed')]

        if len(std) == 0 or len(rev) == 0:
            continue

        merged = std.merge(rev, on=['idno', 'cntry'], suffixes=('_std', '_rev'))
        std_vals = pd.to_numeric(merged['silicon_response_std'], errors='coerce').values
        rev_vals = pd.to_numeric(merged['silicon_response_rev'], errors='coerce').values
        valid = ~(np.isnan(std_vals) | np.isnan(rev_vals))

        if valid.sum() < 10:
            continue

        flip_rate = np.mean(std_vals[valid] != rev_vals[valid])
        r_formats, _ = stats.pearsonr(std_vals[valid], rev_vals[valid])

        rq6_results.append({
            'variable': var_name, 'flip_rate': flip_rate,
            'r_between_formats': r_formats, 'n_valid': valid.sum(),
            'model': model_name
        })

    rq6_df = pd.DataFrame(rq6_results)
    rq6_df.to_csv(os.path.join(RESULTS_DIR, f"silicon_mechanisms_rq6_{model_name}.csv"), index=False)

    logger.info("Mechanism tests complete")
    return results_df, rq3_df, rq6_df


# ============================================================================
# BLOCK: FEW-SHOT ANCHORING (RQ11)
# ============================================================================

def run_fewshot(df, model_name='qwen', n_similar=3, seed=888):
    """Run few-shot anchoring on social trust variables."""
    logger = setup_logging('fewshot', model_name)
    logger.info(f"=== FEW-SHOT ANCHORING (Model: {model_name}, n_similar={n_similar}) ===")

    random.seed(seed)
    np.random.seed(seed)

    trust_vars = SOCIAL_TRUST_VARIABLES
    results = []

    for var_name, var_info in trust_vars.items():
        logger.info(f"Processing: {var_name}")
        question_text = var_info.get('question', var_info['label'])

        prompts_zero = []
        prompts_few = []
        metadata = []

        for target_idx in range(len(df)):
            row = df.iloc[target_idx]

            # Zero-shot
            prompt_z = PROMPT_1P_STANDARD.format(
                backstory=row['backstory'],
                question=question_text,
                scale=var_info['scale']
            )

            # Few-shot examples
            similar_indices = find_similar_respondents(target_idx, df, n_similar=n_similar)
            examples_text = ""
            for i, sim_idx in enumerate(similar_indices):
                sim_row = df.iloc[sim_idx]
                sim_response = sim_row.get(var_name, np.nan)
                if pd.notna(sim_response):
                    examples_text += (f"\n[Person {i + 1}]\n"
                                      f"{sim_row['backstory'][:500]}...\n"
                                      f"Their response: {int(sim_response)}\n")

            prompt_f = PROMPT_FEWSHOT.format(
                examples=examples_text,
                backstory=row['backstory'],
                question=question_text,
                scale=var_info['scale']
            )

            prompts_zero.append(prompt_z)
            prompts_few.append(prompt_f)
            metadata.append({
                'idno': row['idno'],
                'cntry': row['cntry'],
                'variable': var_name,
                'human_response': row.get(var_name, np.nan),
            })

        responses_zero = generate_with_vllm(prompts_zero, model_name=model_name)
        responses_few = generate_with_vllm(prompts_few, model_name=model_name)

        for meta, rz, rf in zip(metadata, responses_zero, responses_few):
            results.append({
                **meta,
                'silicon_zeroshot': parse_response(rz),
                'silicon_fewshot': parse_response(rf),
                'model': model_name
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, f"silicon_fewshot_data_{model_name}.csv"),
                      index=False)

    # Analyze
    rq11_results = []
    for var_name in trust_vars:
        vd = results_df[results_df['variable'] == var_name]
        human = pd.to_numeric(vd['human_response'], errors='coerce').values
        sz = pd.to_numeric(vd['silicon_zeroshot'], errors='coerce').values
        sf = pd.to_numeric(vd['silicon_fewshot'], errors='coerce').values

        vz = ~(np.isnan(human) | np.isnan(sz))
        vf = ~(np.isnan(human) | np.isnan(sf))

        r_z = stats.pearsonr(human[vz], sz[vz])[0] if vz.sum() >= 10 else np.nan
        r_f = stats.pearsonr(human[vf], sf[vf])[0] if vf.sum() >= 10 else np.nan

        rq11_results.append({
            'variable': var_name,
            'r_zeroshot': r_z, 'r_fewshot': r_f,
            'r_improvement': r_f - r_z if not (np.isnan(r_f) or np.isnan(r_z)) else np.nan,
            'n_zero': vz.sum(), 'n_few': vf.sum(),
            'model': model_name
        })

    rq11_df = pd.DataFrame(rq11_results)
    rq11_df.to_csv(os.path.join(RESULTS_DIR, f"silicon_fewshot_rq11_{model_name}.csv"), index=False)

    logger.info("=" * 60)
    logger.info("FEW-SHOT SUMMARY")
    logger.info(f"Mean r zero-shot: {rq11_df['r_zeroshot'].mean():.3f}")
    logger.info(f"Mean r few-shot:  {rq11_df['r_fewshot'].mean():.3f}")
    logger.info(f"Mean improvement: {rq11_df['r_improvement'].mean():+.3f}")
    logger.info("=" * 60)

    return results_df, rq11_df


# ============================================================================
# BLOCK: DISTRIBUTION ANCHORING (RQ12) — now country-specific
# ============================================================================

def run_dist_anchoring(df, model_name='qwen', seed=888):
    """Run distribution anchoring with COUNTRY-SPECIFIC population statistics."""
    logger = setup_logging('dist_anchor', model_name)
    logger.info(f"=== DISTRIBUTION ANCHORING (Model: {model_name}) ===")

    random.seed(seed)
    np.random.seed(seed)

    trust_vars = SOCIAL_TRUST_VARIABLES
    results = []

    for var_name, var_info in trust_vars.items():
        logger.info(f"Processing: {var_name}")
        question_text = var_info.get('question', var_info['label'])

        prompts_base = []
        prompts_anchor = []
        metadata = []

        for idx, row in df.iterrows():
            cntry = row['cntry']
            country_name = COUNTRY_NAMES.get(cntry, cntry)

            # Country-specific population stats
            cntry_df = df[df['cntry'] == cntry]
            pop_mean, pop_sd = compute_weighted_stats(cntry_df, var_name)

            # Baseline
            prompt_b = PROMPT_1P_STANDARD.format(
                backstory=row['backstory'],
                question=question_text,
                scale=var_info['scale']
            )

            # Anchored (with country-specific stats)
            prompt_a = PROMPT_DIST_ANCHOR.format(
                country=country_name,
                question=question_text,
                pop_mean=pop_mean if not np.isnan(pop_mean) else 5.0,
                pop_sd=pop_sd if not np.isnan(pop_sd) else 2.5,
                scale=var_info['scale'],
                backstory=row['backstory']
            )

            prompts_base.append(prompt_b)
            prompts_anchor.append(prompt_a)
            metadata.append({
                'idno': row['idno'], 'cntry': cntry,
                'variable': var_name,
                'human_response': row.get(var_name, np.nan),
                'pop_mean': pop_mean, 'pop_sd': pop_sd
            })

        responses_b = generate_with_vllm(prompts_base, model_name=model_name)
        responses_a = generate_with_vllm(prompts_anchor, model_name=model_name)

        for meta, rb, ra in zip(metadata, responses_b, responses_a):
            results.append({
                **meta,
                'silicon_baseline': parse_response(rb),
                'silicon_anchored': parse_response(ra),
                'model': model_name
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, f"silicon_dist_anchor_data_{model_name}.csv"),
                      index=False)

    # Analyze
    rq12_results = []
    for var_name in trust_vars:
        vd = results_df[results_df['variable'] == var_name]
        human = pd.to_numeric(vd['human_response'], errors='coerce').values
        sb = pd.to_numeric(vd['silicon_baseline'], errors='coerce').values
        sa = pd.to_numeric(vd['silicon_anchored'], errors='coerce').values

        vb = ~(np.isnan(human) | np.isnan(sb))
        va = ~(np.isnan(human) | np.isnan(sa))

        r_b = stats.pearsonr(human[vb], sb[vb])[0] if vb.sum() >= 10 else np.nan
        r_a = stats.pearsonr(human[va], sa[va])[0] if va.sum() >= 10 else np.nan

        mb = compute_variance_metrics(human, sb)
        ma = compute_variance_metrics(human, sa)

        rq12_results.append({
            'variable': var_name,
            'r_baseline': r_b, 'r_anchored': r_a,
            'r_improvement': r_a - r_b if not (np.isnan(r_a) or np.isnan(r_b)) else np.nan,
            'vr_baseline': mb['var_ratio'], 'vr_anchored': ma['var_ratio'],
            'model': model_name
        })

    rq12_df = pd.DataFrame(rq12_results)
    rq12_df.to_csv(os.path.join(RESULTS_DIR, f"silicon_dist_anchor_rq12_{model_name}.csv"),
                    index=False)

    logger.info("=" * 60)
    logger.info("DISTRIBUTION ANCHORING SUMMARY")
    logger.info(f"Mean r baseline:  {rq12_df['r_baseline'].mean():.3f}")
    logger.info(f"Mean r anchored:  {rq12_df['r_anchored'].mean():.3f}")
    logger.info(f"Mean improvement: {rq12_df['r_improvement'].mean():+.3f}")
    logger.info("=" * 60)

    return results_df, rq12_df


# ============================================================================
# BLOCK: MODEL COMPARISON (RQ8)
# ============================================================================

def run_model_comparison(df, seed=888):
    """Run RQ8: Compare Qwen vs Llama on social trust."""
    logger = setup_logging('rq8', 'comparison')
    logger.info("=== RQ8: MODEL COMPARISON ===")

    # Run social trust with both models
    logger.info("Running social trust with Qwen...")
    run_social_trust(df, model_name='qwen', seed=seed)

    logger.info("Running social trust with Llama...")
    run_social_trust(df, model_name='llama', seed=seed)

    # Compare
    qwen = pd.read_csv(os.path.join(RESULTS_DIR, "silicon_trust_individual_qwen.csv"))
    llama = pd.read_csv(os.path.join(RESULTS_DIR, "silicon_trust_individual_llama.csv"))

    # Pooled comparison
    qwen_pooled = qwen[qwen['cntry'] == 'ALL'][['variable', 'r_pearson', 'var_ratio']]
    llama_pooled = llama[llama['cntry'] == 'ALL'][['variable', 'r_pearson', 'var_ratio']]

    comparison = qwen_pooled.merge(llama_pooled, on='variable', suffixes=('_qwen', '_llama'))
    comparison.to_csv(os.path.join(RESULTS_DIR, "silicon_rq8_comparison.csv"), index=False)

    logger.info("=" * 60)
    logger.info("RQ8 SUMMARY")
    logger.info(f"Qwen mean r:  {comparison['r_pearson_qwen'].mean():.3f}")
    logger.info(f"Llama mean r: {comparison['r_pearson_llama'].mean():.3f}")
    logger.info("=" * 60)

    return comparison


# ============================================================================
# BLOCK: DEEP ANALYSIS
# ============================================================================

def run_deep_analysis(df, model_name='qwen', seed=888):
    """Run deep statistical analysis on social trust results."""
    logger = setup_logging('deep', model_name)
    logger.info("=== DEEP STATISTICAL ANALYSIS ===")

    # Load social trust results — try tag-based names first, then legacy
    raw_file = None
    for candidate in [
        os.path.join(RESULTS_DIR, f"silicon_trust_raw_{model_name}_1p_seed{seed}.csv"),
        os.path.join(RESULTS_DIR, f"silicon_trust_raw_{model_name}_3p_seed{seed}.csv"),
        os.path.join(RESULTS_DIR, f"silicon_trust_raw_{model_name}_seed{seed}.csv"),
    ]:
        if os.path.exists(candidate):
            raw_file = candidate
            break
    if raw_file is None:
        logger.error(f"Social trust results not found for {model_name}. Run 'trust' block first.")
        return None

    results_df = pd.read_csv(raw_file)
    deep_results = []

    for var_name in SOCIAL_TRUST_VARIABLES:
        var_data = results_df[results_df['variable'] == var_name]
        human = pd.to_numeric(var_data['human_response'], errors='coerce').values
        silicon = pd.to_numeric(var_data['silicon_response'], errors='coerce').values
        valid = ~(np.isnan(human) | np.isnan(silicon))

        if valid.sum() < 30:
            continue

        hv = human[valid]
        sv = silicon[valid]

        r_p, _ = stats.pearsonr(hv, sv)
        mae = np.mean(np.abs(hv - sv))
        slope, intercept, _, _, _ = stats.linregress(hv, sv)

        n = len(hv)
        decile_size = max(n // 10, 5)

        h_top = set(np.argsort(hv)[-decile_size:])
        s_top = set(np.argsort(sv)[-decile_size:])
        top_recovery = len(h_top & s_top) / decile_size

        h_bot = set(np.argsort(hv)[:decile_size])
        s_bot = set(np.argsort(sv)[:decile_size])
        bot_recovery = len(h_bot & s_bot) / decile_size

        var_ratio = np.var(sv) / np.var(hv) if np.var(hv) > 0 else np.nan

        # KS test
        ks_stat, ks_p = stats.ks_2samp(hv, sv)

        deep_results.append({
            'variable': var_name,
            'n_valid': valid.sum(),
            'r_pearson': r_p,
            'mae': mae,
            'reg_slope': slope,
            'reg_intercept': intercept,
            'top_decile_recovery': top_recovery,
            'bottom_decile_recovery': bot_recovery,
            'var_ratio': var_ratio,
            'ks_stat': ks_stat,
            'ks_p': ks_p,
            'model': model_name
        })

    deep_df = pd.DataFrame(deep_results)
    deep_df.to_csv(os.path.join(RESULTS_DIR, f"silicon_deep_analysis_{model_name}.csv"), index=False)

    logger.info("=" * 60)
    logger.info("DEEP ANALYSIS SUMMARY")
    logger.info(f"Mean r:              {deep_df['r_pearson'].mean():.3f}")
    logger.info(f"Mean MAE:            {deep_df['mae'].mean():.2f}")
    logger.info(f"Mean Reg Slope:      {deep_df['reg_slope'].mean():.3f} (1.0=perfect)")
    logger.info(f"Mean Top Decile:     {deep_df['top_decile_recovery'].mean():.1%}")
    logger.info(f"Mean Bottom Decile:  {deep_df['bottom_decile_recovery'].mean():.1%}")
    logger.info(f"Mean VR:             {deep_df['var_ratio'].mean():.3f}")
    logger.info("=" * 60)

    return deep_df


# ============================================================================
# BLOCK: SUBGROUP HETEROGENEITY (RQ7 — adapted for multi-country)
# ============================================================================

def run_subgroup_analysis(df, model_name='qwen', seed=888):
    """Run subgroup heterogeneity analysis on social trust (multi-country)."""
    logger = setup_logging('subgroup', model_name)
    logger.info("=== SUBGROUP HETEROGENEITY (Model: {}) ===".format(model_name))

    # Load social trust results — try tag-based names first, then legacy
    raw_file = None
    for candidate in [
        os.path.join(RESULTS_DIR, f"silicon_trust_raw_{model_name}_1p_seed{seed}.csv"),
        os.path.join(RESULTS_DIR, f"silicon_trust_raw_{model_name}_3p_seed{seed}.csv"),
        os.path.join(RESULTS_DIR, f"silicon_trust_raw_{model_name}_seed{seed}.csv"),
    ]:
        if os.path.exists(candidate):
            raw_file = candidate
            break
    if raw_file is None:
        logger.error(f"Social trust results not found for {model_name}. Run 'trust' block first.")
        return None

    results_df = pd.read_csv(raw_file)

    # Define groups
    groups = {}
    if 'eduyrs' in df.columns:
        groups['low_education'] = df['eduyrs'] < df['eduyrs'].median()
    if 'hinctnta' in df.columns:
        groups['low_income'] = pd.to_numeric(df['hinctnta'], errors='coerce') <= 3
    if 'gndr' in df.columns:
        groups['female'] = df['gndr'] == 2
    if 'brncntr' in df.columns:
        groups['immigrant'] = df['brncntr'] == 2

    h7_results = []

    for group_name, group_mask in groups.items():
        if group_mask is None:
            continue

        for var_name in SOCIAL_TRUST_VARIABLES:
            var_data = results_df[results_df['variable'] == var_name].copy()
            var_data = var_data.merge(
                df[['idno']].assign(is_disadvantaged=group_mask.values),
                on='idno'
            )

            for is_disadv in [True, False]:
                sub = var_data[var_data['is_disadvantaged'] == is_disadv]
                human = pd.to_numeric(sub['human_response'], errors='coerce').values
                silicon = pd.to_numeric(sub['silicon_response'], errors='coerce').values
                valid = ~(np.isnan(human) | np.isnan(silicon))

                if valid.sum() < 20:
                    continue

                r, _ = stats.pearsonr(human[valid], silicon[valid])
                metrics = compute_variance_metrics(human, silicon)

                h7_results.append({
                    'variable': var_name,
                    'group': group_name,
                    'is_disadvantaged': is_disadv,
                    'r_pearson': r,
                    'var_ratio': metrics['var_ratio'],
                    'n': valid.sum(),
                    'model': model_name
                })

    h7_df = pd.DataFrame(h7_results)
    h7_df.to_csv(os.path.join(RESULTS_DIR, f"silicon_subgroup_{model_name}.csv"), index=False)

    logger.info("Subgroup analysis complete")
    return h7_df


# ============================================================================
# OVERTIME ANALYSIS (Multi-Round ESS)
# ============================================================================

# 15 countries present in all 6 rounds (R1, R3, R6, R7, R9, R11)
OVERTIME_PANEL_COUNTRIES = [
    'BE', 'CH', 'DE', 'ES', 'FI', 'FR', 'GB', 'HU',
    'IE', 'NL', 'NO', 'PL', 'PT', 'SE', 'SI'
]


def run_overtime_analysis(rounds_dir, model_name='qwen', countries=None,
                          sample_per_country=200, seed=888):
    """
    Run overtime analysis across multiple ESS rounds.

    For each round:
      1. Load data, harmonize variables, generate backstories
      2. Sample respondents per country
      3. Generate silicon responses via VLLM
      4. Compute country-level survey mean vs silicon mean

    Then test: does the silicon sample track temporal changes in trust?

    Parameters
    ----------
    rounds_dir : str
        Directory containing ESS round CSV files (ESS*.csv).
    model_name : str
        Model to use.
    countries : list of str, optional
        Countries to analyze. Defaults to the 15-country panel
        present in all 6 available rounds.
    sample_per_country : int
        Respondents per country per round.
    seed : int
        Random seed.
    """
    logger = setup_logging('overtime', model_name)
    logger.info("=== OVERTIME ANALYSIS ===")
    logger.info(f"Rounds directory: {rounds_dir}")

    import glob
    ess_files = sorted(glob.glob(os.path.join(rounds_dir, "ESS*.csv")))

    if not ess_files:
        logger.error(f"No ESS CSV files found in {rounds_dir}")
        logger.info("Expected files like ESS1e06_7.csv, ESS3e03_7.csv, etc.")
        logger.info("Unzip ESS files first: cd data/'ESS Data' && for f in *.zip; do unzip $f; done")
        return None

    logger.info(f"Found {len(ess_files)} ESS files: {[os.path.basename(f) for f in ess_files]}")

    # Default to 15-country panel
    if countries is None:
        countries = OVERTIME_PANEL_COUNTRIES
    countries_upper = [c.upper() for c in countries]
    logger.info(f"Countries: {countries_upper}")

    # ---- Step 1: Process each round ----
    all_results = []
    round_summaries = []

    for ess_file in ess_files:
        fname = os.path.basename(ess_file)
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {fname}")

        # Load and prepare data for this round
        round_df = load_data(
            ess_file=ess_file,
            countries=countries_upper,
            sample_per_country=sample_per_country,
            seed=seed
        )

        if len(round_df) == 0:
            logger.warning(f"  No data after filtering. Skipping.")
            continue

        ess_round = int(round_df['essround'].iloc[0]) if 'essround' in round_df.columns else 0
        survey_year = ESS_ROUND_YEAR.get(ess_round, 0)

        logger.info(f"  Round {ess_round} ({survey_year}): {len(round_df)} respondents, "
                     f"{round_df['cntry'].nunique()} countries")

        # Generate silicon responses for social trust variables
        for var_name, var_info in SOCIAL_TRUST_VARIABLES.items():
            question_text = var_info.get('question', var_info['label'])

            prompts = []
            metadata = []

            for idx, row in round_df.iterrows():
                prompt = PROMPT_1P_STANDARD.format(
                    backstory=row['backstory'],
                    question=question_text,
                    scale=var_info['scale']
                )
                prompts.append(prompt)
                metadata.append({
                    'idno': row['idno'],
                    'cntry': row['cntry'],
                    'variable': var_name,
                    'human_response': row.get(var_name, np.nan),
                    'essround': ess_round,
                    'survey_year': survey_year
                })

            logger.info(f"  {var_name}: {len(prompts)} prompts")
            responses = generate_with_vllm(prompts, model_name=model_name)

            for meta, response in zip(metadata, responses):
                all_results.append({
                    **meta,
                    'silicon_response': parse_response(response, var_info['scale']),
                    'raw_response': response,
                    'model': model_name
                })

    if not all_results:
        logger.error("No results generated. Check data files.")
        return None

    # ---- Step 2: Save raw data ----
    results_df = pd.DataFrame(all_results)
    raw_file = os.path.join(RESULTS_DIR, f"silicon_overtime_raw_{model_name}.csv")
    results_df.to_csv(raw_file, index=False)
    logger.info(f"\nRaw data saved to {raw_file}")
    logger.info(f"Total observations: {len(results_df)}")

    # ---- Step 3: Aggregate by round × country × variable ----
    agg_results = []

    for (ess_round, cntry, var_name), group in results_df.groupby(
            ['essround', 'cntry', 'variable']):
        human = pd.to_numeric(group['human_response'], errors='coerce')
        silicon = pd.to_numeric(group['silicon_response'], errors='coerce')
        valid = human.notna() & silicon.notna()

        survey_year = ESS_ROUND_YEAR.get(int(ess_round), 0)

        # Individual-level correlation for this cell
        r_indiv = np.nan
        if valid.sum() >= 10:
            r_indiv, _ = stats.pearsonr(human[valid], silicon[valid])

        agg_results.append({
            'essround': ess_round,
            'survey_year': survey_year,
            'cntry': cntry,
            'country_name': COUNTRY_NAMES.get(cntry, cntry),
            'variable': var_name,
            'survey_mean': human.mean(),
            'silicon_mean': silicon.mean(),
            'survey_sd': human.std(),
            'silicon_sd': silicon.std(),
            'r_individual': r_indiv,
            'n': len(group),
            'n_valid': valid.sum(),
            'model': model_name
        })

    agg_df = pd.DataFrame(agg_results)
    agg_file = os.path.join(RESULTS_DIR, f"silicon_overtime_agg_{model_name}.csv")
    agg_df.to_csv(agg_file, index=False)
    logger.info(f"Aggregated data saved to {agg_file}")

    # ---- Step 4: Overtime correlation analysis ----
    # For each country × variable: does the silicon trend track the survey trend?
    overtime_corr = []

    for var_name in SOCIAL_TRUST_VARIABLES:
        var_agg = agg_df[agg_df['variable'] == var_name]

        # Overall across all country-round cells
        valid_cells = var_agg.dropna(subset=['survey_mean', 'silicon_mean'])
        if len(valid_cells) >= 5:
            r_overall, p_overall = stats.pearsonr(
                valid_cells['survey_mean'], valid_cells['silicon_mean'])
        else:
            r_overall, p_overall = np.nan, np.nan

        overtime_corr.append({
            'variable': var_name,
            'cntry': 'ALL_POOLED',
            'r_overtime': r_overall,
            'p_overtime': p_overall,
            'n_cells': len(valid_cells),
            'method': 'pooled_country_round_cells'
        })

        # Per country: correlation of survey_mean and silicon_mean across rounds
        for cntry in sorted(var_agg['cntry'].unique()):
            cntry_data = var_agg[var_agg['cntry'] == cntry].sort_values('essround')
            cntry_valid = cntry_data.dropna(subset=['survey_mean', 'silicon_mean'])

            if len(cntry_valid) >= 3:
                r_c, p_c = stats.pearsonr(
                    cntry_valid['survey_mean'], cntry_valid['silicon_mean'])
            else:
                r_c, p_c = np.nan, np.nan

            overtime_corr.append({
                'variable': var_name,
                'cntry': cntry,
                'country_name': COUNTRY_NAMES.get(cntry, cntry),
                'r_overtime': r_c,
                'p_overtime': p_c,
                'n_rounds': len(cntry_valid),
                'survey_trend': cntry_valid['survey_mean'].values.tolist(),
                'silicon_trend': cntry_valid['silicon_mean'].values.tolist(),
                'years': cntry_valid['survey_year'].values.tolist(),
                'method': 'within_country_across_rounds'
            })

    corr_df = pd.DataFrame(overtime_corr)
    corr_file = os.path.join(RESULTS_DIR, f"silicon_overtime_correlation_{model_name}.csv")
    corr_df.to_csv(corr_file, index=False)

    # ---- Step 5: Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("OVERTIME ANALYSIS SUMMARY")
    logger.info("=" * 70)

    for var_name in SOCIAL_TRUST_VARIABLES:
        pooled = corr_df[(corr_df['variable'] == var_name) &
                         (corr_df['cntry'] == 'ALL_POOLED')]
        if len(pooled) > 0:
            r_val = pooled.iloc[0]['r_overtime']
            p_val = pooled.iloc[0]['p_overtime']
            n_val = pooled.iloc[0]['n_cells']
            logger.info(f"  {var_name} pooled: r={r_val:.3f}, p={p_val:.4f}, "
                         f"n_cells={n_val}")

        # Per-country summary
        per_country = corr_df[(corr_df['variable'] == var_name) &
                              (corr_df['cntry'] != 'ALL_POOLED')]
        valid_countries = per_country.dropna(subset=['r_overtime'])
        if len(valid_countries) > 0:
            mean_r = valid_countries['r_overtime'].mean()
            pos_count = (valid_countries['r_overtime'] > 0).sum()
            sig_count = (valid_countries['p_overtime'] < 0.05).sum()
            logger.info(f"  {var_name} per-country: mean r={mean_r:.3f}, "
                         f"{pos_count}/{len(valid_countries)} positive, "
                         f"{sig_count}/{len(valid_countries)} significant (p<.05)")

    logger.info("=" * 70)

    return results_df, agg_df, corr_df


# ============================================================================
# BLOCK: BACKSTORY ABLATION (v4 vs v5 backstory comparison)
# ============================================================================

def run_backstory_ablation(df_raw, model_name='qwen', sample_per_country=500,
                           countries=None, seed=888):
    """
    Ablation test: compare v4 (17-variable) vs v5 (27-variable) backstories.

    Uses IDENTICAL respondents with two different backstory encodings.
    Runs both through the same model, then compares individual-level r.

    This isolates the causal effect of backstory richness, controlling for
    sample composition and model.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Pre-loaded ESS data WITHOUT backstories (will be generated here).
    model_name : str
        Model to use.
    sample_per_country : int
        Respondents per country (default: 500).
    countries : list of str, optional
        Countries to test (default: GB, DE, FI).
    seed : int
        Random seed.
    """
    logger = setup_logging('ablation', model_name)
    logger.info("=== BACKSTORY ABLATION: v4 (17 vars) vs v5 (27 vars) ===")

    random.seed(seed)
    np.random.seed(seed)

    if countries is None:
        countries = ['GB', 'DE', 'FI']

    # ---- Step 1: Sample identical respondents ----
    countries_upper = [c.upper() for c in countries]
    df = df_raw[df_raw['cntry'].isin(countries_upper)].copy()

    sampled_parts = []
    for cntry in sorted(df['cntry'].unique()):
        cntry_df = df[df['cntry'] == cntry]
        if len(cntry_df) > sample_per_country:
            cntry_df = cntry_df.sample(n=sample_per_country, random_state=seed)
        sampled_parts.append(cntry_df)
    df = pd.concat(sampled_parts, ignore_index=True)

    logger.info(f"Sample: {len(df)} respondents from {countries_upper}")
    logger.info(f"  Per-country: {df.groupby('cntry').size().to_dict()}")

    # ---- Step 2: Generate both backstory versions ----
    logger.info("Generating v4 backstories (17 variables)...")
    backstories_v4 = df.apply(lambda row: generate_backstory(row, mode='v4'), axis=1)
    logger.info(f"  v4 mean length: {backstories_v4.str.len().mean():.0f} chars")

    logger.info("Generating v5 backstories (27 variables)...")
    backstories_v5 = df.apply(lambda row: generate_backstory(row, mode='v5'), axis=1)
    logger.info(f"  v5 mean length: {backstories_v5.str.len().mean():.0f} chars")

    logger.info(f"  Mean char increase: +{(backstories_v5.str.len() - backstories_v4.str.len()).mean():.0f} "
                f"({((backstories_v5.str.len().mean() / backstories_v4.str.len().mean()) - 1) * 100:.0f}%)")

    # ---- Step 3: Generate prompts for both conditions ----
    trust_vars = SOCIAL_TRUST_VARIABLES
    all_results = []

    for condition, backstories in [('v4', backstories_v4), ('v5', backstories_v5)]:
        prompts = []
        metadata = []

        for var_name, var_info in trust_vars.items():
            for i in range(len(df)):
                row = df.iloc[i]
                prompt = PROMPT_1P_STANDARD.format(
                    backstory=backstories.iloc[i],
                    question=var_info['question'],
                    scale=var_info['scale']
                )
                prompts.append(prompt)
                metadata.append({
                    'idno': row['idno'],
                    'cntry': row['cntry'],
                    'variable': var_name,
                    'human_response': row.get(var_name, np.nan),
                    'condition': condition
                })

        logger.info(f"Generating {len(prompts)} responses for {condition} condition...")
        responses = generate_with_vllm(prompts, model_name=model_name)

        for meta, response in zip(metadata, responses):
            all_results.append({
                **meta,
                'silicon_response': parse_response(response,
                    trust_vars[meta['variable']]['scale']),
                'raw_response': response,
                'model': model_name
            })

    results_df = pd.DataFrame(all_results)
    raw_file = os.path.join(RESULTS_DIR,
                            f"silicon_ablation_raw_{model_name}_seed{seed}.csv")
    results_df.to_csv(raw_file, index=False)
    logger.info(f"Raw data saved to {raw_file}")

    # ---- Step 4: Compare individual-level r across conditions ----
    ablation_results = []

    for condition in ['v4', 'v5']:
        for var_name in trust_vars:
            cond_data = results_df[
                (results_df['condition'] == condition) &
                (results_df['variable'] == var_name)
            ]
            human = pd.to_numeric(cond_data['human_response'], errors='coerce').values
            silicon = pd.to_numeric(cond_data['silicon_response'], errors='coerce').values
            valid = ~(np.isnan(human) | np.isnan(silicon))

            if valid.sum() < 10:
                continue

            r_p, p_p = stats.pearsonr(human[valid], silicon[valid])
            r_s, p_s = stats.spearmanr(human[valid], silicon[valid])
            metrics = compute_variance_metrics(human, silicon)

            ablation_results.append({
                'condition': condition,
                'variable': var_name,
                'r_pearson': r_p, 'p_pearson': p_p,
                'r_spearman': r_s,
                'var_ratio': metrics['var_ratio'],
                'human_mean': metrics['human_mean'],
                'silicon_mean': metrics['silicon_mean'],
                'mean_bias': metrics['silicon_mean'] - metrics['human_mean'],
                'n_valid': valid.sum(),
                'model': model_name
            })

            # Per-country breakdown
            for cntry in sorted(cond_data['cntry'].unique()):
                cd = cond_data[cond_data['cntry'] == cntry]
                h = pd.to_numeric(cd['human_response'], errors='coerce').values
                s = pd.to_numeric(cd['silicon_response'], errors='coerce').values
                v = ~(np.isnan(h) | np.isnan(s))
                if v.sum() < 10:
                    continue
                r_c, _ = stats.pearsonr(h[v], s[v])
                ablation_results.append({
                    'condition': condition,
                    'variable': var_name,
                    'cntry': cntry,
                    'r_pearson': r_c,
                    'n_valid': v.sum(),
                    'model': model_name
                })

    abl_df = pd.DataFrame(ablation_results)
    abl_file = os.path.join(RESULTS_DIR,
                            f"silicon_ablation_comparison_{model_name}.csv")
    abl_df.to_csv(abl_file, index=False)

    # ---- Step 5: Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("BACKSTORY ABLATION SUMMARY")
    logger.info("=" * 70)

    # Pooled comparison (no cntry column = pooled rows)
    pooled = abl_df[abl_df['cntry'].isna()] if 'cntry' in abl_df.columns else abl_df

    for var_name in trust_vars:
        v4_row = pooled[(pooled['condition'] == 'v4') &
                        (pooled['variable'] == var_name)]
        v5_row = pooled[(pooled['condition'] == 'v5') &
                        (pooled['variable'] == var_name)]
        if len(v4_row) > 0 and len(v5_row) > 0:
            r_v4 = v4_row.iloc[0]['r_pearson']
            r_v5 = v5_row.iloc[0]['r_pearson']
            delta = r_v5 - r_v4
            logger.info(f"  {var_name}: v4 r={r_v4:.4f}, v5 r={r_v5:.4f}, "
                         f"Δr={delta:+.4f}")

    # Overall
    v4_mean = pooled[pooled['condition'] == 'v4']['r_pearson'].mean()
    v5_mean = pooled[pooled['condition'] == 'v5']['r_pearson'].mean()
    logger.info(f"\n  OVERALL: v4 mean r={v4_mean:.4f}, v5 mean r={v5_mean:.4f}, "
                 f"Δ={v5_mean - v4_mean:+.4f}")

    if abs(v5_mean - v4_mean) < 0.01:
        logger.info("  → Backstory expansion has MINIMAL effect on individual-level recovery.")
        logger.info("  → Safe to proceed with v5 backstory for main analysis.")
    elif v5_mean > v4_mean:
        logger.info("  → Expanded backstory IMPROVES individual-level recovery.")
    else:
        logger.info("  → Expanded backstory REDUCES individual-level recovery.")
        logger.info("  → Consider using v4 backstory for main analysis.")

    logger.info("=" * 70)

    return results_df, abl_df


# ============================================================================
# BLOCK: ICC ANALYSIS (Why does ppltrst correlate better?)
# ============================================================================

def run_icc_analysis(ess_file=None, countries=None):
    """
    Compute Intraclass Correlation Coefficient (ICC) for social trust variables.

    ICC(1) = between-country variance / total variance.
    Higher ICC means more of the variance is between countries, which makes
    country-level recovery easier for the LLM.

    This tests the hypothesis that ppltrst has a higher ICC than pplfair/pplhlp,
    explaining why the cross-national gradient is only captured for ppltrst.

    No GPU required — pure statistical computation on ESS data.

    Parameters
    ----------
    ess_file : str, optional
        Path to ESS CSV file.
    countries : list of str, optional
        Country filter.

    Returns
    -------
    pd.DataFrame
        ICC results for each social trust variable.
    """
    logger = setup_logging('icc', 'analysis')
    logger.info("=== ICC ANALYSIS (No GPU required) ===")

    if ess_file is None:
        ess_file = ESS11_FILE

    # Load minimal data (just trust vars + country)
    target_vars = list(SOCIAL_TRUST_VARIABLES.keys())
    cols = ['cntry', 'pspwght'] + target_vars
    available = pd.read_csv(ess_file, nrows=0).columns.tolist()
    cols_to_load = [c for c in cols if c in available]
    df = pd.read_csv(ess_file, usecols=cols_to_load)

    if countries:
        df = df[df['cntry'].isin([c.upper() for c in countries])]

    # Clean missing codes
    for var in target_vars:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors='coerce')
            df.loc[df[var].isin(ESS_MISSING_CODES), var] = np.nan

    logger.info(f"Data: {len(df)} respondents, {df['cntry'].nunique()} countries")

    icc_results = []

    for var_name in target_vars:
        var_data = df[['cntry', var_name]].dropna()
        if len(var_data) < 100:
            continue

        # One-way random effects ANOVA for ICC(1)
        # ICC(1) = (MS_between - MS_within) / (MS_between + (k-1)*MS_within)
        # where k = average group size

        groups = var_data.groupby('cntry')[var_name]
        n_groups = groups.ngroups
        group_sizes = groups.size()
        grand_mean = var_data[var_name].mean()
        total_n = len(var_data)

        # Between-group sum of squares
        ss_between = sum(
            n_j * (mean_j - grand_mean) ** 2
            for (_, mean_j), n_j in zip(groups.mean().items(), group_sizes)
        )

        # Within-group sum of squares
        ss_within = sum(
            ((g - g.mean()) ** 2).sum()
            for _, g in groups
        )

        df_between = n_groups - 1
        df_within = total_n - n_groups

        ms_between = ss_between / df_between if df_between > 0 else 0
        ms_within = ss_within / df_within if df_within > 0 else 0

        # Average group size (harmonic mean for unbalanced data)
        k_0 = (total_n - sum(n ** 2 for n in group_sizes) / total_n) / (n_groups - 1)

        # ICC(1)
        if (ms_between + (k_0 - 1) * ms_within) > 0:
            icc1 = (ms_between - ms_within) / (ms_between + (k_0 - 1) * ms_within)
        else:
            icc1 = 0.0

        # F-test for ICC significance
        f_stat = ms_between / ms_within if ms_within > 0 else np.nan
        from scipy.stats import f as f_dist
        p_val = 1 - f_dist.cdf(f_stat, df_between, df_within) if not np.isnan(f_stat) else np.nan

        # Additional descriptives
        overall_var = var_data[var_name].var()
        between_var = ms_between - ms_within
        if between_var < 0:
            between_var = 0  # can happen with very small ICC
        within_var = ms_within

        # Country-level range of means
        country_means = groups.mean()
        country_range = country_means.max() - country_means.min()
        country_sd = country_means.std()

        icc_results.append({
            'variable': var_name,
            'label': SOCIAL_TRUST_VARIABLES[var_name]['label'],
            'icc1': icc1,
            'f_stat': f_stat,
            'p_value': p_val,
            'n_countries': n_groups,
            'n_total': total_n,
            'grand_mean': grand_mean,
            'total_var': overall_var,
            'between_var': max(between_var, 0),
            'within_var': within_var,
            'between_pct': max(between_var, 0) / overall_var * 100 if overall_var > 0 else 0,
            'country_mean_range': country_range,
            'country_mean_sd': country_sd,
            'k_avg': k_0
        })

    icc_df = pd.DataFrame(icc_results)
    icc_file = os.path.join(RESULTS_DIR, "silicon_icc_analysis.csv")
    icc_df.to_csv(icc_file, index=False)

    # ---- Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("ICC ANALYSIS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Variable':<12} {'ICC(1)':<10} {'F':<10} {'p':<12} "
                f"{'Between%':<10} {'CntryRange':<12} {'CntrySd':<10}")
    logger.info("-" * 76)

    for _, row in icc_df.iterrows():
        sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
        logger.info(f"  {row['variable']:<10} {row['icc1']:<10.4f} {row['f_stat']:<10.1f} "
                     f"{row['p_value']:<12.2e} {row['between_pct']:<10.1f}% "
                     f"{row['country_mean_range']:<12.2f} {row['country_mean_sd']:<10.3f}"
                     f" {sig}")

    # Interpretation
    logger.info("")
    sorted_icc = icc_df.sort_values('icc1', ascending=False)
    highest = sorted_icc.iloc[0]
    lowest = sorted_icc.iloc[-1]
    logger.info(f"Highest ICC: {highest['variable']} (ICC={highest['icc1']:.4f})")
    logger.info(f"Lowest ICC:  {lowest['variable']} (ICC={lowest['icc1']:.4f})")
    logger.info(f"ICC ratio (highest/lowest): {highest['icc1'] / lowest['icc1']:.2f}x")

    if highest['variable'] == 'ppltrst':
        logger.info("\n→ CONFIRMED: ppltrst has the highest ICC among trust items.")
        logger.info("  This explains why cross-national recovery is strongest for ppltrst:")
        logger.info("  more between-country variance = stronger signal for the LLM to capture.")
    else:
        logger.info(f"\n→ NOTE: {highest['variable']} has the highest ICC, not ppltrst.")
        logger.info("  The ICC hypothesis does not fully explain the ppltrst advantage.")

    logger.info("=" * 70)

    return icc_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Silicon Sampling v6 — Multi-Country ESS11 Framework (Auto-Backup)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Block options:
  trust       Social trust analysis (recommended first run)
  full        All 42 ESS11 variables
  mechanisms  Prompt framing (RQ3) + format sensitivity (RQ6)
  fewshot     Few-shot anchoring (RQ11)
  distanchor  Distribution anchoring (RQ12)
  rq8         Model comparison (Qwen vs Llama)
  deep        Deep statistical analysis (requires 'trust' run first)
  subgroup    Subgroup heterogeneity (requires 'trust' run first)
  overtime    Overtime analysis (requires multi-round ESS data)
  ablation    Backstory ablation: v4 (17 vars) vs v5 (27 vars)
  icc         ICC analysis: why ppltrst > pplfair/pplhlp (no GPU needed)
  all         Run trust + mechanisms + fewshot + distanchor + deep + subgroup

Examples:
  python silicon_sampling_extended_v5.py --block trust --countries GB FR DE
  python silicon_sampling_extended_v5.py --block trust --model llama
  python silicon_sampling_extended_v5.py --block ablation --countries GB DE FI
  python silicon_sampling_extended_v5.py --block icc
  python silicon_sampling_extended_v5.py --block all --countries GB FR DE NL SE
  python silicon_sampling_extended_v5.py --block full --countries GB
        """)

    parser.add_argument('--block', type=str, default='trust',
                        choices=['trust', 'full', 'mechanisms', 'fewshot',
                                 'distanchor', 'rq8', 'deep', 'subgroup',
                                 'overtime', 'ablation', 'icc', 'all'],
                        help='Which block to run')
    parser.add_argument('--model', type=str, default='qwen',
                        choices=['qwen', 'llama'],
                        help='Model to use')
    parser.add_argument('--prompt', type=str, default='1p',
                        choices=['1p', '3p', '1p_idiosyncratic'],
                        help='Prompt framing: 1p (role-play), 3p (third-person), 1p_idiosyncratic')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (default: 0.7)')
    parser.add_argument('--backstory', type=str, default='v5_clean',
                        choices=['v4', 'v5', 'v5_clean', 'demo_only', 'minimal', 'ses', 'political', 'full_clean', 'full_nocountry', 'full_noregion', 'full_nopolitical', 'minimal_politics', 'minimal_politics_ses', 'minimal_politics_ses_geo', 'minimal_politics_econ', 'full_noascriptive', 'full_nosocioecon', 'full_nohousehold', 'full_nocivic', 'full_nomembership', 'full_nominority', 'full_nodomicil', 'minimal_ses', 'minimal_membership', 'minimal_household', 'minimal_civic', 'minimal_minority', 'minimal_domicil', 'minimal_region', 'minimal_ageonly', 'minimal_yrbrnonly', 'country_only', 'full_noincome', 'full_swapcountry', 'minimal_union', 'minimal_internet'],  # v10; v12.2-3 rungs; v12.4-5 LOGO; v12.6 add-one + probes + swap; v12.7 membership split
                        help='Backstory mode: v5_clean (20, default), demo_only (3, no country), minimal (4), ses (13), political (14), full_clean (=v5_clean, 20), full_noregion (19), full_nocountry (18), full_nopolitical (18), minimal_politics (6 = minimal + lrscale + clsprty), minimal_politics_ses (10 = +education/activity/income), minimal_politics_ses_geo (14 = +domicile/region/born/citizen), minimal_politics_econ (12 = politics rung + education/activity/income/union/internet), v4 (17, leaked), v5 (27, leaked)')  # v12 FIX: counts were wrong
    parser.add_argument('--max_model_len', type=int, default=2048,  # v11: 1024 in all legacy runs; expose it so the verify gate can isolate KV-block effects from other changes
                        help='vLLM max_model_len (legacy runs used 1024; 2048 gives headroom for anchored scale cards)')
    parser.add_argument('--gen_seed', type=int, default=None,  # v10 CHANGE (B3)
                        help='vLLM ENGINE seed (LLM(seed=...)); default None = unseeded, matching all legacy runs. Set a DIFFERENT value (e.g. 889) for the repeated-run stability arm, otherwise a seeded rerun may be near-deterministic and measure nothing. NEVER passed per-request via SamplingParams: an identical per-request seed would draw the same CDF quantile for every prompt and distort response distributions.')
    parser.add_argument('--tag_suffix', type=str, default='',  # v10 CHANGE (H2)
                        help='Appended to the output tag; use e.g. _verify or _stab to keep verification/stability outputs from overwriting official result files')
    parser.add_argument('--missing_rule', type=str, default='range',  # v12 FIX (P0-A)
                        choices=['range', 'legacy'],
                        help="Human-side missingness rule. 'range' (default) sets values outside "
                             "each item's valid range to missing, removing the ESS 7/8/9 "
                             "refusal/DK/no-answer codes that pre-v12 runs retained on the 16 "
                             "coarse-scale items. 'legacy' reproduces pre-v12 behaviour and should "
                             "be used only to regenerate a legacy comparator.")
    parser.add_argument('--human_weight', type=str, default='pspwght',  # v12 FIX (P0-B)
                        choices=['pspwght', 'none'],
                        help='Weighting of the HUMAN country means. All released runs used '
                             'pspwght (design-weighted); this was undocumented. The silicon side '
                             'is always unweighted. Recorded in the run manifest.')
    parser.add_argument('--scale_labels', type=str, default='numeric',  # v9 CHANGE
                        choices=['numeric', 'anchored'],
                        help='numeric = bare scale range (reproduces existing runs); anchored = append ESS codebook verbal labels to the scale line')
    parser.add_argument('--variables', type=str, nargs='*', default=None,  # v9 CHANGE
                        help='Optional subset of outcome variables (default: all 42)')
    parser.add_argument('--countries', type=str, nargs='*', default=None,
                        help='Country ISO codes (e.g., GB FR DE). Default: all 30.')
    parser.add_argument('--seed', type=int, default=888,
                        help='Random seed')
    parser.add_argument('--n_similar', type=int, default=3,
                        help='Number of similar respondents for few-shot (RQ11)')
    parser.add_argument('--sample_per_country', type=int, default=None,
                        help='Randomly sample N respondents per country (default: use all)')
    parser.add_argument('--ess_file', type=str, default=None,
                        help='Override default ESS data file path')
    parser.add_argument('--rounds_dir', type=str, default=None,
                        help='Directory containing multi-round ESS CSVs for overtime analysis')

    args = parser.parse_args()

    # Create directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # ICC analysis needs no GPU and no backstory generation — handle separately
    if args.block == 'icc':
        run_icc_analysis(ess_file=args.ess_file, countries=args.countries)
        print(f"\n{'=' * 70}")
        print("ICC ANALYSIS COMPLETE")
        print(f"Results in: {os.path.abspath(RESULTS_DIR)}/")
        print(f"{'=' * 70}")
        return

    # Ablation block handles its own data loading (needs both v4 and v5)
    if args.block == 'ablation':
        # Load data for ablation countries only (no backstories yet — ablation
        # generates both v4 and v5 versions internally)
        ablation_countries = args.countries or ['GB', 'DE', 'FI']
        df_raw = load_data(ess_file=args.ess_file, countries=ablation_countries,
                           sample_per_country=None, seed=args.seed,
                           backstory_mode='v4')  # mode irrelevant, will regenerate
        run_backstory_ablation(df_raw, model_name=args.model,
                               sample_per_country=args.sample_per_country or 500,
                               countries=ablation_countries,
                               seed=args.seed)
        print(f"\n{'=' * 70}")
        print("ABLATION COMPLETE")
        print(f"Results in: {os.path.abspath(RESULTS_DIR)}/")
        print(f"{'=' * 70}")
        return

    # --- v9 CHANGE: optional variable subset ---
    if args.variables:
        unknown = [v for v in args.variables if v not in ESS_VARIABLES]
        if unknown:
            raise ValueError(f"Unknown variables: {unknown}")
        for k in list(ESS_VARIABLES.keys()):
            if k not in args.variables:
                del ESS_VARIABLES[k]
        print(f"Variable subset: {len(ESS_VARIABLES)} of 42 items")

    # --- v9 CHANGE: verbal scale anchors (ESS showcard labels) ---
    if args.scale_labels == 'anchored' and args.block != 'full':
        # Other blocks (e.g. mechanisms) re-parse the scale string to build
        # reversed scales and would silently misbehave on anchored strings.
        raise ValueError("--scale_labels anchored is only supported with --block full")
    if args.scale_labels == 'anchored':
        # v12: the three trust items' inner dicts are shared with
        # SOCIAL_TRUST_VARIABLES (merged via **), so the in-place rewrite below
        # used to mutate that module-level dict too. Copy first: the guard above
        # made this harmless in practice, but it is a latent footgun as soon as a
        # long-scale trust item enters an anchored subset.
        for _k in list(ESS_VARIABLES):
            ESS_VARIABLES[_k] = dict(ESS_VARIABLES[_k])
        for k, info in ESS_VARIABLES.items():
            labs = ANCHOR_LABELS.get(k, {})
            if labs:
                lab_str = "; ".join(f"{v} = {t}" for v, t in sorted(labs.items()))
                info['scale'] = f"{info['scale']} ({lab_str})"
        print("Scale labels: ANCHORED (ESS codebook verbal labels appended)")

    # Load data with specified backstory mode
    df = load_data(ess_file=args.ess_file, countries=args.countries,
                   sample_per_country=args.sample_per_country, seed=args.seed,
                   backstory_mode=args.backstory,
                   missing_rule=args.missing_rule)  # v12

    print(f"\n{'=' * 70}")
    print(f"SILICON SAMPLING v6 — Multi-Country ESS11 (Auto-Backup)")
    print(f"{'=' * 70}")
    print(f"Block:      {args.block}")
    print(f"Model:      {args.model}")
    print(f"Prompt:     {args.prompt}")
    print(f"Temperature:{args.temperature}")
    BACKSTORY_DESC = {'v4': '17 variables (leaked)', 'v5': '27 variables (leaked)',
                      'v5_clean': '20 variables (clean)', 'demo_only': '3 variables (no country)',
                      'minimal': '4 variables',
                      'ses': '13 variables', 'political': '14 variables',   # v12 FIX: were 14/16
                      'full_clean': '20 variables (clean)',
                      'full_nocountry': '18 variables (full_clean minus country label and NUTS region)',  # v12
                      'full_noregion': '19 variables (full_clean minus NUTS region; country kept)',   # v12
                      'full_nopolitical': '18 variables (full_clean minus lrscale and clsprty)'}       # v12
    print(f"Backstory:  {args.backstory} ({BACKSTORY_DESC.get(args.backstory, args.backstory)})")
    print(f"Countries:  {sorted(df['cntry'].unique())} ({df['cntry'].nunique()} countries)")
    print(f"Sample:     {len(df)} respondents")
    print(f"Seed:       {args.seed}")
    backstory_tag = args.backstory if args.backstory not in ('v5', 'v5_clean') else ""
    if args.scale_labels == 'anchored':                                  # v9 CHANGE
        backstory_tag = (backstory_tag + "_anchored").lstrip("_")        # v9 CHANGE
    if args.tag_suffix:                                                  # v10 CHANGE (H2)
        backstory_tag = (backstory_tag + args.tag_suffix).lstrip("_")
    tag = f"{args.model}_{args.prompt}" + (f"_{backstory_tag}" if backstory_tag else "")
    print(f"Output tag: {tag}")
    print(f"{'=' * 70}\n")

    # Auto-backup existing results before overwriting (v6 feature)
    auto_backup_results(tag)

    # --- v11: run manifest. Output filenames encode neither n, seeds, edition nor
    # engine settings, yet every attenuation argument depends on n. Write it down. ---
    try:
        import subprocess, json as _json
        try:
            _git = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                  capture_output=True, text=True, timeout=5).stdout.strip() or 'n/a'
        except Exception:
            _git = 'n/a'
        try:
            import vllm as _vllm; _vv = getattr(_vllm, '__version__', 'n/a')
        except Exception:
            _vv = 'n/a'
        _manifest = {
            'tag': tag, 'block': args.block, 'model': args.model, 'prompt': args.prompt,
            'backstory': args.backstory, 'scale_labels': args.scale_labels,
            'variables_n': len(ESS_VARIABLES), 'variables': sorted(ESS_VARIABLES.keys()),
            'sample_per_country': args.sample_per_country, 'n_respondents': int(len(df)),
            'n_countries': int(df['cntry'].nunique()),
            'sampling_seed': args.seed, 'gen_seed': args.gen_seed,
            'temperature': args.temperature, 'max_model_len': args.max_model_len,
            'missing_rule': args.missing_rule,        # v12
            'human_weight': args.human_weight,        # v12
            'pandas_version': pd.__version__,         # v12: nested subsampling depends on it
            'numpy_version': np.__version__,          # v12
            'ess_file': os.path.basename(args.ess_file or ESS11_FILE),
            'eduyrs_guard': ENABLE_EDUYRS_GUARD,
            'vllm_version': _vv, 'git': _git,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }
        with open(os.path.join(RESULTS_DIR, f"manifest_{tag}.json"), 'w') as _f:
            _json.dump(_manifest, _f, indent=2)
        print(f"Manifest: results/manifest_{tag}.json")
    except Exception as _e:
        logging.warning(f"Manifest not written: {_e}")

    if args.block == 'trust':
        run_social_trust(df, model_name=args.model, seed=args.seed,
                         prompt_mode=args.prompt, temperature=args.temperature)

    elif args.block == 'full':
        run_full_variables(df, model_name=args.model, seed=args.seed,
                           prompt_mode=args.prompt, temperature=args.temperature,
                           backstory_tag=backstory_tag,
                           gen_seed=args.gen_seed,  # v11 FIX: None = legacy unseeded generation; set explicitly (e.g. 889) only for repeat arms
                           max_model_len=args.max_model_len,  # v11
                           human_weight=args.human_weight)  # v12

    elif args.block == 'mechanisms':
        run_mechanisms(df, model_name=args.model, seed=args.seed)

    elif args.block == 'fewshot':
        run_fewshot(df, model_name=args.model, n_similar=args.n_similar, seed=args.seed)

    elif args.block == 'distanchor':
        run_dist_anchoring(df, model_name=args.model, seed=args.seed)

    elif args.block == 'rq8':
        run_model_comparison(df, seed=args.seed)

    elif args.block == 'deep':
        run_deep_analysis(df, model_name=args.model, seed=args.seed)

    elif args.block == 'subgroup':
        run_subgroup_analysis(df, model_name=args.model, seed=args.seed)

    elif args.block == 'overtime':
        rounds_dir = args.rounds_dir or ESS_ROUNDS_DIR
        sample_n = args.sample_per_country or 200
        run_overtime_analysis(rounds_dir, model_name=args.model,
                              countries=args.countries,
                              sample_per_country=sample_n, seed=args.seed)

    elif args.block == 'all':
        print("[1/6] Social Trust Analysis...")
        run_social_trust(df, model_name=args.model, seed=args.seed,
                         prompt_mode=args.prompt, temperature=args.temperature)

        print("\n[2/6] Mechanism Tests (RQ3, RQ6)...")
        run_mechanisms(df, model_name=args.model, seed=args.seed)

        print("\n[3/6] Few-shot Anchoring (RQ11)...")
        run_fewshot(df, model_name=args.model, n_similar=args.n_similar, seed=args.seed)

        print("\n[4/6] Distribution Anchoring (RQ12)...")
        run_dist_anchoring(df, model_name=args.model, seed=args.seed)

        print("\n[5/6] Deep Analysis...")
        run_deep_analysis(df, model_name=args.model, seed=args.seed)

        print("\n[6/6] Subgroup Analysis...")
        run_subgroup_analysis(df, model_name=args.model, seed=args.seed)

    print(f"\n{'=' * 70}")
    print("EXPERIMENT COMPLETE")
    print(f"Results in: {os.path.abspath(RESULTS_DIR)}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
