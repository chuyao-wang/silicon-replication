#!/usr/bin/env python3
"""
Silicon Sampling Validation Framework - COMPLETE VERSION
=========================================================
All 3 Blocks, 7 RQs, 50 Variables, n=1000
Optimized for VLLM on Winston HPC

Block 1: RQ1 (Individual Recovery) + RQ2 (Variance Compression)
Block 2: RQ3 (Prompt Framing) + RQ4 (Enrichment) + RQ5 (Covariates) + RQ6 (Format)
Block 3: RQ7a/b/c (Subgroup Heterogeneity)

Usage:
    python silicon_sampling_full_vllm.py --block 1 --sample_size 1000
    python silicon_sampling_full_vllm.py --block 2 --sample_size 1000
    python silicon_sampling_full_vllm.py --block 3 --sample_size 1000
    python silicon_sampling_full_vllm.py --block all --sample_size 1000
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
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, wilcoxon, entropy
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'sample_size': 1000,
    'random_seeds': [888],
    'temperature': 0.7,
    'n_bootstrap': 1000,
    'alpha': 0.05,
    'model_name': 'Qwen/Qwen2.5-7B-Instruct',
    'max_model_len': 4096,
    'gpu_memory_utilization': 0.85,
    'max_tokens': 150,
}

# =============================================================================
# 50 VARIABLES ACROSS 10 DOMAINS
# =============================================================================

VARIABLES = {
    # DOMAIN 1: Political Attitudes (5 items)
    'stfdem': {'domain': 'Political Attitudes', 'question': 'How satisfied with the way democracy works in country (0=extremely dissatisfied, 10=extremely satisfied)', 'scale': (0, 10)},
    'polintr': {'domain': 'Political Attitudes', 'question': 'How interested in politics (1=very interested, 4=not at all interested)', 'scale': (1, 4)},
    'lrscale': {'domain': 'Political Attitudes', 'question': 'Placement on left-right scale (0=left, 10=right)', 'scale': (0, 10)},
    'psppipla': {'domain': 'Political Attitudes', 'question': 'Political system allows people to have influence on politics (1=not at all, 5=a great deal)', 'scale': (1, 5)},
    'psppsgva': {'domain': 'Political Attitudes', 'question': 'Political system allows people to have a say in what government does (1=not at all, 5=a great deal)', 'scale': (1, 5)},
    
    # DOMAIN 2: Political Efficacy (2 items)
    'actrolga': {'domain': 'Political Efficacy', 'question': 'Able to take active role in political group (1=not at all able, 5=completely able)', 'scale': (1, 5)},
    'cptppola': {'domain': 'Political Efficacy', 'question': 'Confident in own ability to participate in politics (1=not at all confident, 5=completely confident)', 'scale': (1, 5)},
    
    # DOMAIN 3: Institutional Trust (7 items)
    'trstprl': {'domain': 'Institutional Trust', 'question': "Trust in country's parliament (0=no trust, 10=complete trust)", 'scale': (0, 10)},
    'trstplt': {'domain': 'Institutional Trust', 'question': 'Trust in politicians (0=no trust, 10=complete trust)', 'scale': (0, 10)},
    'trstlgl': {'domain': 'Institutional Trust', 'question': 'Trust in the legal system (0=no trust, 10=complete trust)', 'scale': (0, 10)},
    'trstprt': {'domain': 'Institutional Trust', 'question': 'Trust in political parties (0=no trust, 10=complete trust)', 'scale': (0, 10)},
    'trstplc': {'domain': 'Institutional Trust', 'question': 'Trust in the police (0=no trust, 10=complete trust)', 'scale': (0, 10)},
    'trstep': {'domain': 'Institutional Trust', 'question': 'Trust in the European Parliament (0=no trust, 10=complete trust)', 'scale': (0, 10)},
    'trstun': {'domain': 'Institutional Trust', 'question': 'Trust in the United Nations (0=no trust, 10=complete trust)', 'scale': (0, 10)},
    
    # DOMAIN 4: Social Trust (5 items)
    'ppltrst': {'domain': 'Social Trust', 'question': "Most people can be trusted or you can't be too careful (0=can't be too careful, 10=most people can be trusted)", 'scale': (0, 10)},
    'pplfair': {'domain': 'Social Trust', 'question': 'Most people try to take advantage of you or try to be fair (0=take advantage, 10=try to be fair)', 'scale': (0, 10)},
    'pplhlp': {'domain': 'Social Trust', 'question': 'People mostly look out for themselves or try to be helpful (0=look out for themselves, 10=try to be helpful)', 'scale': (0, 10)},
    'sclmeet': {'domain': 'Social Trust', 'question': 'How often socially meet with friends, relatives or colleagues (1=never, 7=every day)', 'scale': (1, 7)},
    'inprdsc': {'domain': 'Social Trust', 'question': 'How many people with whom you can discuss intimate and personal matters (0=none, 6=10 or more)', 'scale': (0, 6)},
    
    # DOMAIN 5: Subjective Well-being (6 items)
    'happy': {'domain': 'Subjective Well-being', 'question': 'How happy are you (0=extremely unhappy, 10=extremely happy)', 'scale': (0, 10)},
    'stflife': {'domain': 'Subjective Well-being', 'question': 'How satisfied with life as a whole (0=extremely dissatisfied, 10=extremely satisfied)', 'scale': (0, 10)},
    'stfeco': {'domain': 'Subjective Well-being', 'question': 'How satisfied with present state of economy in country (0=extremely dissatisfied, 10=extremely satisfied)', 'scale': (0, 10)},
    'stfgov': {'domain': 'Subjective Well-being', 'question': 'How satisfied with the national government (0=extremely dissatisfied, 10=extremely satisfied)', 'scale': (0, 10)},
    'stfhlth': {'domain': 'Subjective Well-being', 'question': 'State of health services in country nowadays (0=extremely bad, 10=extremely good)', 'scale': (0, 10)},
    'stfedu': {'domain': 'Subjective Well-being', 'question': 'State of education in country nowadays (0=extremely bad, 10=extremely good)', 'scale': (0, 10)},
    
    # DOMAIN 6: Immigration Attitudes (6 items)
    'imueclt': {'domain': 'Immigration Attitudes', 'question': "Country's cultural life undermined or enriched by immigrants (0=undermined, 10=enriched)", 'scale': (0, 10)},
    'imwbcnt': {'domain': 'Immigration Attitudes', 'question': 'Immigrants make country worse or better place to live (0=worse, 10=better)', 'scale': (0, 10)},
    'imbgeco': {'domain': 'Immigration Attitudes', 'question': "Immigration bad or good for country's economy (0=bad, 10=good)", 'scale': (0, 10)},
    'imsmetn': {'domain': 'Immigration Attitudes', 'question': 'Allow many or few immigrants of same race/ethnic group as majority (1=allow many, 4=allow none)', 'scale': (1, 4)},
    'impcntr': {'domain': 'Immigration Attitudes', 'question': 'Allow many or few immigrants from poorer countries outside Europe (1=allow many, 4=allow none)', 'scale': (1, 4)},
    'imdfetn': {'domain': 'Immigration Attitudes', 'question': 'Allow many or few immigrants of different race/ethnic group from majority (1=allow many, 4=allow none)', 'scale': (1, 4)},
    
    # DOMAIN 7: Social Values (4 items)
    'gincdif': {'domain': 'Social Values', 'question': 'Government should reduce differences in income levels (1=agree strongly, 5=disagree strongly)', 'scale': (1, 5)},
    'freehms': {'domain': 'Social Values', 'question': 'Gays and lesbians free to live as they wish (1=agree strongly, 5=disagree strongly)', 'scale': (1, 5)},
    'hmsfmlsh': {'domain': 'Social Values', 'question': 'Ashamed if close family member gay or lesbian (1=agree strongly, 5=disagree strongly)', 'scale': (1, 5)},
    'euftf': {'domain': 'Social Values', 'question': 'European unification go further or gone too far (0=unification go further, 10=gone too far)', 'scale': (0, 10)},
    
    # DOMAIN 8: Personal Values - Schwartz (10 items)
    'ipcrtiv': {'domain': 'Personal Values', 'question': 'Important to think new ideas and be creative (1=very much like me, 6=not like me at all)', 'scale': (1, 6)},
    'imprich': {'domain': 'Personal Values', 'question': 'Important to be rich, have money and expensive things (1=very much like me, 6=not like me at all)', 'scale': (1, 6)},
    'ipeqopt': {'domain': 'Personal Values', 'question': 'Important that people are treated equally and have equal opportunities (1=very much like me, 6=not like me at all)', 'scale': (1, 6)},
    'impsafe': {'domain': 'Personal Values', 'question': 'Important to live in secure surroundings (1=very much like me, 6=not like me at all)', 'scale': (1, 6)},
    'impfree': {'domain': 'Personal Values', 'question': 'Important to make own decisions and be free (1=very much like me, 6=not like me at all)', 'scale': (1, 6)},
    'iphlppl': {'domain': 'Personal Values', 'question': 'Important to help people and care for others well-being (1=very much like me, 6=not like me at all)', 'scale': (1, 6)},
    'ipsuces': {'domain': 'Personal Values', 'question': 'Important to be successful and that people recognise achievements (1=very much like me, 6=not like me at all)', 'scale': (1, 6)},
    'imptrad': {'domain': 'Personal Values', 'question': 'Important to follow traditions and customs (1=very much like me, 6=not like me at all)', 'scale': (1, 6)},
    'impenv': {'domain': 'Personal Values', 'question': 'Important to care for nature and environment (1=very much like me, 6=not like me at all)', 'scale': (1, 6)},
    'ipfrule': {'domain': 'Personal Values', 'question': 'Important to do what is told and follow rules (1=very much like me, 6=not like me at all)', 'scale': (1, 6)},
    
    # DOMAIN 9: National & European Attachment (2 items)
    'atchctr': {'domain': 'National Attachment', 'question': 'How emotionally attached to your country (0=not at all, 10=very emotionally attached)', 'scale': (0, 10)},
    'atcherp': {'domain': 'National Attachment', 'question': 'How emotionally attached to Europe (0=not at all, 10=very emotionally attached)', 'scale': (0, 10)},
    
    # DOMAIN 10: Health, Religion & Safety (4 items)
    'health': {'domain': 'Health & Wellbeing', 'question': 'Subjective general health (1=very good, 5=very bad)', 'scale': (1, 5)},
    'rlgatnd': {'domain': 'Health & Wellbeing', 'question': 'How often attend religious services apart from special occasions (1=every day, 7=never)', 'scale': (1, 7)},
    'pray': {'domain': 'Health & Wellbeing', 'question': 'How often pray apart from at religious services (1=every day, 7=never)', 'scale': (1, 7)},
    'aesfdrk': {'domain': 'Health & Wellbeing', 'question': 'Feeling of safety walking alone in local area after dark (1=very safe, 4=very unsafe)', 'scale': (1, 4)},
}

# Exclude lrscale from analysis (data leakage - backstory contains left-right placement)
EXCLUDE_VARS = ['lrscale']
ANALYSIS_VARS = [v for v in VARIABLES.keys() if v not in EXCLUDE_VARS]

# Covariates for RQ5
COVARIATE_VARS = ['gndr', 'agea', 'brncntr', 'ctzcntr', 'eduyrs', 'hincfel', 'lrscale', 'vote', 'blgetmg']

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

PROMPT_1P_STANDARD = (
    "Adopt the following persona and answer only based on it. "
    "Do not invent details beyond the provided attributes.\n\n"
    "{backstory}\n\n"
    "Question: {question}\n\n"
    "Reply with ONLY a single number within the valid scale range."
)

PROMPT_3P = (
    "Consider a person with the following characteristics:\n\n"
    "{backstory}\n\n"
    "How would this person answer the following question? "
    "Predict their response.\n\n"
    "Question: {question}\n\n"
    "Reply with ONLY a single number within the valid scale range."
)

PROMPT_1P_IDIOSYNCRATIC = (
    "Adopt the following persona and answer only based on it. "
    "Feel free to give unusual, unexpected, or inconsistent responses "
    "if they feel authentic to this specific individual.\n\n"
    "{backstory}\n\n"
    "Question: {question}\n\n"
    "Reply with ONLY a single number within the valid scale range."
)

DECOMPOSE_PROMPT = """Extract information from this backstory into exactly 4 blocks.
Return ONLY the 4 blocks, one per line, in this EXACT format:

BLOCK1: [demographics - age, gender only]
BLOCK2: [socioeconomic - education, income, employment]
BLOCK3: [political - left-right scale, voting]
BLOCK4: [immigration - birthplace, citizenship]

BACKSTORY:
"{backstory}"

Extract the 4 blocks now:"""

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(block_name):
    """Setup logging for the run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/silicon_{block_name}_{timestamp}.log"
    
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_tail_mass(x, threshold=1.5):
    """Compute proportion of responses beyond threshold SDs."""
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    x = x[~np.isnan(x)]
    if len(x) < 3 or np.std(x) == 0:
        return np.nan
    z = (x - np.mean(x)) / np.std(x)
    return (np.abs(z) > threshold).mean()

def compute_response_entropy(x, var_name=None):
    """Compute response entropy."""
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan
    
    if var_name and var_name in VARIABLES:
        scale_min, scale_max = VARIABLES[var_name]['scale']
    else:
        scale_min, scale_max = 0, 10
    
    n_bins = scale_max - scale_min + 1
    bin_range = (scale_min - 0.5, scale_max + 0.5)
    counts, _ = np.histogram(x, bins=n_bins, range=bin_range)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return entropy(probs, base=2)

def bootstrap_correlation_ci(x, y, n_bootstrap=1000, ci=0.95):
    """Compute correlation with bootstrap CI."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = np.array(x)[mask], np.array(y)[mask]
    if len(x) < 10:
        return np.nan, np.nan, np.nan, np.nan
    
    boot_corrs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(x), size=len(x), replace=True)
        r, _ = pearsonr(x[idx], y[idx])
        boot_corrs.append(r)
    
    r, p = pearsonr(x, y)
    lower = np.percentile(boot_corrs, (1-ci)/2 * 100)
    upper = np.percentile(boot_corrs, (1+ci)/2 * 100)
    return r, p, lower, upper

def parse_response(text, scale_min, scale_max):
    """Parse numeric response from LLM output."""
    if not text:
        return np.nan
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    if numbers:
        for num_str in numbers:
            val = float(num_str)
            if scale_min <= val <= scale_max:
                return val
    return np.nan

def parse_decomposition_output(text):
    """Parse backstory decomposition output."""
    blocks = {'block1': '', 'block2': '', 'block3': '', 'block4': ''}
    text = text.replace('**', '')
    
    patterns = [
        (r'BLOCK\s*1\s*[:\-=]\s*(.+?)(?=BLOCK\s*2|$)', 'block1'),
        (r'BLOCK\s*2\s*[:\-=]\s*(.+?)(?=BLOCK\s*3|$)', 'block2'),
        (r'BLOCK\s*3\s*[:\-=]\s*(.+?)(?=BLOCK\s*4|$)', 'block3'),
        (r'BLOCK\s*4\s*[:\-=]\s*(.+?)$', 'block4'),
    ]
    
    for pattern, block_key in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            content = re.sub(r'^\[|\]$', '', content).strip()
            content = re.sub(r'\s+', ' ', content)
            if content and content.lower() not in ['n/a', 'none', 'not provided']:
                blocks[block_key] = content
    
    return blocks

# =============================================================================
# VLLM MODEL WRAPPER
# =============================================================================

class VLLMModel:
    """Wrapper for VLLM model."""
    
    def __init__(self, model_name, temperature=0.7, max_tokens=150):
        from vllm import LLM, SamplingParams
        
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=CONFIG['max_model_len'],
            gpu_memory_utilization=CONFIG['gpu_memory_utilization'],
            disable_log_stats=True
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["\n\n"]
        )
        
    def generate_batch(self, prompts):
        """Generate responses for a batch of prompts."""
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(backstory_file, ess_file, logger):
    """Load and merge backstory and ESS data."""
    logger.info(f"Loading backstories from: {backstory_file}")
    df_backstory = pd.read_csv(backstory_file)
    logger.info(f"✓ Loaded {len(df_backstory)} backstories")
    
    logger.info(f"Loading ESS data from: {ess_file}")
    import pyreadstat
    df_ess, meta = pyreadstat.read_dta(ess_file)
    logger.info(f"✓ Loaded {len(df_ess)} ESS records")
    
    # Select relevant variables
    all_vars = ['idno'] + list(VARIABLES.keys()) + [v for v in COVARIATE_VARS if v not in VARIABLES]
    available_vars = [v for v in all_vars if v in df_ess.columns]
    df_ess_subset = df_ess[available_vars].copy()
    
    # Clean outcome variables
    logger.info("Cleaning outcome variables...")
    for var in VARIABLES.keys():
        if var not in df_ess_subset.columns:
            continue
        df_ess_subset[var] = pd.to_numeric(df_ess_subset[var], errors='coerce')
        scale_min, scale_max = VARIABLES[var]['scale']
        mask_invalid = (df_ess_subset[var] < scale_min) | (df_ess_subset[var] > scale_max)
        df_ess_subset.loc[mask_invalid, var] = np.nan
    
    # Clean covariates
    for var in ['eduyrs']:
        if var in df_ess_subset.columns:
            df_ess_subset[var] = pd.to_numeric(df_ess_subset[var], errors='coerce')
            df_ess_subset.loc[df_ess_subset[var] > 50, var] = np.nan
    
    for var in ['hincfel']:
        if var in df_ess_subset.columns:
            df_ess_subset[var] = pd.to_numeric(df_ess_subset[var], errors='coerce')
            df_ess_subset.loc[df_ess_subset[var] > 4, var] = np.nan
    
    # Merge
    logger.info("Merging backstory and ESS data...")
    df_full = pd.merge(df_backstory, df_ess_subset, on='idno', how='left')
    logger.info(f"✓ Full dataset: {len(df_full)} records")
    
    return df_full

# =============================================================================
# BLOCK 1: RQ1 + RQ2
# =============================================================================

def run_block1(df_full, model, sample_size, seed, logger):
    """
    Block 1: Core Validation
    - RQ1: Individual-level recovery
    - RQ2: Variance compression
    """
    logger.info("=" * 70)
    logger.info("BLOCK 1: CORE VALIDATION (RQ1 + RQ2)")
    logger.info("=" * 70)
    
    # Sample
    random.seed(seed)
    np.random.seed(seed)
    valid_indices = df_full[df_full['backstory'].notna()].index.tolist()
    sample_indices = random.sample(valid_indices, min(sample_size, len(valid_indices)))
    df_sample = df_full.loc[sample_indices].copy().reset_index(drop=True)
    logger.info(f"Sampled {len(df_sample)} respondents (seed={seed})")
    
    # Build prompts for all variables
    prompts = []
    prompt_metadata = []  # (row_idx, var_name)
    
    for var in ANALYSIS_VARS:
        if var not in df_full.columns:
            continue
        for idx, row in df_sample.iterrows():
            if pd.isna(row['backstory']):
                continue
            prompt = PROMPT_1P_STANDARD.format(
                backstory=row['backstory'],
                question=VARIABLES[var]['question']
            )
            prompts.append(prompt)
            prompt_metadata.append((idx, var))
    
    logger.info(f"Total prompts to process: {len(prompts)}")
    logger.info(f"  ({len(df_sample)} respondents × {len(ANALYSIS_VARS)} variables)")
    
    # Generate responses
    logger.info("Generating responses with VLLM (batch processing)...")
    start_time = time.time()
    responses = model.generate_batch(prompts)
    elapsed = time.time() - start_time
    
    logger.info(f"✓ Generated {len(responses)} responses in {elapsed:.1f} seconds")
    logger.info(f"  Speed: {len(responses)/elapsed:.1f} responses/second")
    
    # Parse responses
    logger.info("Parsing responses...")
    for var in ANALYSIS_VARS:
        df_sample[f'silicon_{var}'] = np.nan
    
    success_count = 0
    for (idx, var), response in zip(prompt_metadata, responses):
        scale_min, scale_max = VARIABLES[var]['scale']
        value = parse_response(response, scale_min, scale_max)
        df_sample.loc[idx, f'silicon_{var}'] = value
        if not np.isnan(value):
            success_count += 1
    
    logger.info(f"✓ Parsing success rate: {success_count}/{len(responses)} ({100*success_count/len(responses):.1f}%)")
    
    # Save raw data
    os.makedirs("results", exist_ok=True)
    df_sample.to_csv(f'results/silicon_block1_data_seed{seed}.csv', index=False)
    logger.info(f"✓ Data saved to: results/silicon_block1_data_seed{seed}.csv")
    
    # ==========================================================================
    # RQ1: Individual-Level Recovery
    # ==========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("RQ1: INDIVIDUAL-LEVEL RECOVERY")
    logger.info("=" * 70)
    
    rq1_results = []
    for var in ANALYSIS_VARS:
        silicon_col = f'silicon_{var}'
        if silicon_col not in df_sample.columns or var not in df_sample.columns:
            continue
        
        mask = df_sample[silicon_col].notna() & df_sample[var].notna()
        n = mask.sum()
        if n < 10:
            continue
        
        silicon = df_sample.loc[mask, silicon_col].values
        human = df_sample.loc[mask, var].values
        
        r, p, ci_lower, ci_upper = bootstrap_correlation_ci(
            silicon, human, n_bootstrap=CONFIG['n_bootstrap']
        )
        rho, p_rho = spearmanr(silicon, human)
        
        rq1_results.append({
            'variable': var,
            'domain': VARIABLES[var]['domain'],
            'scale': f"{VARIABLES[var]['scale'][0]}-{VARIABLES[var]['scale'][1]}",
            'n': n,
            'pearson_r': r,
            'pearson_p': p,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'spearman_rho': rho,
            'sig_05': p < 0.05 if not np.isnan(p) else False,
            'r_above_03': abs(r) > 0.3 if not np.isnan(r) else False
        })
    
    rq1_df = pd.DataFrame(rq1_results)
    
    # Display RQ1 results
    logger.info(f"\nRQ1 Results (Individual-Level Correlations):")
    logger.info(f"{'Variable':<12} {'Domain':<25} {'n':<6} {'r':<10} {'95% CI':<20}")
    logger.info("-" * 80)
    
    for _, row in rq1_df.iterrows():
        ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        logger.info(f"{row['variable']:<12} {row['domain']:<25} {row['n']:<6} {row['pearson_r']:>+7.3f}  {ci_str}")
    
    overall_r = rq1_df['pearson_r'].mean()
    logger.info(f"\n→ Mean correlation: r = {overall_r:.3f}")
    
    # ==========================================================================
    # RQ2: Variance Compression
    # ==========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("RQ2: VARIANCE COMPRESSION")
    logger.info("=" * 70)
    
    rq2_results = []
    for var in ANALYSIS_VARS:
        silicon_col = f'silicon_{var}'
        if silicon_col not in df_sample.columns or var not in df_full.columns:
            continue
        
        silicon = df_sample[silicon_col].dropna()
        human_full = df_full[var].dropna()
        
        if len(silicon) < 10:
            continue
        
        silicon_var = silicon.var()
        human_var = human_full.var()
        var_ratio = silicon_var / human_var if human_var > 0 else np.nan
        mean_diff = silicon.mean() - human_full.mean()
        tail_mass = compute_tail_mass(silicon.values)
        
        rq2_results.append({
            'variable': var,
            'domain': VARIABLES[var]['domain'],
            'var_ratio': var_ratio,
            'mean_diff': mean_diff,
            'silicon_sd': silicon.std(),
            'human_sd': human_full.std(),
            'silicon_mean': silicon.mean(),
            'human_mean': human_full.mean(),
            'tail_mass': tail_mass
        })
    
    rq2_df = pd.DataFrame(rq2_results)
    
    # Display RQ2 results
    logger.info(f"\nRQ2 Results (Variance Compression):")
    logger.info(f"{'Variable':<12} {'VR':<12} {'Silicon SD':<12} {'Human SD':<12} {'Mean Diff':<10}")
    logger.info("-" * 65)
    
    for _, row in rq2_df.iterrows():
        logger.info(f"{row['variable']:<12} {row['var_ratio']:>10.1%}   {row['silicon_sd']:>10.2f}   {row['human_sd']:>10.2f}   {row['mean_diff']:>+8.2f}")
    
    overall_vr = rq2_df['var_ratio'].mean()
    logger.info(f"\n→ Mean variance ratio: {overall_vr:.1%}")
    logger.info("  (Values < 100% indicate variance compression)")
    
    # Save results
    rq1_df.to_csv('results/silicon_block1_rq1_results.csv', index=False)
    rq2_df.to_csv('results/silicon_block1_rq2_results.csv', index=False)
    logger.info("✓ RQ1 results saved to: results/silicon_block1_rq1_results.csv")
    logger.info("✓ RQ2 results saved to: results/silicon_block1_rq2_results.csv")
    
    return df_sample, rq1_df, rq2_df

# =============================================================================
# BLOCK 2: RQ3 + RQ4 + RQ5 + RQ6
# =============================================================================

def run_block2(df_sample, df_full, model, logger):
    """
    Block 2: Mechanism Exploration
    - RQ3: Prompt framing effects
    - RQ4: Backstory enrichment
    - RQ5: Covariate structure
    - RQ6: Format sensitivity
    """
    logger.info("=" * 70)
    logger.info("BLOCK 2: MECHANISM EXPLORATION (RQ3-RQ6)")
    logger.info("=" * 70)
    
    # ==========================================================================
    # RQ3: Prompt Framing Effects
    # ==========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("RQ3: PROMPT FRAMING EFFECTS")
    logger.info("=" * 70)
    
    RQ3_VARS = ['stfdem', 'happy', 'imueclt', 'ppltrst', 'trstprl']
    
    # Generate 3P responses
    prompts_3p = []
    metadata_3p = []
    for var in RQ3_VARS:
        for idx, row in df_sample.iterrows():
            if pd.isna(row['backstory']):
                continue
            prompt = PROMPT_3P.format(
                backstory=row['backstory'],
                question=VARIABLES[var]['question']
            )
            prompts_3p.append(prompt)
            metadata_3p.append((idx, var))
    
    logger.info(f"Generating 3P responses ({len(prompts_3p)} prompts)...")
    responses_3p = model.generate_batch(prompts_3p)
    
    for var in RQ3_VARS:
        df_sample[f'silicon_3p_{var}'] = np.nan
    for (idx, var), response in zip(metadata_3p, responses_3p):
        scale_min, scale_max = VARIABLES[var]['scale']
        df_sample.loc[idx, f'silicon_3p_{var}'] = parse_response(response, scale_min, scale_max)
    
    # Generate Idiosyncratic responses
    prompts_idio = []
    metadata_idio = []
    for var in RQ3_VARS:
        for idx, row in df_sample.iterrows():
            if pd.isna(row['backstory']):
                continue
            prompt = PROMPT_1P_IDIOSYNCRATIC.format(
                backstory=row['backstory'],
                question=VARIABLES[var]['question']
            )
            prompts_idio.append(prompt)
            metadata_idio.append((idx, var))
    
    logger.info(f"Generating Idiosyncratic responses ({len(prompts_idio)} prompts)...")
    responses_idio = model.generate_batch(prompts_idio)
    
    for var in RQ3_VARS:
        df_sample[f'silicon_idio_{var}'] = np.nan
    for (idx, var), response in zip(metadata_idio, responses_idio):
        scale_min, scale_max = VARIABLES[var]['scale']
        df_sample.loc[idx, f'silicon_idio_{var}'] = parse_response(response, scale_min, scale_max)
    
    # Analyze RQ3
    rq3_results = []
    for var in RQ3_VARS:
        human_var = df_full[var].dropna().var()
        
        for condition, prefix in [('1P_Standard', 'silicon'), ('3P_Simulation', 'silicon_3p'), ('1P_Idiosyncratic', 'silicon_idio')]:
            col = f'{prefix}_{var}'
            if col not in df_sample.columns:
                continue
            
            silicon = df_sample[col].dropna()
            mask = df_sample[col].notna() & df_sample[var].notna()
            n = mask.sum()
            
            r, p = pearsonr(df_sample.loc[mask, col], df_sample.loc[mask, var]) if n >= 10 else (np.nan, np.nan)
            var_ratio = silicon.var() / human_var if human_var > 0 else np.nan
            
            rq3_results.append({
                'variable': var, 'condition': condition,
                'pearson_r': r, 'pearson_p': p, 'var_ratio': var_ratio,
                'n': n, 'mean': silicon.mean(), 'std': silicon.std()
            })
    
    rq3_df = pd.DataFrame(rq3_results)
    
    logger.info(f"\nRQ3 Results:")
    logger.info(f"{'Variable':<10} {'Condition':<18} {'r':<10} {'VarRatio':<12} {'n':<8}")
    logger.info("-" * 65)
    for _, row in rq3_df.iterrows():
        sig = '*' if row['pearson_p'] < 0.05 else ''
        r_str = f"{row['pearson_r']:.3f}" if not np.isnan(row['pearson_r']) else "N/A"
        vr_str = f"{row['var_ratio']:.1%}" if not np.isnan(row['var_ratio']) else "N/A"
        logger.info(f"{row['variable']:<10} {row['condition']:<18} {r_str:<10} {vr_str:<12} {row['n']:<8}")
    
    # ==========================================================================
    # RQ4: Backstory Enrichment
    # ==========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("RQ4: BACKSTORY ENRICHMENT")
    logger.info("=" * 70)
    
    # Decompose backstories
    logger.info("Decomposing backstories...")
    decompose_prompts = []
    for idx, row in df_sample.iterrows():
        if pd.notna(row['backstory']):
            decompose_prompts.append(DECOMPOSE_PROMPT.format(backstory=row['backstory']))
        else:
            decompose_prompts.append("")
    
    # Filter out empty prompts
    valid_decompose = [(i, p) for i, p in enumerate(decompose_prompts) if p]
    if valid_decompose:
        indices, prompts = zip(*valid_decompose)
        decompose_responses = model.generate_batch(list(prompts))
        
        df_sample['block1'] = ''
        df_sample['block2'] = ''
        df_sample['block3'] = ''
        df_sample['block4'] = ''
        
        for idx, response in zip(indices, decompose_responses):
            blocks = parse_decomposition_output(response)
            df_sample.loc[idx, 'block1'] = blocks['block1']
            df_sample.loc[idx, 'block2'] = blocks['block2']
            df_sample.loc[idx, 'block3'] = blocks['block3']
            df_sample.loc[idx, 'block4'] = blocks['block4']
    
    # Build cumulative backstories and generate responses
    def build_cumulative_backstory(row, level):
        parts = []
        if level >= 1 and row.get('block1', ''):
            parts.append(str(row['block1']))
        if level >= 2 and row.get('block2', ''):
            parts.append(str(row['block2']))
        if level >= 3 and row.get('block3', ''):
            parts.append(str(row['block3']))
        if level >= 4 and row.get('block4', ''):
            parts.append(str(row['block4']))
        return ' '.join(parts)
    
    LEVEL_NAMES = ['Demographics', 'Socioeconomic', 'Ideology', 'Incorporation']
    RQ4_VARS = ['stfdem', 'happy', 'imueclt', 'ppltrst', 'trstprl', 'stflife', 'trstplt']
    
    for level in [1, 2, 3, 4]:
        logger.info(f"Generating Level {level} ({LEVEL_NAMES[level-1]}) responses...")
        prompts_level = []
        metadata_level = []
        
        for var in RQ4_VARS:
            for idx, row in df_sample.iterrows():
                cum_backstory = build_cumulative_backstory(row, level)
                if cum_backstory and len(cum_backstory) >= 10:
                    prompt = PROMPT_1P_STANDARD.format(
                        backstory=cum_backstory,
                        question=VARIABLES[var]['question']
                    )
                    prompts_level.append(prompt)
                    metadata_level.append((idx, var))
        
        if prompts_level:
            responses_level = model.generate_batch(prompts_level)
            for var in RQ4_VARS:
                if f'silicon_L{level}_{var}' not in df_sample.columns:
                    df_sample[f'silicon_L{level}_{var}'] = np.nan
            
            for (idx, var), response in zip(metadata_level, responses_level):
                scale_min, scale_max = VARIABLES[var]['scale']
                df_sample.loc[idx, f'silicon_L{level}_{var}'] = parse_response(response, scale_min, scale_max)
    
    # Analyze RQ4
    rq4_results = []
    for var in RQ4_VARS:
        if var not in df_full.columns:
            continue
        human_var = df_full[var].dropna().var()
        
        for level in [1, 2, 3, 4]:
            col = f'silicon_L{level}_{var}'
            if col not in df_sample.columns:
                continue
            
            silicon = df_sample[col].dropna()
            mask = df_sample[col].notna() & df_sample[var].notna()
            n = mask.sum()
            
            r, p = pearsonr(df_sample.loc[mask, col], df_sample.loc[mask, var]) if n >= 10 else (np.nan, np.nan)
            var_ratio = silicon.var() / human_var if human_var > 0 else np.nan
            
            rq4_results.append({
                'variable': var, 'level': level, 'level_name': LEVEL_NAMES[level-1],
                'pearson_r': r, 'pearson_p': p, 'var_ratio': var_ratio, 'n': n
            })
    
    rq4_df = pd.DataFrame(rq4_results)
    
    logger.info(f"\nRQ4 Results by Level:")
    for level in [1, 2, 3, 4]:
        level_data = rq4_df[rq4_df['level'] == level]
        if len(level_data) > 0:
            mean_r = level_data['pearson_r'].mean()
            mean_vr = level_data['var_ratio'].mean()
            r_str = f"{mean_r:.3f}" if not np.isnan(mean_r) else "N/A"
            vr_str = f"{mean_vr:.1%}" if not np.isnan(mean_vr) else "N/A"
            logger.info(f"  L{level} ({LEVEL_NAMES[level-1]}): Mean r = {r_str}, Mean VR = {vr_str}")
    
    # ==========================================================================
    # RQ5: Covariate Structure
    # ==========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("RQ5: COVARIATE STRUCTURE")
    logger.info("=" * 70)
    
    df_reg = df_sample.copy()
    df_reg['female'] = (df_reg['gndr'] == 2).astype(float)
    df_reg['age'] = df_reg['agea']
    df_reg['born_uk'] = (df_reg['brncntr'] == 1).astype(float)
    df_reg['education'] = df_reg['eduyrs']
    df_reg['ideology'] = df_reg['lrscale']
    
    predictors = ['female', 'age', 'born_uk', 'ideology', 'education']
    
    rq5_results = []
    for var in ANALYSIS_VARS[:20]:  # Top 20 variables
        silicon_col = f'silicon_{var}'
        if silicon_col not in df_reg.columns or var not in df_reg.columns:
            continue
        
        reg_vars = predictors + [var, silicon_col]
        reg_data = df_reg[reg_vars].dropna()
        
        if len(reg_data) < 30:
            continue
        
        try:
            model_human = ols(f"{var} ~ " + " + ".join(predictors), data=reg_data).fit()
            model_silicon = ols(f"{silicon_col} ~ " + " + ".join(predictors), data=reg_data).fit()
            
            rq5_results.append({
                'variable': var, 'domain': VARIABLES[var]['domain'],
                'r2_human': model_human.rsquared, 'r2_silicon': model_silicon.rsquared,
                'r2_diff': model_silicon.rsquared - model_human.rsquared,
                'r2_ratio': model_silicon.rsquared / model_human.rsquared if model_human.rsquared > 0 else np.nan
            })
        except:
            continue
    
    rq5_df = pd.DataFrame(rq5_results)
    
    if len(rq5_df) > 0:
        avg_ratio = rq5_df['r2_ratio'].mean()
        logger.info(f"\nRQ5 Results:")
        logger.info(f"Mean R² (Human): {rq5_df['r2_human'].mean():.3f}")
        logger.info(f"Mean R² (Silicon): {rq5_df['r2_silicon'].mean():.3f}")
        logger.info(f"Mean R² Ratio: {avg_ratio:.1f}x")
    
    # ==========================================================================
    # RQ6: Format Sensitivity
    # ==========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("RQ6: FORMAT SENSITIVITY")
    logger.info("=" * 70)
    
    RQ6_VARS = ['stfdem', 'happy']
    RQ6_QUESTIONS = {
        'standard': {
            'stfdem': 'How satisfied are you with democracy? Answer from 0 (extremely dissatisfied) to 10 (extremely satisfied). Reply with just the number.',
            'happy': 'How happy are you? Answer from 0 (extremely unhappy) to 10 (extremely happy). Reply with just the number.'
        },
        'reversed': {
            'stfdem': 'How satisfied are you with democracy? Answer from 10 (extremely satisfied) to 0 (extremely dissatisfied). Reply with just the number.',
            'happy': 'How happy are you? Answer from 10 (extremely happy) to 0 (extremely unhappy). Reply with just the number.'
        }
    }
    
    for fmt in ['standard', 'reversed']:
        prompts_fmt = []
        metadata_fmt = []
        
        for var in RQ6_VARS:
            for idx, row in df_sample.iterrows():
                if pd.isna(row['backstory']):
                    continue
                prompt = PROMPT_1P_STANDARD.format(
                    backstory=row['backstory'],
                    question=RQ6_QUESTIONS[fmt][var]
                )
                prompts_fmt.append(prompt)
                metadata_fmt.append((idx, var, fmt))
        
        logger.info(f"Generating {fmt} format responses ({len(prompts_fmt)} prompts)...")
        responses_fmt = model.generate_batch(prompts_fmt)
        
        for var in RQ6_VARS:
            df_sample[f'format_{fmt}_{var}'] = np.nan
        
        for (idx, var, fmt_type), response in zip(metadata_fmt, responses_fmt):
            value = parse_response(response, 0, 10)
            if fmt_type == 'reversed' and not np.isnan(value):
                value = 10 - value  # Reverse the scale back
            df_sample.loc[idx, f'format_{fmt_type}_{var}'] = value
    
    # Analyze RQ6
    rq6_results = []
    for var in RQ6_VARS:
        std_col = f'format_standard_{var}'
        rev_col = f'format_reversed_{var}'
        
        if std_col not in df_sample.columns or rev_col not in df_sample.columns:
            continue
        
        mask = df_sample[std_col].notna() & df_sample[rev_col].notna()
        n = mask.sum()
        
        if n < 10:
            continue
        
        std_vals = df_sample.loc[mask, std_col]
        rev_vals = df_sample.loc[mask, rev_col]
        diff = (std_vals - rev_vals).abs()
        
        r_formats, p_formats = pearsonr(std_vals, rev_vals)
        flip_rate = (diff > 2).mean()
        
        rq6_results.append({
            'variable': var, 'n_paired': n,
            'flip_rate': flip_rate,
            'r_between_formats': r_formats,
            'p_between_formats': p_formats
        })
    
    rq6_df = pd.DataFrame(rq6_results)
    
    if len(rq6_df) > 0:
        logger.info(f"\nRQ6 Results:")
        for _, row in rq6_df.iterrows():
            logger.info(f"  {row['variable']}: Flip rate = {row['flip_rate']:.1%}, r = {row['r_between_formats']:.3f}")
    
    # Save results
    df_sample.to_csv('results/silicon_block2_data.csv', index=False)
    rq3_df.to_csv('results/silicon_block2_rq3_results.csv', index=False)
    rq4_df.to_csv('results/silicon_block2_rq4_results.csv', index=False)
    rq5_df.to_csv('results/silicon_block2_rq5_results.csv', index=False)
    rq6_df.to_csv('results/silicon_block2_rq6_results.csv', index=False)
    
    logger.info("✓ Block 2 results saved")
    
    return df_sample, rq3_df, rq4_df, rq5_df, rq6_df

# =============================================================================
# BLOCK 3: RQ7
# =============================================================================

def run_block3(df_sample, df_full, logger):
    """
    Block 3: Group Heterogeneity (RQ7a/b/c)
    """
    logger.info("=" * 70)
    logger.info("BLOCK 3: SUBGROUP HETEROGENEITY (RQ7)")
    logger.info("=" * 70)
    
    # Define groups
    df_sample['edu_group'] = pd.cut(df_sample['eduyrs'], bins=[0, 12, 100], labels=['Low Education', 'High Education'])
    df_sample['income_group'] = df_sample['hincfel'].apply(
        lambda x: 'Low Income' if x >= 3 else 'High Income' if pd.notna(x) else np.nan
    )
    df_sample['gender_group'] = df_sample['gndr'].apply(
        lambda x: 'Female' if x == 2 else 'Male' if x == 1 else np.nan
    )
    df_sample['minority_group'] = df_sample['brncntr'].apply(
        lambda x: 'Non-UK Born' if x == 2 else 'UK Born' if x == 1 else np.nan
    )
    
    GROUP_DEFINITIONS = {
        'edu_group': ('Low Education', 'High Education'),
        'income_group': ('Low Income', 'High Income'),
        'gender_group': ('Female', 'Male'),
        'minority_group': ('Non-UK Born', 'UK Born')
    }
    
    # ==========================================================================
    # RQ7a: Disadvantaged Group Variance Compression
    # ==========================================================================
    logger.info("")
    logger.info("RQ7a: DISADVANTAGED GROUP VARIANCE COMPRESSION")
    
    h7a_results = []
    for group_var, (disadvantaged, advantaged) in GROUP_DEFINITIONS.items():
        for var in ANALYSIS_VARS[:20]:  # Top 20 variables
            silicon_col = f'silicon_{var}'
            if silicon_col not in df_sample.columns or var not in df_full.columns:
                continue
            
            human_full_var = df_full[var].dropna().var()
            
            for group_val, group_status in [(disadvantaged, 'Disadvantaged'), (advantaged, 'Advantaged')]:
                mask = (df_sample[group_var] == group_val) & df_sample[silicon_col].notna() & df_sample[var].notna()
                n = mask.sum()
                
                if n < 10:
                    continue
                
                silicon = df_sample.loc[mask, silicon_col]
                human = df_sample.loc[mask, var]
                var_ratio = silicon.var() / human_full_var if human_full_var > 0 else np.nan
                
                h7a_results.append({
                    'group_type': group_var.replace('_group', ''),
                    'group_status': group_status,
                    'variable': var,
                    'var_ratio_pop': var_ratio,
                    'n': n
                })
    
    h7a_df = pd.DataFrame(h7a_results)
    
    if len(h7a_df) > 0:
        logger.info(f"\nRQ7a Results by Group Type:")
        for group_type in ['edu', 'income', 'gender', 'minority']:
            disadv_data = h7a_df[(h7a_df['group_type']==group_type) & (h7a_df['group_status']=='Disadvantaged')]
            adv_data = h7a_df[(h7a_df['group_type']==group_type) & (h7a_df['group_status']=='Advantaged')]
            
            if len(disadv_data) > 0 and len(adv_data) > 0:
                disadv_vr = disadv_data['var_ratio_pop'].mean()
                adv_vr = adv_data['var_ratio_pop'].mean()
                diff = disadv_vr - adv_vr
                logger.info(f"  {group_type}: Disadvantaged VR = {disadv_vr:.1%}, Advantaged VR = {adv_vr:.1%}, Δ = {diff:+.1%}")
    
    # ==========================================================================
    # RQ7c: Issue Sensitivity
    # ==========================================================================
    logger.info("")
    logger.info("RQ7c: ISSUE SENSITIVITY")
    
    h7c_results = []
    for var in ANALYSIS_VARS[:20]:
        silicon_col = f'silicon_{var}'
        if silicon_col not in df_sample.columns or var not in df_full.columns:
            continue
        
        human_sd = df_full[var].dropna().std()
        silicon = df_sample[silicon_col].dropna()
        
        if len(silicon) < 10:
            continue
        
        human_var = df_full[var].dropna().var()
        var_ratio = silicon.var() / human_var if human_var > 0 else np.nan
        
        h7c_results.append({
            'variable': var,
            'human_sd': human_sd,
            'var_ratio': var_ratio
        })
    
    h7c_df = pd.DataFrame(h7c_results)
    
    if len(h7c_df) > 0:
        corr_pearson, p_pearson = pearsonr(h7c_df['human_sd'], h7c_df['var_ratio'])
        logger.info(f"\nRQ7c Results:")
        logger.info(f"  Correlation (Human SD vs VR): r = {corr_pearson:.3f}, p = {p_pearson:.4f}")
        logger.info(f"  Interpretation: {'Supported' if corr_pearson < -0.15 and p_pearson < 0.05 else 'Not supported'}")
    
    # Save results
    h7a_df.to_csv('results/silicon_block3_h7a_results.csv', index=False)
    h7c_df.to_csv('results/silicon_block3_h7c_results.csv', index=False)
    
    logger.info("✓ Block 3 results saved")
    
    return h7a_df, h7c_df

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Silicon Sampling Validation Framework')
    parser.add_argument('--block', type=str, default='all', choices=['1', '2', '3', 'all'],
                        help='Which block to run (1, 2, 3, or all)')
    parser.add_argument('--sample_size', type=int, default=1000, help='Sample size')
    parser.add_argument('--seed', type=int, default=888, help='Random seed')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='Model name')
    parser.add_argument('--backstory_file', type=str, default='data/ess_uk_with_backstories_v1.csv')
    parser.add_argument('--ess_file', type=str, default='data/ESS9e03_2_GBonly.dta')
    
    args = parser.parse_args()
    
    CONFIG['sample_size'] = args.sample_size
    CONFIG['model_name'] = args.model
    
    # Setup logging
    logger = setup_logging(f"block{args.block}")
    
    logger.info("=" * 70)
    logger.info("SILICON SAMPLING VALIDATION FRAMEWORK")
    logger.info(f"Block: {args.block} | Sample: {args.sample_size} | Model: {args.model}")
    logger.info("=" * 70)
    
    # Load data
    logger.info("")
    logger.info("LOADING DATA")
    logger.info("=" * 70)
    df_full = load_data(args.backstory_file, args.ess_file, logger)
    
    # Initialize model
    logger.info("")
    logger.info("INITIALIZING MODEL")
    logger.info("=" * 70)
    model = VLLMModel(
        model_name=args.model,
        temperature=CONFIG['temperature'],
        max_tokens=CONFIG['max_tokens']
    )
    logger.info("✓ Model loaded successfully!")
    
    # Run blocks
    if args.block in ['1', 'all']:
        df_sample, rq1_df, rq2_df = run_block1(df_full, model, args.sample_size, args.seed, logger)
    
    if args.block in ['2', 'all']:
        if args.block == '2':
            # Load Block 1 data if running Block 2 separately
            df_sample = pd.read_csv(f'results/silicon_block1_data_seed{args.seed}.csv')
        df_sample, rq3_df, rq4_df, rq5_df, rq6_df = run_block2(df_sample, df_full, model, logger)
    
    if args.block in ['3', 'all']:
        if args.block == '3':
            # Load Block 2 data if running Block 3 separately
            df_sample = pd.read_csv('results/silicon_block2_data.csv')
        h7a_df, h7c_df = run_block3(df_sample, df_full, logger)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
