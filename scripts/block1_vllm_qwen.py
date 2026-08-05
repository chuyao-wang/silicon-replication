"""
Silicon Sampling Validation Framework - Block 1: Core Validation
=================================================================
WINSTON HPC + VLLM + Qwen VERSION (Quick Test: n=50)

Key advantages over API-based approach:
1. VLLM batch inference is MUCH faster (processes all prompts in parallel)
2. No API costs - runs locally on GPU
3. Fully reproducible with fixed model weights

Prerequisites:
- conda activate silicon_vllm
- GPU partition on Winston (--partition=winston-gpu --gres=gpu:1)

Usage:
    python block1_vllm_qwen.py --backstory_file data/ess_uk_with_backstories_v1.csv \
                               --ess_file data/ESS9e03_2_GBonly.dta \
                               --output_dir results/ \
                               --sample_size 50

Author: Chuyao Wang
Date: February 2026
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
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'sample_size': 50,              # Quick test sample size (change to 800 for full run)
    'random_seeds': [888],          # Single seed for quick test
    'temperature': 0.7,
    'n_bootstrap': 500,             # Reduced for quick testing (use 1000 for full run)
    'alpha': 0.05,
    
    # VLLM configuration
    'model_name': "Qwen/Qwen2.5-7B-Instruct",  # Good balance of speed/quality
    'max_tokens': 50,               # Short responses expected (just a number)
    'gpu_memory_utilization': 0.85,
    'max_model_len': 4096,
}

# =============================================================================
# Variable Definitions (10 items for quick test)
# =============================================================================

VARIABLES = {
    # Political Attitudes (2 items)
    'stfdem': {
        'domain': 'Political Attitudes', 
        'question': 'How satisfied with the way democracy works in country (0=extremely dissatisfied, 10=extremely satisfied)', 
        'scale': (0, 10)
    },
    'polintr': {
        'domain': 'Political Attitudes', 
        'question': 'How interested in politics (1=very interested, 4=not at all interested)', 
        'scale': (1, 4)
    },
    
    # Institutional Trust (2 items)
    'trstprl': {
        'domain': 'Institutional Trust', 
        'question': "Trust in country's parliament (0=no trust, 10=complete trust)", 
        'scale': (0, 10)
    },
    'trstplt': {
        'domain': 'Institutional Trust', 
        'question': 'Trust in politicians (0=no trust, 10=complete trust)', 
        'scale': (0, 10)
    },
    
    # Social Trust (2 items)
    'ppltrst': {
        'domain': 'Social Trust', 
        'question': "Most people can be trusted or you can't be too careful (0=can't be too careful, 10=most people can be trusted)", 
        'scale': (0, 10)
    },
    'pplfair': {
        'domain': 'Social Trust', 
        'question': 'Most people try to take advantage of you or try to be fair (0=take advantage, 10=try to be fair)', 
        'scale': (0, 10)
    },
    
    # Subjective Well-being (2 items)
    'happy': {
        'domain': 'Subjective Well-being', 
        'question': 'How happy are you (0=extremely unhappy, 10=extremely happy)', 
        'scale': (0, 10)
    },
    'stflife': {
        'domain': 'Subjective Well-being', 
        'question': 'How satisfied with life as a whole (0=extremely dissatisfied, 10=extremely satisfied)', 
        'scale': (0, 10)
    },
    
    # Immigration Attitudes (2 items)
    'imueclt': {
        'domain': 'Immigration Attitudes', 
        'question': "Country's cultural life undermined or enriched by immigrants (0=undermined, 10=enriched)", 
        'scale': (0, 10)
    },
    'imwbcnt': {
        'domain': 'Immigration Attitudes', 
        'question': 'Immigrants make country worse or better place to live (0=worse, 10=better)', 
        'scale': (0, 10)
    },
}

OUTCOME_VARS = list(VARIABLES.keys())
ANALYSIS_VARS = OUTCOME_VARS

# Covariates for data merging
COVARIATE_VARS = ['gndr', 'agea', 'brncntr', 'ctzcntr', 'eduyrs', 
                  'hincfel', 'lrscale', 'vote', 'blgetmg']

# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir):
    """Configure logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'block1_vllm_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
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
# VLLM Model Initialization
# =============================================================================

def initialize_vllm(model_name, gpu_memory_utilization=0.85, max_model_len=4096):
    """
    Initialize VLLM model for batch inference.
    This loads the model into GPU memory once, then can process many prompts quickly.
    """
    from vllm import LLM
    
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"GPU memory utilization: {gpu_memory_utilization}")
    print(f"This may take a few minutes on first run (downloading model)...")
    print(f"{'='*60}\n")
    
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    
    print("✓ Model loaded successfully!")
    return llm

# =============================================================================
# Prompt Template
# =============================================================================

PROMPT_TEMPLATE = """Adopt the following persona and answer only based on it. Do not invent details beyond the provided attributes.

{backstory}

Question: {question}

Answer with ONLY a single number within the valid range. Do not explain."""


def create_prompt(backstory, question):
    """Create a prompt for the model."""
    return PROMPT_TEMPLATE.format(backstory=backstory, question=question)

# =============================================================================
# Batch Response Generation (VLLM - Much Faster!)
# =============================================================================

def generate_batch_responses(llm, prompts, temperature=0.7, max_tokens=50):
    """
    Generate responses for a batch of prompts using VLLM.
    
    This is MUCH faster than sequential API calls because:
    1. All prompts are processed in parallel on GPU
    2. VLLM uses PagedAttention for efficient memory management
    3. Continuous batching maximizes GPU utilization
    """
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95,
    )
    
    # VLLM processes all prompts in parallel - this is the key speedup!
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract text from outputs
    responses = []
    for output in outputs:
        text = output.outputs[0].text.strip()
        responses.append(text)
    
    return responses


def extract_number_from_response(text, scale):
    """Extract a number from model response and validate against scale."""
    scale_min, scale_max = scale
    
    # Clean the text
    text = text.strip()
    
    # Try direct conversion first (ideal case: response is just a number)
    try:
        value = float(text.split()[0])  # Take first token
        if scale_min <= value <= scale_max:
            return value
    except (ValueError, IndexError):
        pass
    
    # Try to find numbers in the response
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    
    if numbers:
        for num_str in numbers:
            value = float(num_str)
            if scale_min <= value <= scale_max:
                return value
    
    return np.nan


def collect_silicon_responses_vllm(df_sample, llm, logger=None):
    """
    Collect silicon responses using VLLM batch inference.
    
    Process:
    1. Build all prompts at once (n_samples × n_variables)
    2. Send to VLLM for batch processing (single GPU call!)
    3. Parse all responses
    
    This is MUCH faster than calling an API for each prompt individually.
    """
    df = df_sample.copy()
    
    # Initialize silicon columns
    for var in VARIABLES.keys():
        df[f'silicon_{var}'] = np.nan
    
    # Build all prompts at once
    all_prompts = []
    prompt_metadata = []  # Track (row_idx, var_name, scale)
    
    if logger:
        logger.info("Building prompts...")
    
    for var, info in VARIABLES.items():
        for idx, row in df.iterrows():
            backstory = row['backstory']
            if pd.isna(backstory) or str(backstory).strip() == "":
                continue
            
            prompt = create_prompt(backstory, info['question'])
            all_prompts.append(prompt)
            prompt_metadata.append((idx, var, info['scale']))
    
    total_prompts = len(all_prompts)
    if logger:
        logger.info(f"Total prompts to process: {total_prompts}")
        logger.info(f"  ({len(df)} respondents × {len(VARIABLES)} variables)")
    
    if total_prompts == 0:
        if logger:
            logger.warning("No valid prompts to process!")
        return df
    
    # Generate all responses in ONE batch (VLLM handles parallelization)
    if logger:
        logger.info("\nGenerating responses with VLLM (batch processing)...")
        logger.info("This processes ALL prompts in parallel on GPU...")
    
    start_time = time.time()
    responses = generate_batch_responses(
        llm, 
        all_prompts, 
        temperature=CONFIG['temperature'],
        max_tokens=CONFIG['max_tokens']
    )
    elapsed = time.time() - start_time
    
    if logger:
        logger.info(f"\n✓ Generated {len(responses)} responses in {elapsed:.1f} seconds")
        logger.info(f"  Speed: {len(responses)/elapsed:.1f} responses/second")
        if elapsed > 0:
            logger.info(f"  (Compare to ~1-2 responses/sec with API calls)")
    
    # Parse responses and update dataframe
    if logger:
        logger.info("\nParsing responses...")
    
    successful = 0
    for i, (idx, var, scale) in enumerate(prompt_metadata):
        value = extract_number_from_response(responses[i], scale)
        df.loc[idx, f'silicon_{var}'] = value
        if not np.isnan(value):
            successful += 1
    
    if logger:
        success_rate = 100 * successful / total_prompts
        logger.info(f"✓ Parsing success rate: {successful}/{total_prompts} ({success_rate:.1f}%)")
    
    return df

# =============================================================================
# Statistical Analysis Functions
# =============================================================================

def bootstrap_correlation_ci(x, y, n_bootstrap=500, ci=0.95):
    """Compute bootstrap CI for Pearson correlation."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = np.array(x)[mask], np.array(y)[mask]
    
    if len(x) < 10:
        return np.nan, np.nan, np.nan, np.nan
    
    boot_corrs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(x), size=len(x), replace=True)
        try:
            r, _ = pearsonr(x[idx], y[idx])
            boot_corrs.append(r)
        except:
            continue
    
    if len(boot_corrs) < n_bootstrap * 0.5:
        return np.nan, np.nan, np.nan, np.nan
    
    r, p = pearsonr(x, y)
    lower = np.percentile(boot_corrs, (1-ci)/2 * 100)
    upper = np.percentile(boot_corrs, (1+ci)/2 * 100)
    return r, p, lower, upper


def compute_variance_ratio(silicon_values, human_values):
    """Compute variance ratio (silicon variance / human variance)."""
    silicon = np.array(silicon_values)
    human = np.array(human_values)
    
    silicon = silicon[~np.isnan(silicon)]
    human = human[~np.isnan(human)]
    
    if len(silicon) < 5 or len(human) < 5:
        return np.nan
    
    silicon_var = np.var(silicon, ddof=1)
    human_var = np.var(human, ddof=1)
    
    if human_var == 0:
        return np.nan
    
    return silicon_var / human_var

# =============================================================================
# Main Function
# =============================================================================

def main(args):
    """Main execution function."""
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    logger.info("=" * 70)
    logger.info("SILICON SAMPLING - BLOCK 1: CORE VALIDATION")
    logger.info("VLLM + Qwen Version (Quick Test)")
    logger.info("=" * 70)
    
    # Update config from args
    CONFIG['sample_size'] = args.sample_size
    if args.model:
        CONFIG['model_name'] = args.model
    
    # Print configuration
    logger.info(f"\nConfiguration:")
    logger.info(f"  Sample size: {CONFIG['sample_size']}")
    logger.info(f"  Model: {CONFIG['model_name']}")
    logger.info(f"  Variables: {len(VARIABLES)} items")
    logger.info(f"  Temperature: {CONFIG['temperature']}")
    logger.info(f"  Random seeds: {CONFIG['random_seeds']}")
    
    # ==========================================================================
    # Initialize VLLM
    # ==========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: LOADING MODEL")
    logger.info("=" * 70)
    
    llm = initialize_vllm(
        CONFIG['model_name'], 
        CONFIG['gpu_memory_utilization'],
        CONFIG['max_model_len']
    )
    
    # ==========================================================================
    # Load Data
    # ==========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: LOADING DATA")
    logger.info("=" * 70)
    
    logger.info(f"\nLoading backstories from: {args.backstory_file}")
    df_backstory = pd.read_csv(args.backstory_file)
    logger.info(f"✓ Loaded {len(df_backstory)} backstories")
    
    logger.info(f"\nLoading ESS data from: {args.ess_file}")
    import pyreadstat
    df_ess, meta = pyreadstat.read_dta(args.ess_file)
    logger.info(f"✓ Loaded {len(df_ess)} ESS records")
    
    # Prepare ESS data
    ALL_VARS = ['idno'] + OUTCOME_VARS + [v for v in COVARIATE_VARS if v not in OUTCOME_VARS]
    available_vars = [v for v in ALL_VARS if v in df_ess.columns]
    df_ess_subset = df_ess[available_vars].copy()
    
    # Clean outcome variables
    logger.info("\nCleaning outcome variables...")
    for var in OUTCOME_VARS:
        if var not in df_ess_subset.columns:
            logger.warning(f"  Variable {var} not found in ESS data")
            continue
        df_ess_subset[var] = pd.to_numeric(df_ess_subset[var], errors='coerce')
        scale_min, scale_max = VARIABLES[var]['scale']
        mask_invalid = (df_ess_subset[var] < scale_min) | (df_ess_subset[var] > scale_max)
        n_invalid = mask_invalid.sum()
        if n_invalid > 0:
            df_ess_subset.loc[mask_invalid, var] = np.nan
            logger.info(f"  {var}: {n_invalid} invalid values set to NaN")
    
    # Merge data
    logger.info("\nMerging backstory and ESS data...")
    df_full = pd.merge(df_backstory, df_ess_subset, on='idno', how='left')
    logger.info(f"✓ Full dataset: {len(df_full)} records")
    
    # ==========================================================================
    # Run Data Collection
    # ==========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: COLLECTING SILICON RESPONSES")
    logger.info("=" * 70)
    
    all_results = {}
    
    for seed_idx, seed in enumerate(CONFIG['random_seeds']):
        logger.info(f"\n{'='*60}")
        logger.info(f"REPLICATION {seed_idx+1}/{len(CONFIG['random_seeds'])} (seed={seed})")
        logger.info(f"{'='*60}")
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Sample respondents
        valid_indices = df_full[df_full['backstory'].notna()].index.tolist()
        n_sample = min(CONFIG['sample_size'], len(valid_indices))
        sample_indices = random.sample(valid_indices, n_sample)
        df_sample = df_full.loc[sample_indices].copy().reset_index(drop=True)
        
        logger.info(f"Sampled {len(df_sample)} respondents")
        
        # Collect responses using VLLM batch inference
        df_result = collect_silicon_responses_vllm(df_sample, llm, logger=logger)
        
        all_results[seed] = df_result
        
        # Save intermediate results
        output_file = os.path.join(args.output_dir, f'silicon_block1_data_seed{seed}.csv')
        df_result.to_csv(output_file, index=False)
        logger.info(f"✓ Data saved to: {output_file}")
    
    # ==========================================================================
    # RQ1: Individual-Level Recovery Analysis
    # ==========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: RQ1 - INDIVIDUAL-LEVEL RECOVERY")
    logger.info("=" * 70)
    
    rq1_results = []
    
    for seed, df in all_results.items():
        for var in ANALYSIS_VARS:
            silicon_col = f'silicon_{var}'
            
            if silicon_col not in df.columns or var not in df.columns:
                continue
                
            mask = df[silicon_col].notna() & df[var].notna()
            n = mask.sum()
            
            if n < 5:
                logger.warning(f"  {var}: insufficient data (n={n})")
                continue
            
            silicon = df.loc[mask, silicon_col].values
            human = df.loc[mask, var].values
            
            r, p, ci_lower, ci_upper = bootstrap_correlation_ci(
                silicon, human, n_bootstrap=CONFIG['n_bootstrap']
            )
            
            rq1_results.append({
                'seed': seed,
                'variable': var,
                'domain': VARIABLES[var]['domain'],
                'n': n,
                'pearson_r': r,
                'pearson_p': p,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
            })
    
    rq1_df = pd.DataFrame(rq1_results)
    
    # Print RQ1 summary
    if len(rq1_df) > 0:
        logger.info(f"\nRQ1 Results (Individual-Level Correlations):")
        logger.info(f"{'Variable':<12} {'Domain':<25} {'n':<6} {'r':<10} {'95% CI'}")
        logger.info("-" * 70)
        for _, row in rq1_df.iterrows():
            ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
            logger.info(f"{row['variable']:<12} {row['domain']:<25} {row['n']:<6} {row['pearson_r']:>+.3f}    {ci_str}")
        
        overall_r = rq1_df['pearson_r'].mean()
        logger.info(f"\n→ Mean correlation: r = {overall_r:.3f}")
    else:
        overall_r = np.nan
        logger.warning("No RQ1 results - check data parsing")
    
    # ==========================================================================
    # RQ2: Variance Compression Analysis
    # ==========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: RQ2 - VARIANCE COMPRESSION")
    logger.info("=" * 70)
    
    rq2_results = []
    
    for seed, df in all_results.items():
        for var in ANALYSIS_VARS:
            silicon_col = f'silicon_{var}'
            
            if silicon_col not in df.columns or var not in df.columns:
                continue
            
            silicon = df[silicon_col].dropna()
            human_full = df_full[var].dropna() if var in df_full.columns else df[var].dropna()
            
            if len(silicon) < 5:
                continue
            
            var_ratio = compute_variance_ratio(silicon, human_full)
            
            rq2_results.append({
                'seed': seed,
                'variable': var,
                'domain': VARIABLES[var]['domain'],
                'var_ratio': var_ratio,
                'silicon_sd': silicon.std(),
                'human_sd': human_full.std(),
                'silicon_mean': silicon.mean(),
                'human_mean': human_full.mean(),
                'mean_diff': silicon.mean() - human_full.mean(),
            })
    
    rq2_df = pd.DataFrame(rq2_results)
    
    # Print RQ2 summary
    if len(rq2_df) > 0:
        logger.info(f"\nRQ2 Results (Variance Compression):")
        logger.info(f"{'Variable':<12} {'VR':<12} {'Silicon SD':<12} {'Human SD':<12} {'Mean Diff'}")
        logger.info("-" * 65)
        for _, row in rq2_df.iterrows():
            logger.info(f"{row['variable']:<12} {row['var_ratio']:>10.1%}   {row['silicon_sd']:>10.2f}   {row['human_sd']:>10.2f}   {row['mean_diff']:>+8.2f}")
        
        overall_vr = rq2_df['var_ratio'].mean()
        logger.info(f"\n→ Mean variance ratio: {overall_vr:.1%}")
        logger.info(f"  (Values < 100% indicate variance compression)")
    else:
        overall_vr = np.nan
        logger.warning("No RQ2 results - check data parsing")
    
    # ==========================================================================
    # Save Results
    # ==========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: SAVING RESULTS")
    logger.info("=" * 70)
    
    rq1_file = os.path.join(args.output_dir, 'silicon_block1_rq1_results.csv')
    rq2_file = os.path.join(args.output_dir, 'silicon_block1_rq2_results.csv')
    
    rq1_df.to_csv(rq1_file, index=False)
    rq2_df.to_csv(rq2_file, index=False)
    
    logger.info(f"✓ RQ1 results saved to: {rq1_file}")
    logger.info(f"✓ RQ2 results saved to: {rq2_file}")
    
    # ==========================================================================
    # Final Summary
    # ==========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("BLOCK 1 COMPLETE")
    logger.info("=" * 70)
    
    summary = f"""
SUMMARY
-------
Study Design:
  - Sample: {CONFIG['sample_size']} respondents
  - Model: {CONFIG['model_name']}
  - Variables: {len(VARIABLES)} items across {len(set(v['domain'] for v in VARIABLES.values()))} domains
  - Temperature: {CONFIG['temperature']}

Key Results:
  - RQ1 Mean Individual Correlation: r = {overall_r:.3f if not np.isnan(overall_r) else 'N/A'}
  - RQ2 Mean Variance Ratio: {overall_vr:.1% if not np.isnan(overall_vr) else 'N/A'}

Interpretation:
  - Individual correlations near 0 suggest LLMs cannot recover individual-level responses
  - Variance ratios < 100% indicate systematic variance compression
  - These findings replicate established empirical regularities

Files saved to: {args.output_dir}/
  - silicon_block1_data_seed*.csv (raw data with silicon responses)
  - silicon_block1_rq1_results.csv (individual-level correlations)
  - silicon_block1_rq2_results.csv (variance compression metrics)

Next Steps:
  - If results look reasonable, increase --sample_size to 300-800
  - Run Block 2 for mechanism exploration (prompt framing, backstory enrichment)
  - Run Block 3 for group heterogeneity analysis
"""
    
    logger.info(summary)
    
    return rq1_df, rq2_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Silicon Sampling Block 1: Core Validation (VLLM + Qwen)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 50 samples
  python block1_vllm_qwen.py --backstory_file data/backstories.csv --ess_file data/ESS.dta --sample_size 50
  
  # Full run with 800 samples
  python block1_vllm_qwen.py --backstory_file data/backstories.csv --ess_file data/ESS.dta --sample_size 800
  
  # Use smaller model for faster testing
  python block1_vllm_qwen.py --backstory_file data/backstories.csv --ess_file data/ESS.dta --model Qwen/Qwen2.5-1.5B-Instruct
        """
    )
    
    parser.add_argument('--backstory_file', type=str, required=True,
                        help='Path to backstory CSV file (with idno and backstory columns)')
    parser.add_argument('--ess_file', type=str, required=True,
                        help='Path to ESS Stata file (.dta)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results (default: results/)')
    parser.add_argument('--sample_size', type=int, default=50,
                        help='Sample size (default: 50 for quick test, use 800 for full run)')
    parser.add_argument('--model', type=str, default=None,
                        help='Qwen model to use (default: Qwen/Qwen2.5-7B-Instruct)')
    
    args = parser.parse_args()
    main(args)
