#!/usr/bin/env python3
"""
Silicon Sampling Validation Framework - EXTENDED VERSION
=========================================================
Full Sample + RQ8 (Model Comparison) + RQ11 (Few-shot Anchoring)

Changes from previous version:
1. FULL SAMPLE: Uses all ESS UK respondents (no sampling limit)
2. RQ8: Model comparison between Qwen and Llama
3. RQ11: Few-shot anchoring with similar respondents

Usage:
    python silicon_sampling_extended.py --block 1 --model qwen
    python silicon_sampling_extended.py --block 1 --model llama
    python silicon_sampling_extended.py --block 4 --model qwen  # RQ11 few-shot
    python silicon_sampling_extended.py --block all --model qwen
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
BACKSTORY_FILE = os.path.join(DATA_DIR, "ess_uk_with_backstories_v1.csv")
ESS_FILE = os.path.join(DATA_DIR, "ESS9e03_2_GBonly.dta")

# Output directories
RESULTS_DIR = "results"
LOGS_DIR = "logs"

# 50 ESS Variables (excluding lrscale due to data leakage)
ESS_VARIABLES = {
    # Political Attitudes (5)
    'stfdem': {'scale': '0-10', 'label': 'Satisfaction with democracy', 'domain': 'political'},
    'polintr': {'scale': '1-4', 'label': 'Political interest', 'domain': 'political'},
    'psppipla': {'scale': '1-5', 'label': 'Political system allows influence', 'domain': 'political'},
    'psppsgva': {'scale': '1-5', 'label': 'Political system gives say', 'domain': 'political'},
    'vote': {'scale': '1-3', 'label': 'Voted last election', 'domain': 'political'},
    
    # Political Efficacy (2)
    'actrolga': {'scale': '1-5', 'label': 'Able to take active role', 'domain': 'efficacy'},
    'cptppola': {'scale': '1-5', 'label': 'Confident in political participation', 'domain': 'efficacy'},
    
    # Institutional Trust (7)
    'trstprl': {'scale': '0-10', 'label': 'Trust in parliament', 'domain': 'inst_trust'},
    'trstplt': {'scale': '0-10', 'label': 'Trust in politicians', 'domain': 'inst_trust'},
    'trstlgl': {'scale': '0-10', 'label': 'Trust in legal system', 'domain': 'inst_trust'},
    'trstprt': {'scale': '0-10', 'label': 'Trust in political parties', 'domain': 'inst_trust'},
    'trstplc': {'scale': '0-10', 'label': 'Trust in police', 'domain': 'inst_trust'},
    'trstep': {'scale': '0-10', 'label': 'Trust in European Parliament', 'domain': 'inst_trust'},
    'trstun': {'scale': '0-10', 'label': 'Trust in United Nations', 'domain': 'inst_trust'},
    
    # Social Trust (5)
    'ppltrst': {'scale': '0-10', 'label': 'Most people can be trusted', 'domain': 'social_trust'},
    'pplfair': {'scale': '0-10', 'label': 'Most people try to be fair', 'domain': 'social_trust'},
    'pplhlp': {'scale': '0-10', 'label': 'Most people try to be helpful', 'domain': 'social_trust'},
    'sclmeet': {'scale': '1-7', 'label': 'How often socially meet', 'domain': 'social_trust'},
    'inprdsc': {'scale': '0-10', 'label': 'Have people to discuss intimate matters', 'domain': 'social_trust'},
    
    # Subjective Well-being (6)
    'happy': {'scale': '0-10', 'label': 'How happy are you', 'domain': 'wellbeing'},
    'stflife': {'scale': '0-10', 'label': 'Satisfaction with life', 'domain': 'wellbeing'},
    'stfeco': {'scale': '0-10', 'label': 'Satisfaction with economy', 'domain': 'wellbeing'},
    'stfgov': {'scale': '0-10', 'label': 'Satisfaction with government', 'domain': 'wellbeing'},
    'stfhlth': {'scale': '0-10', 'label': 'Satisfaction with health services', 'domain': 'wellbeing'},
    'stfedu': {'scale': '0-10', 'label': 'Satisfaction with education', 'domain': 'wellbeing'},
    
    # Immigration Attitudes (6)
    'imueclt': {'scale': '0-10', 'label': 'Immigration enriches culture', 'domain': 'immigration'},
    'imwbcnt': {'scale': '0-10', 'label': 'Immigration good for country', 'domain': 'immigration'},
    'imbgeco': {'scale': '0-10', 'label': 'Immigration good for economy', 'domain': 'immigration'},
    'imsmetn': {'scale': '1-4', 'label': 'Allow many/few same ethnicity', 'domain': 'immigration'},
    'impcntr': {'scale': '1-4', 'label': 'Allow many/few poor countries', 'domain': 'immigration'},
    'imdfetn': {'scale': '1-4', 'label': 'Allow many/few different ethnicity', 'domain': 'immigration'},
    
    # Social Values (4)
    'gincdif': {'scale': '1-5', 'label': 'Government reduce income differences', 'domain': 'values'},
    'freehms': {'scale': '1-5', 'label': 'Gays free to live as they wish', 'domain': 'values'},
    'hmsfmlsh': {'scale': '1-5', 'label': 'Ashamed if gay family member', 'domain': 'values'},
    'euftf': {'scale': '0-10', 'label': 'European unification go further', 'domain': 'values'},
    
    # Personal Values - Schwartz (10)
    'ipcrtiv': {'scale': '1-6', 'label': 'Important to be creative', 'domain': 'personal_values'},
    'imprich': {'scale': '1-6', 'label': 'Important to be rich', 'domain': 'personal_values'},
    'ipeqopt': {'scale': '1-6', 'label': 'Important equal opportunities', 'domain': 'personal_values'},
    'impsafe': {'scale': '1-6', 'label': 'Important to live safely', 'domain': 'personal_values'},
    'impfree': {'scale': '1-6', 'label': 'Important to make own decisions', 'domain': 'personal_values'},
    'iphlppl': {'scale': '1-6', 'label': 'Important to help others', 'domain': 'personal_values'},
    'ipsuces': {'scale': '1-6', 'label': 'Important to be successful', 'domain': 'personal_values'},
    'imptrad': {'scale': '1-6', 'label': 'Important tradition and customs', 'domain': 'personal_values'},
    'impenv': {'scale': '1-6', 'label': 'Important to care for environment', 'domain': 'personal_values'},
    'ipfrule': {'scale': '1-6', 'label': 'Important to follow rules', 'domain': 'personal_values'},
    
    # National Attachment (2)
    'atchctr': {'scale': '0-10', 'label': 'Emotionally attached to country', 'domain': 'attachment'},
    'atcherp': {'scale': '0-10', 'label': 'Emotionally attached to Europe', 'domain': 'attachment'},
    
    # Health, Religion, Safety (3)
    'health': {'scale': '1-5', 'label': 'Subjective general health', 'domain': 'health'},
    'rlgatnd': {'scale': '1-7', 'label': 'How often attend religious services', 'domain': 'religion'},
    'aesfdrk': {'scale': '1-4', 'label': 'Feel safe walking alone after dark', 'domain': 'safety'},
}

# Variables for similarity matching in RQ11
SIMILARITY_VARS = ['agea', 'gndr', 'eduyrs', 'hinctnta', 'region']

# Prompt templates
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

# Few-shot prompt template for RQ11
PROMPT_FEWSHOT = """Here are responses from people with similar backgrounds:

{examples}

Now consider this person:
{backstory}

Question: {question}
Scale: {scale}

Based on how similar people responded, what would this person answer?
Respond with ONLY a single number within the scale range. No explanation."""

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_logging(block_name, model_name):
    """Setup logging configuration."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"silicon_{block_name}_{model_name}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_data():
    """Load ESS data and backstories."""
    logging.info("Loading data...")
    
    # Load backstories
    backstory_df = pd.read_csv(BACKSTORY_FILE)
    logging.info(f"Loaded {len(backstory_df)} backstories")
    
    # Load ESS data
    ess_df = pd.read_stata(ESS_FILE)
    logging.info(f"Loaded {len(ess_df)} ESS responses")
    
    # Merge on idno
    merged = backstory_df.merge(ess_df, on='idno', how='inner')
    logging.info(f"Merged dataset: {len(merged)} respondents")
    
    return merged

def parse_response(text):
    """Extract numeric response from model output."""
    if text is None:
        return None
    text = str(text).strip()
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[0])
        except:
            return None
    return None

def bootstrap_correlation_ci(x, y, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for correlation."""
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
    lower = np.percentile(correlations, alpha/2 * 100)
    upper = np.percentile(correlations, (1 - alpha/2) * 100)
    
    return np.mean(correlations), lower, upper

def compute_variance_metrics(human_vals, silicon_vals):
    """Compute variance compression metrics."""
    human_var = np.nanvar(human_vals)
    silicon_var = np.nanvar(silicon_vals)
    
    var_ratio = silicon_var / human_var if human_var > 0 else np.nan
    
    # Tail mass (|z| > 1.5)
    human_z = (human_vals - np.nanmean(human_vals)) / np.nanstd(human_vals)
    silicon_z = (silicon_vals - np.nanmean(silicon_vals)) / np.nanstd(silicon_vals)
    
    human_tail = np.nanmean(np.abs(human_z) > 1.5)
    silicon_tail = np.nanmean(np.abs(silicon_z) > 1.5)
    
    return {
        'var_ratio': var_ratio,
        'human_var': human_var,
        'silicon_var': silicon_var,
        'human_mean': np.nanmean(human_vals),
        'silicon_mean': np.nanmean(silicon_vals),
        'human_tail_mass': human_tail,
        'silicon_tail_mass': silicon_tail
    }

def find_similar_respondents(target_idx, df, n_similar=3, exclude_self=True):
    """Find n most similar respondents based on demographics."""
    # Prepare similarity features
    sim_cols = [c for c in SIMILARITY_VARS if c in df.columns]
    
    if not sim_cols:
        # Fallback: random selection
        candidates = [i for i in range(len(df)) if i != target_idx]
        return random.sample(candidates, min(n_similar, len(candidates)))
    
    # Extract and normalize features
    features = df[sim_cols].copy()
    
    # Handle missing values
    for col in sim_cols:
        features[col] = pd.to_numeric(features[col], errors='coerce')
        features[col] = features[col].fillna(features[col].median())
    
    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Compute distances from target
    target_features = features_scaled[target_idx].reshape(1, -1)
    distances = euclidean_distances(target_features, features_scaled)[0]
    
    # Get indices sorted by distance
    sorted_indices = np.argsort(distances)
    
    # Exclude self and get top n_similar
    similar_indices = []
    for idx in sorted_indices:
        if exclude_self and idx == target_idx:
            continue
        similar_indices.append(idx)
        if len(similar_indices) >= n_similar:
            break
    
    return similar_indices

# ============================================================================
# VLLM GENERATION
# ============================================================================

def generate_with_vllm(prompts, model_name='qwen', temperature=0.7, max_tokens=150):
    """Generate responses using VLLM."""
    from vllm import LLM, SamplingParams
    
    model_path = MODELS.get(model_name, MODELS['qwen'])
    logging.info(f"Loading model: {model_path}")
    
    # Initialize VLLM
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85
    )
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95
    )
    
    logging.info(f"Generating {len(prompts)} responses...")
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    elapsed = time.time() - start_time
    logging.info(f"Generated {len(outputs)} responses in {elapsed:.1f}s ({len(outputs)/elapsed:.1f} resp/sec)")
    
    responses = [output.outputs[0].text.strip() for output in outputs]
    return responses

# ============================================================================
# BLOCK 1: RQ1 + RQ2 (Individual Recovery + Variance Compression)
# ============================================================================

def run_block1(df, model_name='qwen', seed=888):
    """Run Block 1: Individual-level recovery and variance compression."""
    logger = setup_logging('block1', model_name)
    logger.info(f"=== BLOCK 1: RQ1 + RQ2 (Model: {model_name}) ===")
    logger.info(f"Sample size: {len(df)} (FULL SAMPLE)")
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Prepare prompts
    prompts = []
    prompt_metadata = []
    
    for var_name, var_info in ESS_VARIABLES.items():
        for idx, row in df.iterrows():
            backstory = row.get('full_backstory', row.get('backstory', ''))
            
            prompt = PROMPT_1P_STANDARD.format(
                backstory=backstory,
                question=var_info['label'],
                scale=var_info['scale']
            )
            
            prompts.append(prompt)
            prompt_metadata.append({
                'idno': row['idno'],
                'variable': var_name,
                'human_response': row.get(var_name, np.nan),
                'domain': var_info['domain']
            })
    
    logger.info(f"Total prompts: {len(prompts)}")
    
    # Generate responses
    responses = generate_with_vllm(prompts, model_name=model_name)
    
    # Parse responses
    results_data = []
    for i, (meta, response) in enumerate(zip(prompt_metadata, responses)):
        silicon_val = parse_response(response)
        results_data.append({
            **meta,
            'silicon_response': silicon_val,
            'raw_response': response
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Save raw data
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_file = os.path.join(RESULTS_DIR, f"silicon_block1_data_{model_name}_seed{seed}.csv")
    results_df.to_csv(raw_file, index=False)
    logger.info(f"Saved raw data to {raw_file}")
    
    # Compute RQ1: Individual correlations
    rq1_results = []
    for var_name in ESS_VARIABLES.keys():
        var_data = results_df[results_df['variable'] == var_name]
        human = var_data['human_response'].values.astype(float)
        silicon = var_data['silicon_response'].values.astype(float)
        
        valid_mask = ~(np.isnan(human) | np.isnan(silicon))
        if valid_mask.sum() < 10:
            continue
        
        r_pearson, p_pearson = stats.pearsonr(human[valid_mask], silicon[valid_mask])
        r_spearman, p_spearman = stats.spearmanr(human[valid_mask], silicon[valid_mask])
        r_mean, r_lower, r_upper = bootstrap_correlation_ci(human, silicon)
        
        rq1_results.append({
            'variable': var_name,
            'domain': ESS_VARIABLES[var_name]['domain'],
            'r_pearson': r_pearson,
            'p_pearson': p_pearson,
            'r_spearman': r_spearman,
            'p_spearman': p_spearman,
            'r_bootstrap_mean': r_mean,
            'r_ci_lower': r_lower,
            'r_ci_upper': r_upper,
            'n_valid': valid_mask.sum(),
            'model': model_name
        })
    
    rq1_df = pd.DataFrame(rq1_results)
    rq1_file = os.path.join(RESULTS_DIR, f"silicon_block1_rq1_{model_name}.csv")
    rq1_df.to_csv(rq1_file, index=False)
    logger.info(f"RQ1 results saved to {rq1_file}")
    logger.info(f"RQ1 Mean r: {rq1_df['r_pearson'].mean():.3f}")
    
    # Compute RQ2: Variance compression
    rq2_results = []
    for var_name in ESS_VARIABLES.keys():
        var_data = results_df[results_df['variable'] == var_name]
        human = var_data['human_response'].values.astype(float)
        silicon = var_data['silicon_response'].values.astype(float)
        
        metrics = compute_variance_metrics(human, silicon)
        
        rq2_results.append({
            'variable': var_name,
            'domain': ESS_VARIABLES[var_name]['domain'],
            **metrics,
            'model': model_name
        })
    
    rq2_df = pd.DataFrame(rq2_results)
    rq2_file = os.path.join(RESULTS_DIR, f"silicon_block1_rq2_{model_name}.csv")
    rq2_df.to_csv(rq2_file, index=False)
    logger.info(f"RQ2 results saved to {rq2_file}")
    logger.info(f"RQ2 Mean VR: {rq2_df['var_ratio'].mean():.3f}")
    
    return results_df, rq1_df, rq2_df

# ============================================================================
# BLOCK 2: RQ3-RQ6 (Mechanism Exploration)
# ============================================================================

def run_block2(df, model_name='qwen', seed=888):
    """Run Block 2: Mechanism exploration (RQ3-RQ6)."""
    logger = setup_logging('block2', model_name)
    logger.info(f"=== BLOCK 2: RQ3-RQ6 (Model: {model_name}) ===")
    
    random.seed(seed)
    np.random.seed(seed)
    
    # 10 theoretically important variables for mechanism tests
    # Covers: political, trust, wellbeing, immigration, values, health
    test_vars = [
        'stfdem',    # Political: satisfaction with democracy
        'trstplt',   # Inst Trust: trust in politicians
        'ppltrst',   # Social Trust: most people can be trusted
        'happy',     # Wellbeing: happiness
        'imueclt',   # Immigration: cultural enrichment
        'gincdif',   # Values: reduce income differences
        'polintr',   # Political: political interest
        'trstplc',   # Inst Trust: trust in police
        'imprich',   # Personal Values: important to be rich
        'health',    # Health: subjective health
    ]
    
    all_results = []
    
    # RQ3: Prompt framing (1P vs 3P vs idiosyncratic)
    logger.info("Running RQ3: Prompt framing...")
    prompt_conditions = {
        '1p_standard': PROMPT_1P_STANDARD,
        '3p': PROMPT_3P,
        '1p_idiosyncratic': PROMPT_1P_IDIOSYNCRATIC
    }
    
    for condition, template in prompt_conditions.items():
        prompts = []
        metadata = []
        
        for var_name in test_vars:
            var_info = ESS_VARIABLES[var_name]
            for idx, row in df.iterrows():
                backstory = row.get('full_backstory', row.get('backstory', ''))
                prompt = template.format(
                    backstory=backstory,
                    question=var_info['label'],
                    scale=var_info['scale']
                )
                prompts.append(prompt)
                metadata.append({
                    'idno': row['idno'],
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
    
    # RQ6: Format sensitivity (standard vs reversed)
    logger.info("Running RQ6: Format sensitivity...")
    for var_name in test_vars:
        var_info = ESS_VARIABLES[var_name]
        
        # Standard format
        prompts_std = []
        prompts_rev = []
        metadata_pairs = []
        
        for idx, row in df.iterrows():
            backstory = row.get('full_backstory', row.get('backstory', ''))
            
            # Standard
            prompt_std = PROMPT_1P_STANDARD.format(
                backstory=backstory,
                question=var_info['label'],
                scale=var_info['scale']
            )
            
            # Reversed scale
            scale_parts = var_info['scale'].split('-')
            if len(scale_parts) == 2:
                rev_scale = f"{scale_parts[1]}-{scale_parts[0]}"
            else:
                rev_scale = var_info['scale']
            
            prompt_rev = PROMPT_1P_STANDARD.format(
                backstory=backstory,
                question=var_info['label'] + " (reversed scale)",
                scale=rev_scale
            )
            
            prompts_std.append(prompt_std)
            prompts_rev.append(prompt_rev)
            metadata_pairs.append({
                'idno': row['idno'],
                'variable': var_name,
                'human_response': row.get(var_name, np.nan)
            })
        
        responses_std = generate_with_vllm(prompts_std, model_name=model_name)
        responses_rev = generate_with_vllm(prompts_rev, model_name=model_name)
        
        for meta, resp_std, resp_rev in zip(metadata_pairs, responses_std, responses_rev):
            all_results.append({
                **meta,
                'silicon_response': parse_response(resp_std),
                'condition': 'standard',
                'rq': 'RQ6',
                'model': model_name
            })
            all_results.append({
                **meta,
                'silicon_response': parse_response(resp_rev),
                'condition': 'reversed',
                'rq': 'RQ6',
                'model': model_name
            })
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(RESULTS_DIR, f"silicon_block2_data_{model_name}.csv")
    results_df.to_csv(results_file, index=False)
    logger.info(f"Block 2 results saved to {results_file}")
    
    # Analyze RQ3
    rq3_results = []
    for condition in prompt_conditions.keys():
        cond_data = results_df[(results_df['rq'] == 'RQ3') & (results_df['condition'] == condition)]
        for var_name in test_vars:
            var_data = cond_data[cond_data['variable'] == var_name]
            human = var_data['human_response'].values.astype(float)
            silicon = var_data['silicon_response'].values.astype(float)
            
            valid = ~(np.isnan(human) | np.isnan(silicon))
            if valid.sum() < 10:
                continue
            
            r, p = stats.pearsonr(human[valid], silicon[valid])
            metrics = compute_variance_metrics(human, silicon)
            
            rq3_results.append({
                'variable': var_name,
                'condition': condition,
                'r_pearson': r,
                'var_ratio': metrics['var_ratio'],
                'model': model_name
            })
    
    rq3_df = pd.DataFrame(rq3_results)
    rq3_df.to_csv(os.path.join(RESULTS_DIR, f"silicon_block2_rq3_{model_name}.csv"), index=False)
    
    # Analyze RQ6
    rq6_results = []
    rq6_data = results_df[results_df['rq'] == 'RQ6']
    for var_name in test_vars:
        std_data = rq6_data[(rq6_data['variable'] == var_name) & (rq6_data['condition'] == 'standard')]
        rev_data = rq6_data[(rq6_data['variable'] == var_name) & (rq6_data['condition'] == 'reversed')]
        
        if len(std_data) == 0 or len(rev_data) == 0:
            continue
        
        merged = std_data.merge(rev_data, on='idno', suffixes=('_std', '_rev'))
        
        std_vals = merged['silicon_response_std'].values.astype(float)
        rev_vals = merged['silicon_response_rev'].values.astype(float)
        
        valid = ~(np.isnan(std_vals) | np.isnan(rev_vals))
        if valid.sum() < 10:
            continue
        
        # Flip rate
        flip_rate = np.mean(std_vals[valid] != rev_vals[valid])
        
        # Correlation between formats
        r_formats, _ = stats.pearsonr(std_vals[valid], rev_vals[valid])
        
        rq6_results.append({
            'variable': var_name,
            'flip_rate': flip_rate,
            'r_between_formats': r_formats,
            'n_valid': valid.sum(),
            'model': model_name
        })
    
    rq6_df = pd.DataFrame(rq6_results)
    rq6_df.to_csv(os.path.join(RESULTS_DIR, f"silicon_block2_rq6_{model_name}.csv"), index=False)
    
    logger.info("Block 2 complete")
    return results_df

# ============================================================================
# BLOCK 3: RQ7 (Subgroup Heterogeneity)
# ============================================================================

def run_block3(df, model_name='qwen', seed=888):
    """Run Block 3: Subgroup heterogeneity (RQ7)."""
    logger = setup_logging('block3', model_name)
    logger.info(f"=== BLOCK 3: RQ7 (Model: {model_name}) ===")
    
    # Load Block 1 results
    block1_file = os.path.join(RESULTS_DIR, f"silicon_block1_data_{model_name}_seed{seed}.csv")
    if not os.path.exists(block1_file):
        logger.error(f"Block 1 results not found: {block1_file}")
        return None
    
    results_df = pd.read_csv(block1_file)
    
    # Define disadvantaged groups
    groups = {
        'low_education': df['eduyrs'] < df['eduyrs'].median() if 'eduyrs' in df.columns else None,
        'low_income': df['hinctnta'] <= 3 if 'hinctnta' in df.columns else None,
        'female': df['gndr'] == 2 if 'gndr' in df.columns else None,
        'immigrant': df['brncntr'] == 2 if 'brncntr' in df.columns else None
    }
    
    # RQ7a: Subgroup variance compression
    h7a_results = []
    for group_name, group_mask in groups.items():
        if group_mask is None:
            continue
        
        for var_name in ESS_VARIABLES.keys():
            var_data = results_df[results_df['variable'] == var_name].copy()
            
            # Match with group membership
            var_data = var_data.merge(
                df[['idno']].assign(is_disadvantaged=group_mask.values),
                on='idno'
            )
            
            for is_disadv in [True, False]:
                subgroup = var_data[var_data['is_disadvantaged'] == is_disadv]
                
                human = subgroup['human_response'].values.astype(float)
                silicon = subgroup['silicon_response'].values.astype(float)
                
                if len(human) < 20:
                    continue
                
                metrics = compute_variance_metrics(human, silicon)
                
                h7a_results.append({
                    'variable': var_name,
                    'group': group_name,
                    'is_disadvantaged': is_disadv,
                    'var_ratio': metrics['var_ratio'],
                    'n': len(human),
                    'model': model_name
                })
    
    h7a_df = pd.DataFrame(h7a_results)
    h7a_df.to_csv(os.path.join(RESULTS_DIR, f"silicon_block3_h7a_{model_name}.csv"), index=False)
    
    # RQ7c: Issue sensitivity (correlation of human SD with VR)
    h7c_results = []
    for var_name in ESS_VARIABLES.keys():
        var_data = results_df[results_df['variable'] == var_name]
        
        human = var_data['human_response'].values.astype(float)
        silicon = var_data['silicon_response'].values.astype(float)
        
        human_sd = np.nanstd(human)
        metrics = compute_variance_metrics(human, silicon)
        
        h7c_results.append({
            'variable': var_name,
            'domain': ESS_VARIABLES[var_name]['domain'],
            'human_sd': human_sd,
            'var_ratio': metrics['var_ratio'],
            'model': model_name
        })
    
    h7c_df = pd.DataFrame(h7c_results)
    
    # Compute correlation
    r_sd_vr, p_sd_vr = stats.pearsonr(h7c_df['human_sd'], h7c_df['var_ratio'])
    logger.info(f"RQ7c: Correlation of human SD with VR: r={r_sd_vr:.3f}, p={p_sd_vr:.4f}")
    
    h7c_df['r_sd_vr'] = r_sd_vr
    h7c_df['p_sd_vr'] = p_sd_vr
    h7c_df.to_csv(os.path.join(RESULTS_DIR, f"silicon_block3_h7c_{model_name}.csv"), index=False)
    
    logger.info("Block 3 complete")
    return h7a_df, h7c_df

# ============================================================================
# BLOCK 4: RQ11 (Few-shot Anchoring) - NEW
# ============================================================================

def run_block4_fewshot(df, model_name='qwen', n_similar=3, seed=888):
    """Run Block 4: Few-shot anchoring with similar respondents (RQ11)."""
    logger = setup_logging('block4_fewshot', model_name)
    logger.info(f"=== BLOCK 4: RQ11 Few-shot Anchoring (Model: {model_name}) ===")
    logger.info(f"Using {n_similar} similar respondents as few-shot examples")
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Same 10 variables as Block 2 for direct comparison
    test_vars = [
        'stfdem',    # Political: satisfaction with democracy
        'trstplt',   # Inst Trust: trust in politicians
        'ppltrst',   # Social Trust: most people can be trusted
        'happy',     # Wellbeing: happiness
        'imueclt',   # Immigration: cultural enrichment
        'gincdif',   # Values: reduce income differences
        'polintr',   # Political: political interest
        'trstplc',   # Inst Trust: trust in police
        'imprich',   # Personal Values: important to be rich
        'health',    # Health: subjective health
    ]
    
    results = []
    
    for var_name in test_vars:
        var_info = ESS_VARIABLES[var_name]
        logger.info(f"Processing variable: {var_name}")
        
        # Prepare prompts for all respondents
        prompts_zeroshot = []
        prompts_fewshot = []
        metadata = []
        
        for target_idx in range(len(df)):
            row = df.iloc[target_idx]
            backstory = row.get('full_backstory', row.get('backstory', ''))
            human_response = row.get(var_name, np.nan)
            
            # Zero-shot prompt (baseline)
            prompt_zero = PROMPT_1P_STANDARD.format(
                backstory=backstory,
                question=var_info['label'],
                scale=var_info['scale']
            )
            
            # Find similar respondents (excluding self)
            similar_indices = find_similar_respondents(target_idx, df, n_similar=n_similar)
            
            # Build few-shot examples
            examples_text = ""
            for i, sim_idx in enumerate(similar_indices):
                sim_row = df.iloc[sim_idx]
                sim_backstory = sim_row.get('full_backstory', sim_row.get('backstory', ''))
                sim_response = sim_row.get(var_name, np.nan)
                
                if pd.notna(sim_response):
                    examples_text += f"\n[Person {i+1}]\n{sim_backstory[:500]}...\nTheir response: {int(sim_response)}\n"
            
            # Few-shot prompt
            prompt_few = PROMPT_FEWSHOT.format(
                examples=examples_text,
                backstory=backstory,
                question=var_info['label'],
                scale=var_info['scale']
            )
            
            prompts_zeroshot.append(prompt_zero)
            prompts_fewshot.append(prompt_few)
            metadata.append({
                'idno': row['idno'],
                'variable': var_name,
                'human_response': human_response,
                'similar_indices': similar_indices
            })
        
        # Generate zero-shot responses
        logger.info(f"  Generating zero-shot responses...")
        responses_zero = generate_with_vllm(prompts_zeroshot, model_name=model_name)
        
        # Generate few-shot responses
        logger.info(f"  Generating few-shot responses...")
        responses_few = generate_with_vllm(prompts_fewshot, model_name=model_name)
        
        # Store results
        for meta, resp_zero, resp_few in zip(metadata, responses_zero, responses_few):
            results.append({
                **meta,
                'silicon_zeroshot': parse_response(resp_zero),
                'silicon_fewshot': parse_response(resp_few),
                'model': model_name
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_file = os.path.join(RESULTS_DIR, f"silicon_block4_fewshot_data_{model_name}.csv")
    results_df.to_csv(results_file, index=False)
    logger.info(f"Raw data saved to {results_file}")
    
    # Analyze RQ11: Compare zero-shot vs few-shot
    rq11_results = []
    for var_name in test_vars:
        var_data = results_df[results_df['variable'] == var_name]
        
        human = var_data['human_response'].values.astype(float)
        silicon_zero = var_data['silicon_zeroshot'].values.astype(float)
        silicon_few = var_data['silicon_fewshot'].values.astype(float)
        
        # Zero-shot metrics
        valid_zero = ~(np.isnan(human) | np.isnan(silicon_zero))
        if valid_zero.sum() >= 10:
            r_zero, p_zero = stats.pearsonr(human[valid_zero], silicon_zero[valid_zero])
            metrics_zero = compute_variance_metrics(human, silicon_zero)
        else:
            r_zero, p_zero = np.nan, np.nan
            metrics_zero = {'var_ratio': np.nan}
        
        # Few-shot metrics
        valid_few = ~(np.isnan(human) | np.isnan(silicon_few))
        if valid_few.sum() >= 10:
            r_few, p_few = stats.pearsonr(human[valid_few], silicon_few[valid_few])
            metrics_few = compute_variance_metrics(human, silicon_few)
        else:
            r_few, p_few = np.nan, np.nan
            metrics_few = {'var_ratio': np.nan}
        
        rq11_results.append({
            'variable': var_name,
            'domain': ESS_VARIABLES[var_name]['domain'],
            'r_zeroshot': r_zero,
            'r_fewshot': r_few,
            'r_improvement': r_few - r_zero if not (np.isnan(r_few) or np.isnan(r_zero)) else np.nan,
            'vr_zeroshot': metrics_zero['var_ratio'],
            'vr_fewshot': metrics_few['var_ratio'],
            'n_zeroshot': valid_zero.sum(),
            'n_fewshot': valid_few.sum(),
            'model': model_name
        })
    
    rq11_df = pd.DataFrame(rq11_results)
    rq11_file = os.path.join(RESULTS_DIR, f"silicon_block4_rq11_{model_name}.csv")
    rq11_df.to_csv(rq11_file, index=False)
    
    # Summary statistics
    mean_r_zero = rq11_df['r_zeroshot'].mean()
    mean_r_few = rq11_df['r_fewshot'].mean()
    mean_improvement = rq11_df['r_improvement'].mean()
    
    logger.info("=" * 60)
    logger.info("RQ11 SUMMARY: Few-shot Anchoring Results")
    logger.info("=" * 60)
    logger.info(f"Mean r (zero-shot): {mean_r_zero:.3f}")
    logger.info(f"Mean r (few-shot):  {mean_r_few:.3f}")
    logger.info(f"Mean improvement:   {mean_improvement:+.3f}")
    logger.info("=" * 60)
    
    return results_df, rq11_df

# ============================================================================
# RQ8: MODEL COMPARISON
# ============================================================================

def run_model_comparison(df, seed=888):
    """Run RQ8: Compare Qwen vs Llama."""
    logger = setup_logging('rq8_comparison', 'both')
    logger.info("=== RQ8: Model Comparison ===")
    
    # Run Block 1 for both models
    logger.info("Running Block 1 with Qwen...")
    run_block1(df, model_name='qwen', seed=seed)
    
    logger.info("Running Block 1 with Llama...")
    run_block1(df, model_name='llama', seed=seed)
    
    # Compare results
    qwen_rq1 = pd.read_csv(os.path.join(RESULTS_DIR, "silicon_block1_rq1_qwen.csv"))
    llama_rq1 = pd.read_csv(os.path.join(RESULTS_DIR, "silicon_block1_rq1_llama.csv"))
    
    qwen_rq2 = pd.read_csv(os.path.join(RESULTS_DIR, "silicon_block1_rq2_qwen.csv"))
    llama_rq2 = pd.read_csv(os.path.join(RESULTS_DIR, "silicon_block1_rq2_llama.csv"))
    
    # Merge for comparison
    comparison_rq1 = qwen_rq1[['variable', 'domain', 'r_pearson']].merge(
        llama_rq1[['variable', 'r_pearson']],
        on='variable',
        suffixes=('_qwen', '_llama')
    )
    
    comparison_rq2 = qwen_rq2[['variable', 'domain', 'var_ratio']].merge(
        llama_rq2[['variable', 'var_ratio']],
        on='variable',
        suffixes=('_qwen', '_llama')
    )
    
    # Save comparison
    comparison_rq1.to_csv(os.path.join(RESULTS_DIR, "silicon_rq8_model_comparison_r.csv"), index=False)
    comparison_rq2.to_csv(os.path.join(RESULTS_DIR, "silicon_rq8_model_comparison_vr.csv"), index=False)
    
    # Summary
    logger.info("=" * 60)
    logger.info("RQ8 SUMMARY: Model Comparison")
    logger.info("=" * 60)
    logger.info(f"Qwen mean r:  {comparison_rq1['r_pearson_qwen'].mean():.3f}")
    logger.info(f"Llama mean r: {comparison_rq1['r_pearson_llama'].mean():.3f}")
    logger.info(f"Qwen mean VR:  {comparison_rq2['var_ratio_qwen'].mean():.3f}")
    logger.info(f"Llama mean VR: {comparison_rq2['var_ratio_llama'].mean():.3f}")
    
    # Correlation between model outputs
    r_models, p_models = stats.pearsonr(
        comparison_rq1['r_pearson_qwen'],
        comparison_rq1['r_pearson_llama']
    )
    logger.info(f"Correlation of r across models: {r_models:.3f} (p={p_models:.4f})")
    logger.info("=" * 60)
    
    return comparison_rq1, comparison_rq2

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Silicon Sampling Extended Framework')
    parser.add_argument('--block', type=str, default='all',
                        choices=['1', '2', '3', '4', 'rq8', 'all'],
                        help='Which block to run (1, 2, 3, 4=fewshot, rq8=model comparison, all)')
    parser.add_argument('--model', type=str, default='qwen',
                        choices=['qwen', 'llama'],
                        help='Model to use')
    parser.add_argument('--seed', type=int, default=888,
                        help='Random seed')
    parser.add_argument('--n_similar', type=int, default=3,
                        help='Number of similar respondents for few-shot (RQ11)')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Load data (FULL SAMPLE)
    df = load_data()
    print(f"Loaded FULL SAMPLE: {len(df)} respondents")
    
    if args.block == 'all':
        print("Running ALL blocks with Qwen...")
        run_block1(df, model_name='qwen', seed=args.seed)
        run_block2(df, model_name='qwen', seed=args.seed)
        run_block3(df, model_name='qwen', seed=args.seed)
        run_block4_fewshot(df, model_name='qwen', n_similar=args.n_similar, seed=args.seed)
    elif args.block == '1':
        run_block1(df, model_name=args.model, seed=args.seed)
    elif args.block == '2':
        run_block2(df, model_name=args.model, seed=args.seed)
    elif args.block == '3':
        run_block3(df, model_name=args.model, seed=args.seed)
    elif args.block == '4':
        run_block4_fewshot(df, model_name=args.model, n_similar=args.n_similar, seed=args.seed)
    elif args.block == 'rq8':
        run_model_comparison(df, seed=args.seed)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
