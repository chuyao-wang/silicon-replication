#!/bin/bash
# ============================================================================
# Silicon Sampling v5 — PARALLEL FULL RUN
# ============================================================================
# Submits Qwen and Llama as SEPARATE SLURM jobs → run on different GPUs.
#
# Performance optimizations (v5):
#   - max_tokens: 150 → 16  (output is 1-2 tokens, not 150)
#   - max_model_len: 4096 → 1024  (prompts ~280 tokens, frees KV cache)
#   - gpu_memory_utilization: 0.85 → 0.92  (inference-only workload)
#   - Model caching: no reload between calls within same process
#
# Estimated speedup: 2-3x → ~3-4h per model instead of ~7.5h
# With parallel: both complete in ~3-4h wall time
#
# Usage:
#   bash submit_parallel_full.sh              # Both models, full sample
#   bash submit_parallel_full.sh 1000         # Both models, 1000/country
# ============================================================================

SAMPLE_N=${1:-""}   # Optional: per-country sample size (empty = full)

echo "======================================================================"
echo "SILICON SAMPLING v5 — PARALLEL FULL RUN"
echo "======================================================================"
echo "Sample: ${SAMPLE_N:-FULL (no sampling)}"
echo "Submitting two independent GPU jobs..."
echo "======================================================================"

cd ~/Winston_Code

# Build sample argument
SAMPLE_ARG=""
if [ -n "$SAMPLE_N" ]; then
    SAMPLE_ARG="--sample_per_country $SAMPLE_N"
fi

# ---- Job 1: Qwen ----
JOB1=$(sbatch --parsable --job-name=silicon_qwen \
    --partition=winston-gpu \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=12:00:00 \
    --output=logs/silicon_full_qwen_%j.out \
    --error=logs/silicon_full_qwen_%j.err \
    --wrap="
cd ~/Winston_Code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
mkdir -p logs results plots

echo 'GPU Status:'
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

python silicon_sampling_extended_v5.py \
    --block trust \
    --model qwen \
    --backstory v5 \
    --seed 888 \
    $SAMPLE_ARG

# Auto-generate plots
pip install matplotlib --quiet --break-system-packages 2>/dev/null
python plot_silicon_results.py --model qwen 2>&1 || true

echo 'QWEN COMPLETE'
ls -lh results/silicon_trust_*qwen*.csv
")

echo "  Qwen job submitted: $JOB1"

# ---- Job 2: Llama ----
JOB2=$(sbatch --parsable --job-name=silicon_llama \
    --partition=winston-gpu \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=12:00:00 \
    --output=logs/silicon_full_llama_%j.out \
    --error=logs/silicon_full_llama_%j.err \
    --wrap="
cd ~/Winston_Code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
mkdir -p logs results plots

echo 'GPU Status:'
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

python silicon_sampling_extended_v5.py \
    --block trust \
    --model llama \
    --backstory v5 \
    --seed 888 \
    $SAMPLE_ARG

# Auto-generate plots
pip install matplotlib --quiet --break-system-packages 2>/dev/null
python plot_silicon_results.py --model llama 2>&1 || true

echo 'LLAMA COMPLETE'
ls -lh results/silicon_trust_*llama*.csv
")

echo "  Llama job submitted: $JOB2"

# ---- Job 3: Comparison (runs AFTER both complete) ----
JOB3=$(sbatch --parsable --job-name=silicon_compare \
    --partition=winston-gpu \
    --gres=gpu:0 \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:30:00 \
    --dependency=afterok:${JOB1}:${JOB2} \
    --output=logs/silicon_compare_%j.out \
    --error=logs/silicon_compare_%j.err \
    --wrap="
cd ~/Winston_Code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

echo '======================================================================'
echo 'MODEL COMPARISON — Qwen vs Llama'
echo '======================================================================'

python3 -c \"
import pandas as pd
from scipy import stats

qwen = pd.read_csv('results/silicon_trust_individual_qwen.csv')
llama = pd.read_csv('results/silicon_trust_individual_llama.csv')

print('=== POOLED COMPARISON ===')
for var in ['ppltrst', 'pplfair', 'pplhlp']:
    rq = qwen[(qwen['variable']==var) & (qwen['cntry']=='ALL')]['r_pearson'].values[0]
    rl = llama[(llama['variable']==var) & (llama['cntry']=='ALL')]['r_pearson'].values[0]
    print(f'  {var}: Qwen r={rq:.4f}, Llama r={rl:.4f}, Δ={rl-rq:+.4f}')

print()
print('=== COUNTRY-LEVEL COMPARISON ===')
for var in ['ppltrst', 'pplfair', 'pplhlp']:
    sq = pd.read_csv('results/silicon_trust_country_scatter_qwen.csv')
    sl = pd.read_csv('results/silicon_trust_country_scatter_llama.csv')
    sq_v = sq[sq['variable']==var].dropna(subset=['survey_mean','silicon_mean'])
    sl_v = sl[sl['variable']==var].dropna(subset=['survey_mean','silicon_mean'])
    rq, _ = stats.pearsonr(sq_v['survey_mean'], sq_v['silicon_mean'])
    rl, _ = stats.pearsonr(sl_v['survey_mean'], sl_v['silicon_mean'])
    print(f'  {var}: Qwen country r={rq:.3f}, Llama country r={rl:.3f}')

# Save comparison
comp = qwen[qwen['cntry']=='ALL'][['variable','r_pearson','var_ratio']].merge(
    llama[llama['cntry']=='ALL'][['variable','r_pearson','var_ratio']],
    on='variable', suffixes=('_qwen','_llama'))
comp.to_csv('results/silicon_rq8_comparison.csv', index=False)
print()
print('Saved: results/silicon_rq8_comparison.csv')
\"

echo '======================================================================'
echo 'ALL JOBS COMPLETE'
echo '======================================================================'
echo 'Results:'
ls -lh results/silicon_trust_*.csv results/silicon_rq8_*.csv
echo ''
echo 'Plots:'
ls -lh plots/*.png 2>/dev/null
")

echo "  Comparison job submitted: $JOB3 (depends on $JOB1 and $JOB2)"

echo ""
echo "======================================================================"
echo "SUBMITTED 3 JOBS:"
echo "  Job $JOB1: Qwen full run (GPU)"
echo "  Job $JOB2: Llama full run (GPU)"
echo "  Job $JOB3: Comparison (after both complete, no GPU)"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/silicon_full_qwen_${JOB1}.out"
echo "  tail -f logs/silicon_full_llama_${JOB2}.out"
echo "======================================================================"
