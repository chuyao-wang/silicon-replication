#!/usr/bin/env python3
"""
Silicon Sampling — Visualization Suite
=======================================
Generates all plots requested by supervisors (Patrick Sturgis & Daniel De Kadt).

Reads CSV outputs from silicon_sampling_extended_v5.py and produces:
  1. Country-level scatterplots: survey mean vs silicon mean (per variable + composite)
  2. Within-country individual-level correlation forest plot (heterogeneity)
  3. Variance compression panel (human vs silicon distributions by country)
  4. Overtime tracking: line plots per country showing survey vs silicon trends
  5. Backstory diagnostic: silicon mean by demographic subgroup vs survey

Usage:
    python plot_silicon_results.py                          # All available plots
    python plot_silicon_results.py --plot scatter           # Country scatterplots only
    python plot_silicon_results.py --plot forest            # Within-country r forest
    python plot_silicon_results.py --plot overtime           # Overtime line plots
    python plot_silicon_results.py --results_dir results    # Custom results path
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy import stats
import os
import argparse
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = "results"
PLOTS_DIR = "plots"

# Plot style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Variable labels for plot titles
VAR_LABELS = {
    'ppltrst': 'Generalised Trust\n("Most people can be trusted")',
    'pplfair': 'Perceived Fairness\n("Most people try to be fair")',
    'pplhlp': 'Perceived Helpfulness\n("Most people try to be helpful")',
    'composite': 'Composite Social Trust\n(Mean of 3 items)'
}

VAR_SHORT = {
    'ppltrst': 'Generalised Trust',
    'pplfair': 'Perceived Fairness',
    'pplhlp': 'Perceived Helpfulness',
    'composite': 'Composite Trust'
}

# Regional colour coding for scatterplot labels
REGION_COLOURS = {
    # Nordic
    'FI': '#2171b5', 'NO': '#2171b5', 'SE': '#2171b5', 'IS': '#2171b5',
    # Western Europe
    'NL': '#41ab5d', 'CH': '#41ab5d', 'DE': '#41ab5d', 'AT': '#41ab5d',
    'BE': '#41ab5d', 'FR': '#41ab5d', 'IE': '#41ab5d', 'GB': '#41ab5d',
    # Southern Europe
    'ES': '#fe9929', 'IT': '#fe9929', 'PT': '#fe9929', 'GR': '#fe9929',
    'CY': '#fe9929', 'HR': '#fe9929', 'SI': '#fe9929', 'ME': '#fe9929',
    'RS': '#fe9929',
    # Eastern Europe
    'PL': '#e7298a', 'HU': '#e7298a', 'SK': '#e7298a', 'BG': '#e7298a',
    'LT': '#e7298a', 'LV': '#e7298a', 'EE': '#e7298a', 'UA': '#e7298a',
    # Other
    'IL': '#969696'
}

REGION_LABELS = {
    '#2171b5': 'Nordic',
    '#41ab5d': 'Western Europe',
    '#fe9929': 'Southern & SE Europe',
    '#e7298a': 'Eastern Europe',
    '#969696': 'Other'
}

# Country code to short name (for plot labels)
COUNTRY_NAMES = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CH': 'Switzerland',
    'CY': 'Cyprus', 'DE': 'Germany', 'EE': 'Estonia', 'ES': 'Spain',
    'FI': 'Finland', 'FR': 'France', 'GB': 'UK', 'GR': 'Greece',
    'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland', 'IL': 'Israel',
    'IS': 'Iceland', 'IT': 'Italy', 'LT': 'Lithuania', 'LV': 'Latvia',
    'ME': 'Montenegro', 'NL': 'Netherlands', 'NO': 'Norway', 'PL': 'Poland',
    'PT': 'Portugal', 'RS': 'Serbia', 'SE': 'Sweden', 'SI': 'Slovenia',
    'SK': 'Slovakia', 'UA': 'Ukraine'
}


# ============================================================================
# PLOT 1: COUNTRY-LEVEL SCATTERPLOTS (Survey Mean vs Silicon Mean)
# ============================================================================

def plot_country_scatterplots(results_dir=RESULTS_DIR, model='qwen'):
    """
    Generate country-level scatterplots for each social trust variable.

    Produces a 2×2 panel: ppltrst, pplfair, pplhlp, composite.
    Each point = one country. Diagonal = perfect calibration.
    Points are colour-coded by European region and labelled with ISO codes.
    """
    # Load data
    scatter_file = os.path.join(results_dir, f"silicon_trust_country_scatter_{model}.csv")
    composite_file = os.path.join(results_dir, f"silicon_trust_composite_scatter_{model}.csv")

    if not os.path.exists(scatter_file):
        logging.warning(f"Scatter data not found: {scatter_file}")
        return None

    scatter_df = pd.read_csv(scatter_file)
    has_composite = os.path.exists(composite_file)
    if has_composite:
        composite_df = pd.read_csv(composite_file)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    variables = ['ppltrst', 'pplfair', 'pplhlp']

    for i, var_name in enumerate(variables):
        ax = axes[i // 2, i % 2]
        var_data = scatter_df[scatter_df['variable'] == var_name].dropna(
            subset=['survey_mean', 'silicon_mean'])

        _draw_scatter_panel(ax, var_data, 'survey_mean', 'silicon_mean',
                            'cntry', var_name, scale_range=(0, 10))

    # Composite panel
    ax = axes[1, 1]
    if has_composite:
        _draw_scatter_panel(ax, composite_df, 'survey_composite_mean',
                            'silicon_composite_mean', 'cntry', 'composite',
                            scale_range=(0, 10))
    else:
        ax.text(0.5, 0.5, 'Composite data\nnot available',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='grey')
        ax.set_title(VAR_LABELS.get('composite', 'Composite'))

    # Regional legend
    legend_handles = []
    seen_colours = set()
    for colour, label in REGION_LABELS.items():
        if colour not in seen_colours:
            legend_handles.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=colour,
                       markersize=8, label=label))
            seen_colours.add(colour)

    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'Country-Level Validation: Survey Mean vs Silicon Mean\n'
                 f'(Qwen 2.5-7B, ESS Round 11, {len(var_data)} countries)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"country_scatterplots_{model}.png")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved: {out_path}")
    return out_path


def _draw_scatter_panel(ax, data, x_col, y_col, country_col, var_name,
                         scale_range=(0, 10)):
    """Draw a single scatter panel with diagonal, regression line, and labels."""
    x = data[x_col].values
    y = data[y_col].values
    countries = data[country_col].values

    # Diagonal (perfect calibration)
    lo, hi = scale_range
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, linewidth=1, zorder=1)

    # Regression line
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() >= 3:
        slope, intercept, r_val, p_val, _ = stats.linregress(x[valid], y[valid])
        x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color='#e41a1c', linewidth=1.5, alpha=0.7, zorder=2)

        # Annotation
        sig_str = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        ax.text(0.05, 0.95,
                f'r = {r_val:.3f} ({sig_str})\nn = {valid.sum()}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='grey', alpha=0.8))

    # Scatter points with regional colours and country labels
    for j in range(len(x)):
        if np.isnan(x[j]) or np.isnan(y[j]):
            continue
        cntry = countries[j]
        colour = REGION_COLOURS.get(cntry, '#969696')
        ax.scatter(x[j], y[j], c=colour, s=50, edgecolors='white',
                   linewidth=0.5, zorder=3, alpha=0.85)
        # Country label (offset to avoid overlap)
        ax.annotate(cntry, (x[j], y[j]), fontsize=7, fontweight='bold',
                    xytext=(4, 4), textcoords='offset points',
                    color=colour, alpha=0.9)

    ax.set_xlabel('Survey Mean (ESS)')
    ax.set_ylabel('Silicon Mean (Qwen 2.5-7B)')
    ax.set_title(VAR_LABELS.get(var_name, var_name), fontsize=12)

    # Symmetric axes
    all_vals = np.concatenate([x[~np.isnan(x)], y[~np.isnan(y)]])
    if len(all_vals) > 0:
        margin = (all_vals.max() - all_vals.min()) * 0.15
        ax_lo = max(lo, all_vals.min() - margin)
        ax_hi = min(hi, all_vals.max() + margin)
        ax.set_xlim(ax_lo, ax_hi)
        ax.set_ylim(ax_lo, ax_hi)
    ax.set_aspect('equal', adjustable='box')


# ============================================================================
# PLOT 2: WITHIN-COUNTRY INDIVIDUAL-LEVEL CORRELATION FOREST PLOT
# ============================================================================

def plot_within_country_forest(results_dir=RESULTS_DIR, model='qwen'):
    """
    Forest plot showing individual-level Pearson r per country for ppltrst.
    Addresses Patrick's question: "are the silicon samples close to humans
    in some countries compared to others?"
    """
    indiv_file = os.path.join(results_dir, f"silicon_trust_individual_{model}.csv")
    if not os.path.exists(indiv_file):
        logging.warning(f"Individual results not found: {indiv_file}")
        return None

    indiv_df = pd.read_csv(indiv_file)

    fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)

    variables = ['ppltrst', 'pplfair', 'pplhlp']

    for i, var_name in enumerate(variables):
        ax = axes[i]
        var_data = indiv_df[
            (indiv_df['variable'] == var_name) &
            (indiv_df['cntry'] != 'ALL')
        ].copy()

        if len(var_data) == 0:
            ax.set_title(VAR_SHORT.get(var_name, var_name))
            continue

        # Sort by r_pearson
        var_data = var_data.sort_values('r_pearson', ascending=True)

        y_pos = range(len(var_data))
        countries = var_data['cntry'].values
        r_vals = var_data['r_pearson'].values
        n_vals = var_data['n_valid'].values

        # Colour bars by sign
        colours = ['#e41a1c' if r < 0 else '#4daf4a' for r in r_vals]

        ax.barh(y_pos, r_vals, color=colours, alpha=0.7, edgecolor='white',
                linewidth=0.5, height=0.7)

        # Vertical line at 0
        ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)

        # Pooled r line
        pooled = indiv_df[
            (indiv_df['variable'] == var_name) &
            (indiv_df['cntry'] == 'ALL')
        ]
        if len(pooled) > 0:
            pooled_r = pooled.iloc[0]['r_pearson']
            ax.axvline(x=pooled_r, color='#377eb8', linewidth=1.5,
                       linestyle='--', alpha=0.8)
            ax.text(pooled_r + 0.005, len(var_data) - 0.5,
                    f'Pooled r = {pooled_r:.3f}',
                    fontsize=9, color='#377eb8', fontweight='bold')

        # Y-axis labels with country names
        labels = [f"{c} (n={int(n)})" for c, n in zip(countries, n_vals)]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)

        ax.set_xlabel('Individual-Level Pearson r')
        ax.set_title(VAR_SHORT.get(var_name, var_name), fontsize=12,
                     fontweight='bold')

        # Add grid
        ax.grid(axis='x', alpha=0.2)
        ax.set_xlim(-0.25, 0.30)

    fig.suptitle(f'Within-Country Individual-Level Recovery\n'
                 f'(Qwen 2.5-7B, ESS R11, per-country Pearson r)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, f"within_country_forest_{model}.png")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved: {out_path}")
    return out_path


# ============================================================================
# PLOT 3: VARIANCE COMPRESSION — DISTRIBUTION COMPARISON
# ============================================================================

def plot_variance_compression(results_dir=RESULTS_DIR, model='qwen'):
    """
    Show distribution of human vs silicon responses for selected countries.
    Illustrates range compression and optimism bias.
    """
    raw_file = os.path.join(results_dir, f"silicon_trust_raw_{model}_seed888.csv")
    if not os.path.exists(raw_file):
        logging.warning(f"Raw data not found: {raw_file}")
        return None

    raw_df = pd.read_csv(raw_file)

    # Select diverse countries: high trust (FI), medium (DE, FR), low (UA, BG)
    highlight_countries = ['FI', 'NO', 'DE', 'FR', 'PL', 'UA']
    available = [c for c in highlight_countries if c in raw_df['cntry'].unique()]

    if len(available) < 3:
        available = sorted(raw_df['cntry'].unique())[:6]

    fig, axes = plt.subplots(len(available), 3, figsize=(14, 3 * len(available)),
                              sharex=True)

    variables = ['ppltrst', 'pplfair', 'pplhlp']

    for i, cntry in enumerate(available):
        for j, var_name in enumerate(variables):
            ax = axes[i, j] if len(available) > 1 else axes[j]

            cntry_var = raw_df[
                (raw_df['cntry'] == cntry) &
                (raw_df['variable'] == var_name)
            ]

            human = pd.to_numeric(cntry_var['human_response'], errors='coerce').dropna()
            silicon = pd.to_numeric(cntry_var['silicon_response'], errors='coerce').dropna()

            bins = np.arange(-0.5, 11.5, 1)

            ax.hist(human, bins=bins, alpha=0.5, density=True, color='#377eb8',
                    label='Survey', edgecolor='white', linewidth=0.5)
            ax.hist(silicon, bins=bins, alpha=0.5, density=True, color='#e41a1c',
                    label='Silicon', edgecolor='white', linewidth=0.5)

            # Means
            h_mean = human.mean()
            s_mean = silicon.mean()
            ax.axvline(h_mean, color='#377eb8', linestyle='--', linewidth=1.2)
            ax.axvline(s_mean, color='#e41a1c', linestyle='--', linewidth=1.2)

            if i == 0:
                ax.set_title(VAR_SHORT.get(var_name, var_name), fontsize=11,
                             fontweight='bold')
            if j == 0:
                cntry_name = COUNTRY_NAMES.get(cntry, cntry)
                ax.set_ylabel(cntry_name, fontsize=10, fontweight='bold')

            ax.text(0.97, 0.95,
                    f'H: {h_mean:.1f} ({human.std():.1f})\n'
                    f'S: {s_mean:.1f} ({silicon.std():.1f})',
                    transform=ax.transAxes, fontsize=8,
                    ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='grey', alpha=0.8))

            if i == 0 and j == 2:
                ax.legend(fontsize=8, loc='upper left')

    fig.suptitle(f'Distribution Comparison: Survey vs Silicon Responses\n'
                 f'(Qwen 2.5-7B, ESS R11)',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.supxlabel('Response (0–10 scale)', fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, f"variance_compression_{model}.png")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved: {out_path}")
    return out_path


# ============================================================================
# PLOT 4: OVERTIME TRACKING — SURVEY vs SILICON TRENDS
# ============================================================================

def plot_overtime_tracking(results_dir=RESULTS_DIR, model='qwen'):
    """
    Line plots: survey mean vs silicon mean over ESS rounds, per country.
    Shows temporal tracking failure.
    """
    agg_file = os.path.join(results_dir, f"silicon_overtime_agg_{model}.csv")
    if not os.path.exists(agg_file):
        logging.warning(f"Overtime data not found: {agg_file}")
        return None

    agg_df = pd.read_csv(agg_file)

    # Focus on ppltrst (strongest cross-national signal)
    var_name = 'ppltrst'
    var_data = agg_df[agg_df['variable'] == var_name]

    countries = sorted(var_data['cntry'].unique())
    n_countries = len(countries)
    n_cols = 5
    n_rows = int(np.ceil(n_countries / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows),
                              sharex=True, sharey=True)
    axes_flat = axes.flatten() if n_countries > 1 else [axes]

    for i, cntry in enumerate(countries):
        ax = axes_flat[i]
        cntry_data = var_data[var_data['cntry'] == cntry].sort_values('survey_year')

        years = cntry_data['survey_year'].values
        survey_means = cntry_data['survey_mean'].values
        silicon_means = cntry_data['silicon_mean'].values

        ax.plot(years, survey_means, 'o-', color='#377eb8', linewidth=1.5,
                markersize=5, label='Survey', zorder=3)
        ax.plot(years, silicon_means, 's--', color='#e41a1c', linewidth=1.5,
                markersize=5, label='Silicon', zorder=3)

        # Within-country correlation
        valid = ~(np.isnan(survey_means) | np.isnan(silicon_means))
        if valid.sum() >= 3:
            r, p = stats.pearsonr(survey_means[valid], silicon_means[valid])
            sig = '*' if p < 0.05 else ''
            ax.text(0.05, 0.05, f'r = {r:.2f}{sig}',
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                              edgecolor='grey', alpha=0.8))

        cntry_name = COUNTRY_NAMES.get(cntry, cntry)
        ax.set_title(cntry_name, fontsize=10, fontweight='bold')
        ax.grid(alpha=0.2)

        if i == 0:
            ax.legend(fontsize=7, loc='upper right')

    # Hide unused axes
    for j in range(len(countries), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f'Temporal Tracking: Generalised Trust (ppltrst)\n'
                 f'Survey Mean vs Silicon Mean across ESS Rounds (2002–2023)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.supxlabel('Survey Year', fontsize=12)
    fig.supylabel('Mean Response (0–10)', fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, f"overtime_tracking_{model}.png")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved: {out_path}")
    return out_path


# ============================================================================
# PLOT 5: OVERTIME POOLED SCATTER (90 Country×Round Cells)
# ============================================================================

def plot_overtime_pooled_scatter(results_dir=RESULTS_DIR, model='qwen'):
    """
    Pooled scatterplot for overtime analysis: each point = one country×round cell.
    Colour-coded by country, showing that between-country variation drives the
    pooled correlation while within-country trajectories are flat/random.
    """
    agg_file = os.path.join(results_dir, f"silicon_overtime_agg_{model}.csv")
    if not os.path.exists(agg_file):
        logging.warning(f"Overtime data not found: {agg_file}")
        return None

    agg_df = pd.read_csv(agg_file)

    variables = ['ppltrst', 'pplfair', 'pplhlp']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for i, var_name in enumerate(variables):
        ax = axes[i]
        var_data = agg_df[agg_df['variable'] == var_name].dropna(
            subset=['survey_mean', 'silicon_mean'])

        # Each point = country×round cell, coloured by country
        countries = sorted(var_data['cntry'].unique())
        cmap = plt.cm.tab20
        colour_map = {c: cmap(j / max(len(countries) - 1, 1))
                      for j, c in enumerate(countries)}

        for cntry in countries:
            cd = var_data[var_data['cntry'] == cntry].sort_values('survey_year')
            colour = REGION_COLOURS.get(cntry, '#969696')

            ax.scatter(cd['survey_mean'], cd['silicon_mean'],
                       c=colour, s=40, alpha=0.7, edgecolors='white',
                       linewidth=0.5, zorder=3)

            # Connect points within same country with thin lines
            ax.plot(cd['survey_mean'], cd['silicon_mean'],
                    color=colour, alpha=0.3, linewidth=0.8, zorder=2)

            # Label the latest round point
            if len(cd) > 0:
                last = cd.iloc[-1]
                ax.annotate(cntry, (last['survey_mean'], last['silicon_mean']),
                            fontsize=6, color=colour, alpha=0.8,
                            xytext=(3, 3), textcoords='offset points')

        # Diagonal
        ax.plot([0, 10], [0, 10], 'k--', alpha=0.2, linewidth=1)

        # Regression
        x = var_data['survey_mean'].values
        y = var_data['silicon_mean'].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() >= 5:
            slope, intercept, r_val, p_val, _ = stats.linregress(x[valid], y[valid])
            x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ax.plot(x_fit, slope * x_fit + intercept, color='#e41a1c',
                    linewidth=1.5, alpha=0.7)

            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
            ax.text(0.05, 0.95,
                    f'Pooled r = {r_val:.3f} ({sig})\nn = {valid.sum()} cells',
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='grey', alpha=0.8))

        ax.set_xlabel('Survey Mean')
        ax.set_ylabel('Silicon Mean')
        ax.set_title(VAR_SHORT.get(var_name, var_name), fontsize=12,
                     fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.15)

    fig.suptitle(f'Overtime Pooled Scatter: 90 Country×Round Cells\n'
                 f'(Lines connect same country across rounds; '
                 f'within-country trajectories are flat)',
                 fontsize=13, fontweight='bold', y=1.04)
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, f"overtime_pooled_scatter_{model}.png")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved: {out_path}")
    return out_path


# ============================================================================
# PLOT 6: WITHIN-COUNTRY r HETEROGENEITY — CORRELATION WITH COUNTRY FEATURES
# ============================================================================

def plot_r_vs_country_features(results_dir=RESULTS_DIR, model='qwen'):
    """
    Explore why some countries have higher individual-level r than others.
    Scatter: per-country r vs (a) sample size, (b) survey mean, (c) survey SD.
    """
    indiv_file = os.path.join(results_dir, f"silicon_trust_individual_{model}.csv")
    scatter_file = os.path.join(results_dir, f"silicon_trust_country_scatter_{model}.csv")

    if not (os.path.exists(indiv_file) and os.path.exists(scatter_file)):
        logging.warning("Required files not found for r-heterogeneity plot")
        return None

    indiv_df = pd.read_csv(indiv_file)
    scatter_df = pd.read_csv(scatter_file)

    var_name = 'ppltrst'

    # Per-country r
    country_r = indiv_df[
        (indiv_df['variable'] == var_name) &
        (indiv_df['cntry'] != 'ALL')
    ][['cntry', 'r_pearson', 'n_valid', 'var_ratio']].copy()

    # Survey features
    country_stats = scatter_df[scatter_df['variable'] == var_name][
        ['cntry', 'survey_mean', 'survey_sd']
    ].copy()

    merged = country_r.merge(country_stats, on='cntry')

    if len(merged) < 5:
        logging.warning("Too few countries for heterogeneity plot")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    predictors = [
        ('survey_mean', 'Survey Mean Trust'),
        ('survey_sd', 'Survey SD'),
        ('var_ratio', 'Variance Ratio (Silicon/Human)')
    ]

    for i, (pred, label) in enumerate(predictors):
        ax = axes[i]
        x = merged[pred].values
        y = merged['r_pearson'].values

        for j in range(len(x)):
            cntry = merged.iloc[j]['cntry']
            colour = REGION_COLOURS.get(cntry, '#969696')
            ax.scatter(x[j], y[j], c=colour, s=50, edgecolors='white',
                       linewidth=0.5, zorder=3)
            ax.annotate(cntry, (x[j], y[j]), fontsize=7, color=colour,
                        xytext=(3, 3), textcoords='offset points')

        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() >= 5:
            r, p = stats.pearsonr(x[valid], y[valid])
            sig = '*' if p < 0.05 else 'n.s.'
            ax.text(0.05, 0.95, f'r = {r:.3f} ({sig})',
                    transform=ax.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='grey', alpha=0.8))

        ax.axhline(y=0, color='grey', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel('Individual-Level r (ppltrst)')
        ax.grid(alpha=0.15)

    fig.suptitle('What Predicts Within-Country Recovery?\n'
                 '(Individual-level r for ppltrst by country features)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, f"r_heterogeneity_{model}.png")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved: {out_path}")
    return out_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Silicon Sampling — Visualization Suite')
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR,
                        help='Directory containing CSV results')
    parser.add_argument('--model', type=str, default='qwen',
                        help='Model name (qwen or llama)')
    parser.add_argument('--plot', type=str, default='all',
                        choices=['all', 'scatter', 'forest', 'variance',
                                 'overtime', 'overtime_scatter', 'heterogeneity'],
                        help='Which plot to generate')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    os.makedirs(PLOTS_DIR, exist_ok=True)

    plot_funcs = {
        'scatter': lambda: plot_country_scatterplots(args.results_dir, args.model),
        'forest': lambda: plot_within_country_forest(args.results_dir, args.model),
        'variance': lambda: plot_variance_compression(args.results_dir, args.model),
        'overtime': lambda: plot_overtime_tracking(args.results_dir, args.model),
        'overtime_scatter': lambda: plot_overtime_pooled_scatter(args.results_dir, args.model),
        'heterogeneity': lambda: plot_r_vs_country_features(args.results_dir, args.model),
    }

    if args.plot == 'all':
        for name, func in plot_funcs.items():
            logging.info(f"Generating: {name}")
            try:
                func()
            except Exception as e:
                logging.error(f"Failed to generate {name}: {e}")
    else:
        plot_funcs[args.plot]()

    logging.info("Done. Plots saved to: " + os.path.abspath(PLOTS_DIR))


if __name__ == "__main__":
    main()
