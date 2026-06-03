"""
Cultural Homogenisation Over Time
Measures vocabulary richness, phrase repetition, and thematic diversity
across decades to evaluate whether popular music lyrics have homogenised
over 1950–2019. Findings framed for an AI Safety audience.

Run from src/ directory:
    python homogenisation_analysis.py                # default: 95th percentile
    python homogenisation_analysis.py --percentile 75
"""

import argparse
import os

# Silence HuggingFace tokenizer fork-parallelism warning and the macOS
# "leaked semaphore" message at shutdown. Must be set before any import that
# pulls in `tokenizers` (i.e. before sentence_transformers).
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
# Belt-and-braces against the Intel MKL / LLVM OpenMP "duplicate library" segfault
# that can occur when torch and sklearn link to different OpenMP runtimes on macOS.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# IMPORTANT: import sentence_transformers (which pulls in torch) BEFORE numpy /
# scipy / sklearn. On macOS, sklearn ships its own libomp.dylib; if it loads
# before torch's libomp, the two collide and torch segfaults inside
# `_load_from_state_dict`. Loading torch first means everything else
# (numpy/scipy/sklearn) shares torch's OpenMP runtime and the conflict is avoided.
from sentence_transformers import SentenceTransformer

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from scipy import stats

# ── CLI ───────────────────────────────────────────────────────────────────────
# Maps a view-count percentile filter to the corresponding cleaned dataset.
# Add new entries here to support further percentiles without code changes.
DATASET_MAP = {
    95: '../data/processed/song_lyrics_clean_df.csv',
    75: '../data/processed/song_lyrics_clean_75th_df.csv',
}

parser = argparse.ArgumentParser(description='Cultural Homogenisation Over Time')
parser.add_argument('--percentile', type=int, default=95,
                    choices=sorted(DATASET_MAP.keys()),
                    help='View-count percentile filter (selects input dataset). '
                         'Output filenames are suffixed accordingly.')
args = parser.parse_args()

PERCENTILE = args.percentile
SUFFIX = f'_{PERCENTILE}th'
DATA_PATH = DATASET_MAP[PERCENTILE]

# ── Setup ─────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUT_DIR = '../data/processed/'
SAMPLE_OUT_DIR = '../outputs/'
LOW_CONFIDENCE_THRESHOLD = 100
SAMPLE_PER_DECADE = 2000
DIVERSITY_SAMPLE_CAP = 200
SENTENCE_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'  # 384-dim, small, fast; swap for 'sentence-transformers/all-mpnet-base-v2' for higher quality
# Force CPU for sentence-transformers. PyTorch's MPS (Apple Silicon GPU) backend
# has been observed to segfault with this model on macOS; CPU is slower but reliable.
# Set to 'mps' or 'cuda' to experiment with hardware acceleration.
SENTENCE_DEVICE = 'mps'

print(f"Running homogenisation analysis at the {PERCENTILE}th percentile")
print(f"  input : {DATA_PATH}")
print(f"  suffix: {SUFFIX}")

os.makedirs(SAMPLE_OUT_DIR, exist_ok=True)


def report(label, frame, cols=None):
    print(f"\n[{label}] shape={frame.shape}")
    target_cols = cols if cols is not None else frame.columns
    nulls = frame[target_cols].isnull().sum()
    print(f"[{label}] nulls:\n{nulls}")


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
report('loaded', df, ['clean_lyrics', 'year', 'views', 'genre', 'artist'])

#Remove data before 2020 - dataset after that is a bit meh
print("Filter for music before 2020...")
df= df[(df['year'] < 2020)]

#Remove rap from the analysis
print("Remove rap from the analysis...")
df = df[(df['genre'] != 'rap')]

# ── Step 1 — Distribution check ───────────────────────────────────────────────
print("\n--- Step 1: Decade distribution ---")
df['decade'] = (df['year'] // 5 * 5).astype(int).astype(str) + 's'
decade_counts = df['decade'].value_counts().sort_index()
print(decade_counts)

low_confidence_decades = set(decade_counts[decade_counts < LOW_CONFIDENCE_THRESHOLD].index)
if low_confidence_decades:
    print(f"[LOW CONFIDENCE] decades with <{LOW_CONFIDENCE_THRESHOLD} songs: "
          f"{sorted(low_confidence_decades)}")

fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#d62728' if d in low_confidence_decades else '#4c72b0'
          for d in decade_counts.index]
ax.bar(decade_counts.index, decade_counts.values, color=colors, edgecolor='white')
ax.set_xlabel('Decade', fontsize=11)
ax.set_ylabel('Number of Songs', fontsize=11)
ax.set_title('Available Songs per Decade (red = low confidence, n<100)', fontsize=13)
for x_label, val in zip(decade_counts.index, decade_counts.values):
    ax.text(x_label, val, f'{val:,}', ha='center', va='bottom', fontsize=9)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}decade_distribution{SUFFIX}.png', dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}decade_distribution{SUFFIX}.png")


# ── Step 2 — Stratified sample by decade ──────────────────────────────────────
print("\n--- Step 2: Stratified sample by decade ---")
sampled = (df.groupby('decade', group_keys=False)
             .apply(lambda x: x.sample(min(len(x), SAMPLE_PER_DECADE),
                                       random_state=RANDOM_SEED)))
sampled = sampled.reset_index(drop=True)
report('sampled', sampled, ['clean_lyrics', 'year', 'views', 'genre', 'decade'])
print(sampled['decade'].value_counts().sort_index())

sampled.to_csv(f'{SAMPLE_OUT_DIR}sampled_dataset{SUFFIX}.csv', index=False)
print(f"Saved {SAMPLE_OUT_DIR}sampled_dataset{SUFFIX}.csv")


# ── Step 3 — Feature engineering ──────────────────────────────────────────────
print("\n--- Step 3: Feature engineering ---")


def ttr(text):
    tokens = str(text).split()
    return len(set(tokens)) / len(tokens) if tokens else 0


def mattr(text, window=50):
    tokens = str(text).split()
    if len(tokens) < window:
        return ttr(text)
    ratios = []
    for i in range(len(tokens) - window + 1):
        chunk = tokens[i:i + window]
        ratios.append(len(set(chunk)) / window)
    return float(np.mean(ratios)) if ratios else 0.0


def repetition_rate(text, n=3):
    tokens = str(text).split()
    if len(tokens) < n:
        return 0
    grams = list(ngrams(tokens, n))
    counts = Counter(grams)
    repeated = sum(v for v in counts.values() if v > 1)
    return repeated / len(grams)


print("Computing TTR...")
sampled['ttr'] = sampled['clean_lyrics'].map(ttr)
print("Computing MATTR (window=50)...")
sampled['mattr'] = sampled['clean_lyrics'].map(mattr)
print("Computing fivegram repetition rate...")
sampled['rep_fivegram'] = sampled['clean_lyrics'].map(lambda x: repetition_rate(x, n=5))
print("Computing fourgram repetition rate...")
sampled['rep_fourgram'] = sampled['clean_lyrics'].map(lambda x: repetition_rate(x, n=4))
print("Computing trigram repetition rate...")
sampled['rep_trigram'] = sampled['clean_lyrics'].map(lambda x: repetition_rate(x, n=3))

report('features', sampled, ['ttr', 'mattr', 'rep_trigram', 'rep_fourgram', 'rep_fivegram'])


# ── Step 3 (cont.) — Thematic diversity (TF-IDF cosine distance) ──────────────
print("\n--- Step 3b: Thematic diversity ---")
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(sampled['clean_lyrics'].fillna(''))
sampled['tfidf_index'] = range(len(sampled))


def decade_diversity(group, matrix, cap=DIVERSITY_SAMPLE_CAP, weights=None):
    """Mean pairwise cosine distance for songs in a decade.

    weights: optional array aligned with `group` for weighted sampling.
    Returns NaN if too few songs to form a pair.
    """
    idx = group['tfidf_index'].values
    if len(idx) < 2:
        return np.nan
    sample_size = min(len(idx), cap)
    rng = np.random.default_rng(RANDOM_SEED)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum() if w.sum() > 0 else None
        sample_idx = rng.choice(len(idx), sample_size, replace=False, p=w)
    else:
        sample_idx = rng.choice(len(idx), sample_size, replace=False)
    subset = matrix[idx[sample_idx]]
    dist = cosine_distances(subset)
    return float(dist[np.triu_indices(sample_size, k=1)].mean())


diversity = {}
for decade, group in sampled.groupby('decade'):
    diversity[decade] = decade_diversity(group, tfidf_matrix)
    print(f"  {decade}: thematic_diversity={diversity[decade]:.4f}  (n={len(group)})")

diversity_df = pd.DataFrame.from_dict(diversity, orient='index',
                                     columns=['thematic_diversity'])
diversity_df.index.name = 'decade'


# ── Step 3 (cont.) — Thematic diversity (sentence-transformer embeddings) ────
# Complementary semantic-similarity view: TF-IDF measures vocabulary overlap;
# sentence embeddings capture meaning, so older/newer songs that share themes
# but use different vocabulary are recognised as related.
print("\n--- Step 3c: Thematic diversity (sentence embeddings) ---")
print(f"Loading sentence-transformer model: {SENTENCE_MODEL_NAME} (device={SENTENCE_DEVICE})")
st_model = SentenceTransformer(SENTENCE_MODEL_NAME, device=SENTENCE_DEVICE)
print(f"Encoding {len(sampled)} lyrics (may take 5–15 minutes on CPU)...")
embed_matrix = st_model.encode(
    sampled['clean_lyrics'].fillna('').tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
)
print(f"Embedding matrix shape: {embed_matrix.shape}")

diversity_embed = {}
for decade, group in sampled.groupby('decade'):
    diversity_embed[decade] = decade_diversity(group, embed_matrix)
    print(f"  {decade}: thematic_diversity_embed={diversity_embed[decade]:.4f}")

diversity_embed_df = pd.DataFrame.from_dict(diversity_embed, orient='index',
                                            columns=['thematic_diversity_embed'])
diversity_embed_df.index.name = 'decade'


# ── Step 4 — Decade aggregation ───────────────────────────────────────────────
print("\n--- Step 4: Decade aggregation ---")
metric_cols = ['ttr', 'mattr', 'rep_trigram', 'rep_fourgram', 'rep_fivegram']
decade_stats = sampled.groupby('decade')[metric_cols].agg(['mean', 'std'])
# Assign diversity / scalar columns directly into the 2-level column shape.
# (Joining a MultiIndex frame here collapses decade_stats to a single column
# level, which breaks the subsequent join — so assign by tuple key instead.)
decade_stats[('thematic_diversity', '')] = diversity_df['thematic_diversity']
decade_stats[('thematic_diversity_embed', '')] = diversity_embed_df['thematic_diversity_embed']
decade_stats[('n_songs', '')] = sampled.groupby('decade').size()
decade_stats[('low_confidence', '')] = decade_stats.index.isin(low_confidence_decades)
print(decade_stats)
decade_stats.to_csv(f'{OUT_DIR}homogenisation_decade_stats{SUFFIX}.csv')
print(f"Saved {OUT_DIR}homogenisation_decade_stats{SUFFIX}.csv")


# ── Step 5 — Trend testing (Kendall's tau) ────────────────────────────────────
print("\n--- Step 5: Trend testing (Kendall's tau) ---")
decade_order = sorted(sampled['decade'].unique())
x = list(range(len(decade_order)))

trend_rows = []
for metric in metric_cols + ['thematic_diversity', 'thematic_diversity_embed']:
    if metric == 'thematic_diversity':
        y = [diversity_df.loc[d, 'thematic_diversity'] for d in decade_order]
    elif metric == 'thematic_diversity_embed':
        y = [diversity_embed_df.loc[d, 'thematic_diversity_embed'] for d in decade_order]
    else:
        y = [decade_stats[(metric, 'mean')][d] for d in decade_order]
    tau, p = stats.kendalltau(x, y)
    print(f"  {metric}: tau={tau:.3f}, p={p:.4f}")
    trend_rows.append({'metric': metric, 'tau': tau, 'p_value': p,
                       'n_decades': len(decade_order)})

trend_df = pd.DataFrame(trend_rows)
trend_df.to_csv(f'{OUT_DIR}trend_test_results{SUFFIX}.csv', index=False)
print(f"Saved {OUT_DIR}trend_test_results{SUFFIX}.csv")


# ── Step 6 — Visualisation ────────────────────────────────────────────────────
print("\n--- Step 6: Visualisation ---")


def annotate_low_confidence(ax, decades, ymax):
    for i, d in enumerate(decades):
        if d in low_confidence_decades:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.15, color='red')
            ax.text(i, ymax, 'low n', ha='center', va='bottom',
                    fontsize=8, color='red')


def plot_metric_band(metric, ylabel, title, out_path,
                     stats_table, decades):
    means = [stats_table[(metric, 'mean')][d] for d in decades]
    stds = [stats_table[(metric, 'std')][d] for d in decades]
    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(decades, means, marker='o', color='#1f77b4', label='Mean')
    ax.fill_between(decades, means - stds, means + stds, alpha=0.2,
                    color='#1f77b4', label='± 1 std dev')
    ax.set_xlabel('Decade', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    annotate_low_confidence(ax, decades, (means + stds).max())
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# TTR (and MATTR) — vocabulary richness
fig, ax = plt.subplots(figsize=(9, 5))
ttr_means = np.array([decade_stats[('ttr', 'mean')][d] for d in decade_order])
ttr_stds = np.array([decade_stats[('ttr', 'std')][d] for d in decade_order])
mattr_means = np.array([decade_stats[('mattr', 'mean')][d] for d in decade_order])
mattr_stds = np.array([decade_stats[('mattr', 'std')][d] for d in decade_order])

ax.plot(decade_order, ttr_means, marker='o', color='#1f77b4', label='TTR (mean)')
ax.fill_between(decade_order, ttr_means - ttr_stds, ttr_means + ttr_stds,
                alpha=0.15, color='#1f77b4')
ax.plot(decade_order, mattr_means, marker='s', color='#2ca02c',
        label='MATTR window=50 (mean)')
ax.fill_between(decade_order, mattr_means - mattr_stds, mattr_means + mattr_stds,
                alpha=0.15, color='#2ca02c')
ax.set_xlabel('Decade', fontsize=11)
ax.set_ylabel('Vocabulary richness (proportion of unique tokens)', fontsize=11)
ax.set_title('Vocabulary Richness in Popular Music Lyrics by Decade', fontsize=13)
ax.legend(fontsize=9)
annotate_low_confidence(ax, decade_order, max((ttr_means + ttr_stds).max(),
                                               (mattr_means + mattr_stds).max()))
fig.tight_layout()
fig.savefig(f'{OUT_DIR}ttr_by_decade{SUFFIX}.png', dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}ttr_by_decade{SUFFIX}.png")

# Repetition — combine trigrams/fourgrams/fivegrams on a single chart
fig, ax = plt.subplots(figsize=(9, 5))
rep_palette = {'rep_trigram': '#1f77b4',
               'rep_fourgram': '#ff7f0e',
               'rep_fivegram': '#9467bd'}
labels = {'rep_trigram': 'Trigram repetition',
          'rep_fourgram': 'Fourgram repetition',
          'rep_fivegram': 'Fivegram repetition'}
for metric, colour in rep_palette.items():
    means = np.array([decade_stats[(metric, 'mean')][d] for d in decade_order])
    stds = np.array([decade_stats[(metric, 'std')][d] for d in decade_order])
    ax.plot(decade_order, means, marker='o', color=colour, label=labels[metric])
    ax.fill_between(decade_order, means - stds, means + stds, alpha=0.12, color=colour)
ax.set_xlabel('Decade', fontsize=11)
ax.set_ylabel('Repeated n-gram proportion within song', fontsize=11)
ax.set_title('Phrase Repetition in Popular Music Lyrics by Decade', fontsize=13)
ax.legend(fontsize=9)
all_top = np.concatenate([
    np.array([decade_stats[(m, 'mean')][d] for d in decade_order]) +
    np.array([decade_stats[(m, 'std')][d] for d in decade_order])
    for m in rep_palette
])
annotate_low_confidence(ax, decade_order, all_top.max())
fig.tight_layout()
fig.savefig(f'{OUT_DIR}repetition_by_decade{SUFFIX}.png', dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}repetition_by_decade{SUFFIX}.png")

# Thematic diversity — side-by-side comparison of TF-IDF vs sentence embeddings.
# TF-IDF measures vocabulary overlap; embeddings measure semantic similarity.
# Absolute scales differ (TF-IDF on sparse vectors sits near 1.0, embeddings lower),
# so each method gets its own y-axis in a 1x2 panel; the *trend shape* is what to compare.
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
div_vals_tfidf = [diversity_df.loc[d, 'thematic_diversity'] for d in decade_order]
div_vals_embed = [diversity_embed_df.loc[d, 'thematic_diversity_embed'] for d in decade_order]

axes[0].plot(decade_order, div_vals_tfidf, marker='o', color='#d62728',
             label='TF-IDF cosine distance')
axes[0].set_xlabel('Decade', fontsize=11)
axes[0].set_ylabel('Mean pairwise cosine distance', fontsize=11)
axes[0].set_title('Thematic Diversity (TF-IDF / vocabulary overlap)', fontsize=12)
axes[0].legend(fontsize=9)
annotate_low_confidence(axes[0], decade_order, max(div_vals_tfidf))

axes[1].plot(decade_order, div_vals_embed, marker='s', color='#1f77b4',
             label=f'Sentence embedding cosine distance ({SENTENCE_MODEL_NAME})')
axes[1].set_xlabel('Decade', fontsize=11)
axes[1].set_ylabel('Mean pairwise cosine distance', fontsize=11)
axes[1].set_title('Thematic Diversity (Sentence embeddings / semantic)', fontsize=12)
axes[1].legend(fontsize=9)
annotate_low_confidence(axes[1], decade_order, max(div_vals_embed))

for ax in axes:
    ax.tick_params(axis='x', rotation=30)

fig.suptitle('Thematic Diversity of Popular Music Lyrics by Decade — Two Methods',
             fontsize=13)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}thematic_diversity_by_decade{SUFFIX}.png', dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}thematic_diversity_by_decade{SUFFIX}.png")


# ── Step 7 — View-count-weighted re-run ───────────────────────────────────────
print("\n--- Step 7: View-count weighted analysis ---")
sampled['log_views_weight'] = np.log1p(sampled['views'].fillna(0))


def weighted_mean(values, weights):
    mask = ~np.isnan(values) & ~np.isnan(weights)
    if not mask.any() or weights[mask].sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def weighted_std(values, weights):
    mask = ~np.isnan(values) & ~np.isnan(weights)
    if not mask.any() or weights[mask].sum() == 0:
        return np.nan
    v = values[mask]
    w = weights[mask]
    mean = np.average(v, weights=w)
    variance = np.average((v - mean) ** 2, weights=w)
    return float(np.sqrt(variance))


weighted_rows = []
for decade, group in sampled.groupby('decade'):
    weights = group['log_views_weight'].to_numpy()
    row = {'decade': decade, 'n_songs': len(group),
           'low_confidence': decade in low_confidence_decades}
    for m in metric_cols:
        vals = group[m].to_numpy()
        row[(m, 'mean')] = weighted_mean(vals, weights)
        row[(m, 'std')] = weighted_std(vals, weights)
    weighted_rows.append(row)

weighted_stats = pd.DataFrame(weighted_rows).set_index('decade')
# Reorder columns to mimic decade_stats layout: MultiIndex(metric, agg)
weighted_metric_cols = [(m, agg) for m in metric_cols for agg in ('mean', 'std')]
weighted_stats = weighted_stats[['n_songs', 'low_confidence'] + weighted_metric_cols]
weighted_stats.columns = pd.MultiIndex.from_tuples(
    [('n_songs', '') if c == 'n_songs'
     else ('low_confidence', '') if c == 'low_confidence'
     else c for c in weighted_stats.columns]
)

# Weighted thematic diversity: weighted sampling of songs by log(views+1)
# Computed for both TF-IDF and sentence-embedding matrices.
weighted_diversity = {}
weighted_diversity_embed = {}
for decade, group in sampled.groupby('decade'):
    w = group['log_views_weight'].to_numpy()
    weighted_diversity[decade] = decade_diversity(group, tfidf_matrix, weights=w)
    weighted_diversity_embed[decade] = decade_diversity(group, embed_matrix, weights=w)
    print(f"  {decade}: weighted thematic_diversity={weighted_diversity[decade]:.4f}"
          f"  | weighted thematic_diversity_embed={weighted_diversity_embed[decade]:.4f}")

weighted_div_df = pd.DataFrame.from_dict(weighted_diversity, orient='index',
                                        columns=['thematic_diversity'])
weighted_div_df.index.name = 'decade'

weighted_div_embed_df = pd.DataFrame.from_dict(weighted_diversity_embed, orient='index',
                                               columns=['thematic_diversity_embed'])
weighted_div_embed_df.index.name = 'decade'

# Assign by tuple key to preserve the 2-level column shape (a MultiIndex join
# would collapse weighted_stats to a single column level — see Step 4 above).
weighted_stats[('thematic_diversity', '')] = weighted_div_df['thematic_diversity']
weighted_stats[('thematic_diversity_embed', '')] = weighted_div_embed_df['thematic_diversity_embed']

weighted_stats.to_csv(f'{OUT_DIR}homogenisation_decade_stats_weighted{SUFFIX}.csv')
print(f"Saved {OUT_DIR}homogenisation_decade_stats_weighted{SUFFIX}.csv")

# Weighted trend tests
weighted_trend_rows = []
for metric in metric_cols + ['thematic_diversity', 'thematic_diversity_embed']:
    if metric == 'thematic_diversity':
        y = [weighted_div_df.loc[d, 'thematic_diversity'] for d in decade_order]
    elif metric == 'thematic_diversity_embed':
        y = [weighted_div_embed_df.loc[d, 'thematic_diversity_embed'] for d in decade_order]
    else:
        y = [weighted_stats[(metric, 'mean')][d] for d in decade_order]
    tau, p = stats.kendalltau(x, y)
    print(f"  weighted {metric}: tau={tau:.3f}, p={p:.4f}")
    weighted_trend_rows.append({'metric': metric, 'tau': tau, 'p_value': p,
                                'n_decades': len(decade_order)})

weighted_trend_df = pd.DataFrame(weighted_trend_rows)
weighted_trend_df.to_csv(f'{OUT_DIR}trend_test_results_weighted{SUFFIX}.csv', index=False)
print(f"Saved {OUT_DIR}trend_test_results_weighted{SUFFIX}.csv")

# Side-by-side comparison: unweighted vs weighted tau for each metric
comparison = trend_df.rename(columns={'tau': 'tau_unweighted',
                                      'p_value': 'p_unweighted'})
comparison = comparison.merge(
    weighted_trend_df.rename(columns={'tau': 'tau_weighted',
                                      'p_value': 'p_weighted'}),
    on=['metric', 'n_decades']
)
comparison['tau_delta'] = comparison['tau_weighted'] - comparison['tau_unweighted']
comparison.to_csv(f'{OUT_DIR}trend_test_results_comparison{SUFFIX}.csv', index=False)
print(f"Saved {OUT_DIR}trend_test_results_comparison{SUFFIX}.csv")
print(comparison)

# Weighted versions of the charts
def plot_weighted_metric(metric_keys, labels, ylabel, title, out_path,
                         stats_table, decades):
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    all_top = []
    for i, m in enumerate(metric_keys):
        means = np.array([stats_table[(m, 'mean')][d] for d in decades])
        stds = np.array([stats_table[(m, 'std')][d] for d in decades])
        colour = palette[i % len(palette)]
        ax.plot(decades, means, marker='o', color=colour, label=labels[m])
        ax.fill_between(decades, means - stds, means + stds, alpha=0.15, color=colour)
        all_top.append((means + stds).max())
    ax.set_xlabel('Decade', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    annotate_low_confidence(ax, decades, max(all_top))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


plot_weighted_metric(
    ['ttr', 'mattr'],
    {'ttr': 'TTR (log-views weighted)',
     'mattr': 'MATTR window=50 (log-views weighted)'},
    'Vocabulary richness (proportion of unique tokens)',
    'Vocabulary Richness by Decade — Weighted by log(views+1)',
    f'{OUT_DIR}ttr_by_decade_weighted{SUFFIX}.png',
    weighted_stats, decade_order,
)

plot_weighted_metric(
    ['rep_trigram', 'rep_fourgram', 'rep_fivegram'],
    {'rep_trigram': 'Trigram repetition',
     'rep_fourgram': 'Fourgram repetition',
     'rep_fivegram': 'Fivegram repetition'},
    'Repeated n-gram proportion within song',
    'Phrase Repetition by Decade — Weighted by log(views+1)',
    f'{OUT_DIR}repetition_by_decade_weighted{SUFFIX}.png',
    weighted_stats, decade_order,
)

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
unweighted_tfidf = [diversity_df.loc[d, 'thematic_diversity'] for d in decade_order]
weighted_tfidf = [weighted_div_df.loc[d, 'thematic_diversity'] for d in decade_order]
unweighted_embed = [diversity_embed_df.loc[d, 'thematic_diversity_embed'] for d in decade_order]
weighted_embed = [weighted_div_embed_df.loc[d, 'thematic_diversity_embed'] for d in decade_order]

axes[0].plot(decade_order, unweighted_tfidf, marker='o', color='#1f77b4', label='Unweighted')
axes[0].plot(decade_order, weighted_tfidf, marker='s', color='#d62728',
             label='Weighted by log(views+1)')
axes[0].set_xlabel('Decade', fontsize=11)
axes[0].set_ylabel('Mean pairwise cosine distance', fontsize=11)
axes[0].set_title('TF-IDF (vocabulary overlap)', fontsize=12)
axes[0].legend(fontsize=9)
annotate_low_confidence(axes[0], decade_order, max(max(unweighted_tfidf), max(weighted_tfidf)))

axes[1].plot(decade_order, unweighted_embed, marker='o', color='#1f77b4', label='Unweighted')
axes[1].plot(decade_order, weighted_embed, marker='s', color='#d62728',
             label='Weighted by log(views+1)')
axes[1].set_xlabel('Decade', fontsize=11)
axes[1].set_ylabel('Mean pairwise cosine distance', fontsize=11)
axes[1].set_title(f'Sentence embeddings ({SENTENCE_MODEL_NAME})', fontsize=12)
axes[1].legend(fontsize=9)
annotate_low_confidence(axes[1], decade_order, max(max(unweighted_embed), max(weighted_embed)))

for ax in axes:
    ax.tick_params(axis='x', rotation=30)

fig.suptitle('Thematic Diversity by Decade — Unweighted vs Weighted, Two Methods',
             fontsize=13)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}thematic_diversity_by_decade_weighted{SUFFIX}.png', dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}thematic_diversity_by_decade_weighted{SUFFIX}.png")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Homogenisation analysis complete ===")
print(f"Outputs:")
print(f"  {OUT_DIR}decade_distribution{SUFFIX}.png")
print(f"  {OUT_DIR}ttr_by_decade{SUFFIX}.png")
print(f"  {OUT_DIR}repetition_by_decade{SUFFIX}.png")
print(f"  {OUT_DIR}thematic_diversity_by_decade{SUFFIX}.png")
print(f"  {OUT_DIR}homogenisation_decade_stats{SUFFIX}.csv")
print(f"  {OUT_DIR}trend_test_results{SUFFIX}.csv")
print(f"  {OUT_DIR}ttr_by_decade_weighted{SUFFIX}.png")
print(f"  {OUT_DIR}repetition_by_decade_weighted{SUFFIX}.png")
print(f"  {OUT_DIR}thematic_diversity_by_decade_weighted{SUFFIX}.png")
print(f"  {OUT_DIR}homogenisation_decade_stats_weighted{SUFFIX}.csv")
print(f"  {OUT_DIR}trend_test_results_weighted{SUFFIX}.csv")
print(f"  {OUT_DIR}trend_test_results_comparison{SUFFIX}.csv")
print(f"  {SAMPLE_OUT_DIR}sampled_dataset{SUFFIX}.csv")
if low_confidence_decades:
    print(f"Low-confidence decades (<{LOW_CONFIDENCE_THRESHOLD} songs): "
          f"{sorted(low_confidence_decades)}")
