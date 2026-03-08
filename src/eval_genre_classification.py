"""
eval_genre_classification.py

Evaluates genre classification accuracy across multiple Claude models.

Outputs:
  - data/processed/genre_eval_predictions_{timestamp}.csv   (per-song predictions)
  - data/processed/genre_eval_results_{timestamp}.json      (metrics summary)
  - reports/confusion_matrix_{model_id}_{timestamp}.png     (one per model)

Usage: Run from src/ directory
  python eval_genre_classification.py
"""

import pandas as pd
import numpy as np
import os
import json
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import anthropic

# ──────────────────────────────────────────────
# CONFIG — adjust these before each run
# ──────────────────────────────────────────────
SONGS_PER_GENRE = 50          # equal sample size per genre
RANDOM_SEED     = 42
GENRES          = ['country', 'pop', 'r&b', 'rap', 'rock']

CLAUDE_MODELS = [
    {"model_id": "claude-haiku-4-5-20251001", "display_name": "Claude Haiku 4.5"},
    {"model_id": "claude-sonnet-4-6",          "display_name": "Claude Sonnet 4.6"},
    {"model_id": "claude-opus-4-6",            "display_name": "Claude Opus 4.6"},
]

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
DATA_PATH   = '../data/processed/song_lyrics_clean_df.csv'
OUTPUT_DIR  = '../data/processed'
REPORTS_DIR = '../reports'

# ──────────────────────────────────────────────
# LABEL NORMALIZATION
# ──────────────────────────────────────────────
LABEL_MAP = {
    'R&B': 'r&b', 'rb': 'r&b',
    'Rock': 'rock', 'indie rock': 'rock', 'Indie Rock': 'rock',
    'Pop': 'pop', 'musical': 'pop', 'dancehall': 'pop',
    'gospel': 'pop', 'Christian': 'pop', 'reggae': 'pop', 'folk': 'pop',
    'Rap': 'rap', 'hip-hop': 'rap', 'hip hop': 'rap', 'Hip-Hop': 'rap',
    'Country': 'country',
}

def normalize_genre(label):
    if not isinstance(label, str):
        return 'unknown'
    label = label.strip()
    return LABEL_MAP.get(label, label.lower())

# ──────────────────────────────────────────────
# CLAUDE PREDICTION
# ──────────────────────────────────────────────
anthropic_client = anthropic.Anthropic()

def get_genre_from_lyrics_claude(lyrics, model_id):
    options_str = ", ".join(GENRES)
    prompt = f"""
<role>You are an expert music critic.</role>
<task>Based on these lyrics, classify the song into one of these genres: {options_str}</task>
<instruction>Respond with only the genre name.</instruction>
<lyrics>
\"\"\"
{lyrics}
\"\"\"
</lyrics>
"""
    try:
        response = anthropic_client.messages.create(
            model=model_id,
            max_tokens=50,
            system="You are an expert music critic",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"  [Error] {model_id}: {e}")
        return None

# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────
def evaluate_model(df, pred_col, display_name, model_id, timestamp):
    """Compute metrics, save confusion matrix PNG, return metrics dict."""
    valid = df[df[pred_col].isin(GENRES)].copy()
    n_invalid = len(df) - len(valid)
    if n_invalid:
        print(f"  [{display_name}] {n_invalid} predictions outside known genres — excluded")

    y_true = valid['genre']
    y_pred = valid[pred_col]

    accuracy = accuracy_score(y_true, y_pred)
    report   = classification_report(y_true, y_pred, labels=GENRES, zero_division=0, output_dict=True)
    cm       = confusion_matrix(y_true, y_pred, labels=GENRES)

    print(f"  Accuracy: {accuracy:.3f}  (n={len(valid)})")

    # Confusion matrix image
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GENRES).plot(ax=ax, colorbar=True)
    ax.set_title(f"Confusion Matrix — {display_name}")
    plt.tight_layout()
    safe_id  = model_id.replace('/', '_')
    img_name = f"confusion_matrix_{safe_id}_{timestamp}.png"
    img_path = os.path.join(REPORTS_DIR, img_name)
    fig.savefig(img_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved: {img_path}")

    return {
        "display_name":           display_name,
        "n_evaluated":            int(len(valid)),
        "n_invalid":              int(n_invalid),
        "accuracy":               round(float(accuracy), 4),
        "classification_report":  report,
        "confusion_matrix":       cm.tolist(),
        "confusion_matrix_image": img_path,
    }

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Load data
print("Loading data...")
df = pd.concat(
    [chunk for chunk in pd.read_csv(DATA_PATH, chunksize=100_000)],
    ignore_index=True,
)
df['genre'] = df['genre'].replace({'rb': 'r&b'})

# Equal sample per genre
sample_df = (
    df[df['genre'].isin(GENRES)]
    .groupby('genre', group_keys=False)
    .apply(lambda x: x.sample(n=min(SONGS_PER_GENRE, len(x)), random_state=RANDOM_SEED))
    .reset_index(drop=True)
)
print(f"Sample: {len(sample_df)} songs ({SONGS_PER_GENRE} per genre)")
print(sample_df['genre'].value_counts().sort_index().to_string())

results = {
    "metadata": {
        "timestamp":       timestamp,
        "songs_per_genre": SONGS_PER_GENRE,
        "total_songs":     int(len(sample_df)),
        "genres":          GENRES,
        "random_seed":     RANDOM_SEED,
    },
    "models": {},
}

# Run each Claude model
for cfg in CLAUDE_MODELS:
    model_id     = cfg["model_id"]
    display_name = cfg["display_name"]
    col          = f"pred_{model_id.replace('-', '_').replace('.', '_')}"

    print(f"\nRunning {display_name} on {len(sample_df)} samples...")
    sample_df[col] = sample_df["clean_lyrics"].apply(
        lambda lyrics, mid=model_id: get_genre_from_lyrics_claude(lyrics, mid)
    )
    sample_df[col] = sample_df[col].map(normalize_genre)
    results["models"][model_id] = evaluate_model(
        sample_df, col, display_name, model_id, timestamp,
    )

# Save predictions CSV
csv_path = os.path.join(OUTPUT_DIR, f"genre_eval_predictions_{timestamp}.csv")
sample_df.to_csv(csv_path, index=False)
print(f"\nPredictions CSV:  {csv_path}")

# Save JSON results
json_path = os.path.join(OUTPUT_DIR, f"genre_eval_results_{timestamp}.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results JSON:     {json_path}")
print("\nDone.")
