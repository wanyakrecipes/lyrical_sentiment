# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project analyzing sentiment trends in song lyrics over time (1950-2022) using multiple ML approaches. Compares GPT-4o, BERT (Twitter RoBERTa), and Claude models to understand if music lyrics have become more negative and to evaluate LLM capabilities in genre classification.

**Dataset:** Kaggle Genius Song Lyrics (~1M+ songs)

## Commands

### Setup
```bash
pip install -r requirements.txt
```

Set API keys as environment variables:
```bash
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here
```

Or add them to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) for persistence.

### Data Processing
```bash
# Clean raw data (creates data/processed/song_lyrics_clean_df.csv)
cd src
python clean_data.py
```

### Running Analysis Scripts
All scripts must be run from `src/` directory (they use relative paths `../data/`):

```bash
cd src

# Sentiment over time analyses
python sentiment_per_year_gpt_4o.py              # GPT-4o-mini full lyrics
python sentiment_per_year_trbs_model.py          # BERT full lyrics
python chorus_sentiment_per_year_gpt_4o.py       # GPT-4o phrase extraction
python sentiment_per_year_trbs_model_chorus_only.py  # BERT chorus only

# Evaluation scripts
python eval_genre_bias.py                        # Genre classification accuracy
python self_labelled_gpt-4o.py                   # GPT on labeled data
python self_labelled_trbs_model.py               # BERT on labeled data
```

### Notebooks
```bash
# Open Jupyter notebooks from root directory
jupyter notebook notebooks/exploratory_data_analysis.ipynb
jupyter notebook notebooks/sentiment_across_time.ipynb
```

## Architecture

### Data Flow Pipeline
```
Raw CSV → Clean (clean_data.py) → Sample by Year → Apply Model → Aggregate by Year → Visualize
```

### Three Model Approaches

**1. GPT Models** (`gpt_4o_prompts.py`)
- Functions use structured XML prompts with retry logic (max 3 attempts)
- Sentiment scale: -1 (negative) to 1 (positive)
- Temperature: 0.0 for scoring, 0.3 for phrase extraction
- Models: `gpt-4o-mini` (primary), `gpt-4o`, `gpt-5-nano`

**2. BERT Twitter RoBERTa** (`bert_model_sentiment.py`)
- Model: `cardiffnlp/twitter-roberta-base-sentiment`
- Chunks lyrics into 512-token segments (handles long text)
- Auto-detects Apple MPS GPU (`mps`) or falls back to CPU
- Returns probabilities: negative, neutral, positive

**3. Claude** (evaluation scripts)
- Used for genre classification and bias research
- Outperforms GPT-4o in multi-choice tasks

### Key Data Structures

**Cleaned DataFrame Schema:**
```python
{
    'track_id': str,          # UUID
    'artist': str,
    'title': str,
    'year': int,              # 1950-2022
    'genre': str,             # rock, rap, pop, r&b, country
    'views': int,             # popularity metric
    'lyrics': str,            # original with [Verse], [Chorus] markup
    'clean_lyrics': str,      # cleaned, no newlines
    'clean_lyrics_compact': str  # alternative version
}
```

**Analysis Results Columns:**
```python
# BERT outputs
'sentiment_trbs': dict
'sentiment_trbs_label': str        # negative/neutral/positive
'sentiment_trbs_positive': float   # probability
'sentiment_trbs_negative': float
'sentiment_trbs_neutral': float

# GPT outputs
'sentiment_score_gpt': float       # -1 to 1
'common_phrases_gpt': str          # XML format
'common_phrases_gpt_list': list
'common_phrases_sentiment_gpt': list[tuple]
'common_phrases_average_sentiment_gpt': float
```

### Sampling Strategy

All analysis scripts follow this pattern:
```python
# Stratified by year to prevent recent year bias
sample_df = df.groupby('year').apply(
    lambda x: x.sample(n=100, random_state=42) if len(x) > 100 else x
)
sample_df = sample_df.reset_index(drop=True)
```
- Random seed: 42 (reproducibility)
- Sample size: 30-100 songs per year
- Final sample: ~2,000-7,000 songs depending on analysis

### Data Cleaning Steps

The `clean_data.py` pipeline applies transformations in this order:
1. Load in 100k row chunks (memory management)
2. Filter: English language only
3. Filter: Remove "Genius" artist entries
4. Filter: Years 1950-2022
5. Filter: Exclude "misc" genre
6. Filter: Top 95th percentile by views (quality signal)
7. Text cleaning:
   - Remove section labels `[Verse 1]`, `[Chorus]`
   - Remove newlines, tabs, special chars, digits
   - Normalize whitespace, lowercase
8. Add `track_id` UUID and rename "tag" → "genre"

### Prompt Engineering Pattern

GPT prompts use this XML structure:
```python
prompt = f"""
<role>You are an expert music critic.</role>
<task>Your specific task here</task>
<instruction>Response format instructions</instruction>
<lyrics>
\"\"\"
{lyrics}
\"\"\"
</lyrics>
"""
```

### Device Optimization

BERT scripts auto-detect hardware:
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```
Apple M1/M2 Macs use MPS GPU acceleration.

## File Organization

```
src/
├── gpt_4o_prompts.py              # All GPT API calls and prompts
├── bert_model_sentiment.py        # BERT model utilities
├── clean_data.py                  # Main data cleaning pipeline
├── get_api_keys.py                # Example of loading API keys from environment
├── sentiment_per_year_*.py        # Time series analyses
├── chorus_sentiment_*.py          # Phrase-level analyses
├── eval_*.py                      # Model evaluation scripts
└── self_labelled_*.py             # Labeled data validation

data/
├── raw/                           # song_lyrics.csv (not in git)
└── processed/                     # song_lyrics_clean_df.csv (not in git)

reports/
├── *.png                          # Generated visualizations
└── summary.md                     # Research findings summary

notebooks/
├── exploratory_data_analysis.ipynb
└── sentiment_across_time.ipynb
```

## Important Conventions

### API Key Management
All scripts load API keys directly from environment variables:
```python
import os
import openai
import anthropic

# Clients automatically use environment variables
openai_client = openai.Client()  # Uses OPENAI_API_KEY
anthropic_client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY

# Or access explicitly
api_key = os.environ.get('OPENAI_API_KEY')
```

No `.env` files or `python-dotenv` required - API keys must be set in your shell environment.

### Path Assumptions
- All Python scripts assume execution from `src/` directory
- Use relative paths: `../data/raw/`, `../data/processed/`
- Large CSVs read in chunks (100k rows)

### API Error Handling
```python
# Standard retry pattern
for attempt in range(max_retries):
    try:
        response = openai.chat.completions.create(...)
        return parse_result(response)
    except Exception as e:
        print(f"[Retry {attempt+1}] Error: {e}")
        time.sleep(2)
return None
```

### Model Temperature Settings
- `temperature=0.0`: Deterministic sentiment scoring
- `temperature=0.3`: Creative phrase extraction
- Never use `temperature=1.0+` (defeats reproducibility)

## Research Findings

From `reports/summary.md`:

**Sentiment Over Time:**
- Lyrics have become progressively more negative (1950s → 2020s)
- Both GPT and BERT models agree on this trend
- Potential availability bias: older songs may over-represent uplifting tracks

**Genre Classification:**
- GPT-4o accuracy: ~70% across 5 genres
- Claude Sonnet 4 outperforms GPT-4o
- Pop and R&B frequently confused (dataset labels may be too high-level)

**Common Phrases:**
- Repeated lines also show decreasing sentiment over time
- Negative themes becoming more prominent in choruses/hooks

## Dependencies Note

From `requirements.txt`:
- PyTorch (no version pinned - may cause compatibility issues)
- `transformers==4.41.2` - HuggingFace models
- `openai==1.52.1` - OpenAI API client
- `anthropic` - Anthropic API client (Claude)
- `pandas==2.2.2` - Data manipulation
- `numpy`, `matplotlib`, `seaborn` - Numerical computing and visualization
- `plotly==5.22.0` - Interactive visualizations
- `scikit-learn==1.5.1` - ML metrics and evaluation
- `sentence-transformers` - Semantic similarity (used in notebooks)
- `tqdm` - Progress bars (used in notebooks)
- `nbformat==5.9.2`, `ipykernel` - Jupyter notebook support
