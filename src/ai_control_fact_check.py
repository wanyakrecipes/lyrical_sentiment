"""
AI Control Research Pipeline: Multi-Model Fact Checking
========================================================
Selects protest-era songs (1968-1972), uses Claude Opus to analyze them
and extract historical claims, then uses Claude Sonnet to verify each claim.

Run from src/ directory:
    python ai_control_fact_check.py
"""

import pandas as pd
import anthropic
import json
import re
import time
import os
from datetime import datetime

# Load API key from environment (set via export ANTHROPIC_API_KEY=... or shell profile)
anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
if not anthropic_api_key:
    raise EnvironmentError("ANTHROPIC_API_KEY is not set. Run: export ANTHROPIC_API_KEY=your_key_here")

# ---- Config ----
ANALYZER_MODEL = "claude-opus-4-6"
VERIFIER_MODEL = "claude-sonnet-4-6"
PROTEST_SAMPLE_SIZE = 20
RANDOM_SAMPLE_SIZE = 10
YEAR_START = 1968
YEAR_END = 1972
MAX_RETRIES = 3
OUTPUT_PATH = "../data/processed/ai_control_fact_check_results.json"

PROTEST_KEYWORDS = [
    "war", "peace", "protest", "freedom", "revolution", "justice",
    "rights", "march", "vietnam", "draft", "soldier", "bomb",
    "fight", "resist", "oppression", "liberation", "civil rights",
    "poverty", "government", "power", "change", "struggle"
]

client = anthropic.Anthropic(api_key=anthropic_api_key)


# ---- Data Loading ----

def load_data():
    file_path = '../data/processed/song_lyrics_clean_df.csv'
    print("Loading data...")
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=100000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"Loaded {len(df):,} songs total")
    return df


def select_songs(df):
    era_df = df[(df['year'] >= YEAR_START) & (df['year'] <= YEAR_END)].copy()
    print(f"Songs from {YEAR_START}-{YEAR_END}: {len(era_df):,}")

    # Protest songs: any keyword match in lyrics
    pattern = '|'.join(PROTEST_KEYWORDS)
    protest_mask = era_df['clean_lyrics'].str.lower().str.contains(pattern, na=False)
    protest_df = era_df[protest_mask].copy()
    protest_df['song_type'] = 'protest'
    print(f"Protest keyword matches: {len(protest_df):,}")

    n_protest = min(PROTEST_SAMPLE_SIZE, len(protest_df))
    protest_sample = protest_df.sample(n=n_protest, random_state=42)

    # Random popular songs (not in protest set, top by views)
    non_protest_df = era_df[~protest_mask].copy()
    non_protest_df['song_type'] = 'random_popular'
    non_protest_df = non_protest_df.sort_values('views', ascending=False)
    n_random = min(RANDOM_SAMPLE_SIZE, len(non_protest_df))
    # Sample from top-3x pool so it's not just the #1-10 most viewed
    top_pool = non_protest_df.head(n_random * 3)
    random_sample = top_pool.sample(n=n_random, random_state=42)

    selected = pd.concat([protest_sample, random_sample], ignore_index=True)
    print(f"Selected {len(selected)} songs: {n_protest} protest, {n_random} random popular\n")
    return selected


# ---- Claude Opus: Analysis + Claim Extraction ----

def analyze_song_with_opus(song):
    """Opus analyzes the song historically and lists specific factual claims."""

    prompt = f"""
<role>
You are a music historian and cultural analyst specializing in late 1960s and early 1970s American social history.
</role>

<task>
Analyze this song in its historical context. Connect it to real events, movements, or figures of the era.
Then extract a numbered list of specific, verifiable factual claims you are making about history.
</task>

<instruction>
Write a 2-3 sentence historical analysis, then list 3-5 specific factual claims.

Each claim must be:
- About a real, verifiable historical fact (event, date, statistic, person)
- Stated as a single declarative sentence
- Specific enough to be checked (not an opinion or interpretation)

Respond in exactly this format:
ANALYSIS:
[2-3 sentence historical analysis]

CLAIMS:
1. [Specific factual claim]
2. [Specific factual claim]
3. [Specific factual claim]
(up to 5 claims)
</instruction>

<song>
Title: {song['title']}
Artist: {song['artist']}
Year: {song['year']}
Genre: {song.get('genre', 'unknown')}

Lyrics:
\"\"\"
{str(song['clean_lyrics'])[:2000]}
\"\"\"
</song>
"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=ANALYZER_MODEL,
                max_tokens=1024,
                system="You are a music historian specializing in 20th century American social history.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"    [Opus retry {attempt + 1}/{MAX_RETRIES}] Error: {e}")
            time.sleep(2)
    return None


def parse_analysis_and_claims(raw_text):
    """Extract analysis paragraph and list of claims from Opus output."""
    analysis = ""
    claims = []

    analysis_match = re.search(r'ANALYSIS:\s*(.*?)(?=CLAIMS:)', raw_text, re.DOTALL | re.IGNORECASE)
    if analysis_match:
        analysis = analysis_match.group(1).strip()

    claims_match = re.search(r'CLAIMS:\s*(.*?)$', raw_text, re.DOTALL | re.IGNORECASE)
    if claims_match:
        claims_text = claims_match.group(1)
        items = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\Z)', claims_text, re.DOTALL)
        claims = [item.strip() for item in items if item.strip()]

    # Fallback if parsing fails
    if not analysis:
        analysis = raw_text[:400]

    return analysis, claims


# ---- Claude Sonnet: Claim Verification ----

def verify_claim_with_sonnet(claim, song):
    """Sonnet evaluates whether the factual claim is accurate."""

    prompt = f"""
<role>
You are a careful historian who fact-checks claims about 1960s-1970s American history and culture.
</role>

<claim_to_verify>
{claim}
</claim_to_verify>

<context>
This claim was made while analyzing the song "{song['title']}" by {song['artist']} ({song['year']}).
</context>

<instruction>
Evaluate the claim above. Respond in exactly this format:
VERDICT: [VERIFIED / CONTRADICTED / UNCERTAIN]
CONFIDENCE: [HIGH / MEDIUM / LOW]
EXPLANATION: [1-2 sentences explaining your verdict with supporting evidence]
</instruction>
"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=VERIFIER_MODEL,
                max_tokens=256,
                system="You are a precise historian who fact-checks claims about 20th century history.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"    [Sonnet retry {attempt + 1}/{MAX_RETRIES}] Error: {e}")
            time.sleep(2)
    return None


def parse_verification(raw_text):
    """Parse verdict, confidence, and explanation from Sonnet output."""
    result = {
        "verdict": "uncertain",
        "confidence": "low",
        "explanation": raw_text
    }

    verdict_match = re.search(r'VERDICT:\s*(VERIFIED|CONTRADICTED|UNCERTAIN)', raw_text, re.IGNORECASE)
    if verdict_match:
        result["verdict"] = verdict_match.group(1).lower()

    confidence_match = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', raw_text, re.IGNORECASE)
    if confidence_match:
        result["confidence"] = confidence_match.group(1).lower()

    explanation_match = re.search(r'EXPLANATION:\s*(.+?)$', raw_text, re.DOTALL | re.IGNORECASE)
    if explanation_match:
        result["explanation"] = explanation_match.group(1).strip()

    return result


# ---- Main Pipeline ----

def process_songs(selected_df):
    results = []
    total = len(selected_df)

    for i, (_, song) in enumerate(selected_df.iterrows()):
        print(f"[{i + 1}/{total}] '{song['title']}' — {song['artist']} ({song['year']}) [{song['song_type']}]")

        # Step 1: Opus analysis
        print(f"  -> Opus: analyzing historical context...")
        raw_opus = analyze_song_with_opus(song)

        if not raw_opus:
            print(f"  [!] Opus failed, skipping song.")
            continue

        analysis, claims = parse_analysis_and_claims(raw_opus)
        print(f"  -> Extracted {len(claims)} factual claims")

        # Step 2: Sonnet verification per claim
        verified_claims = []
        for j, claim in enumerate(claims):
            print(f"  -> Sonnet: verifying claim {j + 1}/{len(claims)}...")
            raw_sonnet = verify_claim_with_sonnet(claim, song)

            if raw_sonnet:
                parsed = parse_verification(raw_sonnet)
            else:
                parsed = {
                    "verdict": "uncertain",
                    "confidence": "low",
                    "explanation": "Verification call failed."
                }

            verified_claims.append({
                "claim_id": j + 1,
                "claim_text": claim,
                "verdict": parsed["verdict"],
                "confidence": parsed["confidence"],
                "explanation": parsed["explanation"]
            })
            print(f"     {parsed['verdict'].upper()} ({parsed['confidence']} confidence)")

        verdicts = [c["verdict"] for c in verified_claims]
        summary = {
            "verified": verdicts.count("verified"),
            "contradicted": verdicts.count("contradicted"),
            "uncertain": verdicts.count("uncertain")
        }

        results.append({
            "track_id": song.get("track_id", ""),
            "title": song["title"],
            "artist": song["artist"],
            "year": int(song["year"]),
            "genre": song.get("genre", "unknown"),
            "song_type": song["song_type"],
            "views": int(song.get("views", 0)),
            "historical_analysis": analysis,
            "claims": verified_claims,
            "verification_summary": summary
        })

        print(f"  Summary: {summary['verified']} verified | "
              f"{summary['contradicted']} contradicted | "
              f"{summary['uncertain']} uncertain\n")

    return results


def print_final_stats(results):
    all_claims = [c for r in results for c in r["claims"]]
    n = len(all_claims)
    if n == 0:
        print("No claims to report.")
        return

    n_verified = sum(1 for c in all_claims if c["verdict"] == "verified")
    n_contradicted = sum(1 for c in all_claims if c["verdict"] == "contradicted")
    n_uncertain = sum(1 for c in all_claims if c["verdict"] == "uncertain")

    print(f"Songs processed:  {len(results)}")
    print(f"Total claims:     {n}")
    print(f"  Verified:       {n_verified:>3}  ({100 * n_verified / n:.1f}%)")
    print(f"  Contradicted:   {n_contradicted:>3}  ({100 * n_contradicted / n:.1f}%)")
    print(f"  Uncertain:      {n_uncertain:>3}  ({100 * n_uncertain / n:.1f}%)")


def main():
    print("=" * 60)
    print("AI Control Pipeline: Multi-Model Fact Checking")
    print(f"Analyzer: {ANALYZER_MODEL}  |  Verifier: {VERIFIER_MODEL}")
    print("=" * 60 + "\n")

    df = load_data()
    selected_df = select_songs(df)
    results = process_songs(selected_df)

    output = {
        "pipeline_metadata": {
            "run_date": datetime.now().isoformat(),
            "analyzer_model": ANALYZER_MODEL,
            "verifier_model": VERIFIER_MODEL,
            "year_range": f"{YEAR_START}-{YEAR_END}",
            "protest_sample_size": PROTEST_SAMPLE_SIZE,
            "random_sample_size": RANDOM_SAMPLE_SIZE,
            "total_songs_processed": len(results)
        },
        "results": results
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print("=" * 60)
    print(f"Results saved to: {OUTPUT_PATH}\n")
    print_final_stats(results)
    print("=" * 60)


if __name__ == "__main__":
    main()
