"""
Agreement Analysis: Compare human review against model verdicts.
================================================================
Loads the reviewed results JSON and measures how often the human
assessor agreed that the model's verdict was correct, giving a
reliability signal for the multi-model fact-checking pipeline.

Agreement logic:
    CORRECT           → human agrees with the model verdict
    INCORRECT         → human disagrees
    PARTIALLY_CORRECT → counts as 0.5 agreements in the rate calculation

Run from src/ directory:
    python analyze_agreement.py
    python analyze_agreement.py --input ../data/processed/ai_control_fact_check_results_reviewed.json
    python analyze_agreement.py --show-agreements    # also print matching cases
"""

import json
import argparse
import os
import sys

INPUT_PATH = "../data/processed/ai_control_fact_check_results_reviewed.json"


# ---- Loading ----

def load_reviewed(path):
    """Load the reviewed results JSON. Exits with a helpful message if missing."""
    if not os.path.exists(path):
        sys.exit(
            f"[Error] Reviewed file not found: {path}\n"
            f"        Run interactive_review.py first to generate it."
        )
    with open(path) as f:
        return json.load(f)


# ---- Extraction ----

def extract_reviewed_claims(data):
    """
    Return a flat list of dicts for every claim that has a
    non-skipped human_review entry.

    Each dict contains:
        song_title, artist, year, song_type,
        claim_id, claim_text,
        model_verdict, model_confidence, model_explanation,
        human_assessment, human_notes, human_ground_truth
    """
    rows = []
    for song in data["results"]:
        for claim in song["claims"]:
            review = claim.get("human_review", {})
            if not review.get("reviewed", False):
                continue
            assessment = review.get("human_assessment")
            if assessment in (None, "SKIP"):
                continue
            rows.append({
                "song_title":         song["title"],
                "artist":             song["artist"],
                "year":               song["year"],
                "song_type":          song["song_type"],
                "claim_id":           claim["claim_id"],
                "claim_text":         claim["claim_text"],
                "model_verdict":      claim["verdict"],
                "model_confidence":   claim["confidence"],
                "model_explanation":  claim["explanation"],
                "human_assessment":   assessment,
                "human_notes":        review.get("notes"),
                "human_ground_truth": review.get("ground_truth"),
            })
    return rows


# ---- Agreement calculation ----

def score_agreement(human_assessment):
    """
    Return a numeric agreement score for a single claim:
        CORRECT           → 1.0
        PARTIALLY_CORRECT → 0.5
        INCORRECT         → 0.0
    """
    return {"CORRECT": 1.0, "PARTIALLY_CORRECT": 0.5, "INCORRECT": 0.0}.get(
        human_assessment, 0.0
    )


def compute_agreement_stats(claims):
    """
    Compute overall and per-verdict agreement statistics.

    Returns a dict with:
        overall_rate, total, weighted_correct,
        by_verdict: {verdict: {total, correct, partial, incorrect, rate}}
    """
    if not claims:
        return None

    total            = len(claims)
    weighted_correct = sum(score_agreement(c["human_assessment"]) for c in claims)
    overall_rate     = weighted_correct / total if total else 0.0

    by_verdict = {}
    for c in claims:
        v = c["model_verdict"]
        if v not in by_verdict:
            by_verdict[v] = {"total": 0, "correct": 0, "partial": 0, "incorrect": 0}
        by_verdict[v]["total"] += 1
        a = c["human_assessment"]
        if a == "CORRECT":
            by_verdict[v]["correct"] += 1
        elif a == "PARTIALLY_CORRECT":
            by_verdict[v]["partial"] += 1
        else:
            by_verdict[v]["incorrect"] += 1

    for v, s in by_verdict.items():
        weighted = s["correct"] + 0.5 * s["partial"]
        s["rate"] = weighted / s["total"] if s["total"] else 0.0

    return {
        "overall_rate":     overall_rate,
        "total":            total,
        "weighted_correct": weighted_correct,
        "by_verdict":       by_verdict,
    }


def split_by_agreement(claims):
    """Split claims into disagreements, partial, and agreements."""
    disagreements = [c for c in claims if c["human_assessment"] == "INCORRECT"]
    partials      = [c for c in claims if c["human_assessment"] == "PARTIALLY_CORRECT"]
    agreements    = [c for c in claims if c["human_assessment"] == "CORRECT"]
    return agreements, partials, disagreements


# ---- Printing ----

def divider(char="─", width=72):
    print(char * width)


def print_claim_detail(rank, claim):
    """Print full detail for a single reviewed claim."""
    divider()
    print(f"  #{rank}  Model: {claim['model_verdict'].upper()} ({claim['model_confidence']} conf)  "
          f"→  Human: {claim['human_assessment']}")
    print(f"  Song:  \"{claim['song_title']}\" — {claim['artist']} ({claim['year']}) [{claim['song_type']}]")
    print(f"  Claim {claim['claim_id']}: {claim['claim_text']}")
    print(f"  Model explanation: {claim['model_explanation']}")
    if claim["human_notes"]:
        print(f"  Human notes:       {claim['human_notes']}")
    if claim["human_ground_truth"]:
        print(f"  Correct answer:    {claim['human_ground_truth']}")


def print_stats_table(stats):
    """Print the agreement stats as a formatted table."""
    divider("═")
    print(f"  Agreement Rate (weighted: PARTIAL = 0.5)")
    divider()
    print(f"  {'Verdict':<16}  {'Total':>5}  {'Correct':>7}  {'Partial':>7}  {'Incorrect':>9}  {'Rate':>6}")
    divider()

    order = ["contradicted", "uncertain", "verified"]
    for v in order:
        if v not in stats["by_verdict"]:
            continue
        s = stats["by_verdict"][v]
        print(f"  {v.capitalize():<16}  {s['total']:>5}  {s['correct']:>7}  "
              f"{s['partial']:>7}  {s['incorrect']:>9}  {s['rate']:>5.1%}")

    divider()
    print(f"  {'OVERALL':<16}  {stats['total']:>5}  {stats['weighted_correct']:>7.1f}  "
          f"{'':>7}  {'':>9}  {stats['overall_rate']:>5.1%}")
    divider("═")


def main():
    parser = argparse.ArgumentParser(description="Analyse human–model agreement on fact-check results.")
    parser.add_argument("--input",           default=INPUT_PATH, help="Path to reviewed results JSON")
    parser.add_argument("--show-agreements", action="store_true", help="Also print agreeing claims")
    parser.add_argument("--show-partials",   action="store_true", help="Also print partial-agreement claims")
    args = parser.parse_args()

    data   = load_reviewed(args.input)
    meta   = data["pipeline_metadata"]
    claims = extract_reviewed_claims(data)

    total_claims   = sum(len(s["claims"]) for s in data["results"])
    skipped        = total_claims - len(claims)

    print(f"\n{'=' * 72}")
    print(f"  Agreement Analysis — {meta['year_range']} ({meta['total_songs_processed']} songs)")
    print(f"  Analyzer: {meta['analyzer_model']}  |  Verifier: {meta['verifier_model']}")
    print(f"  Human reviews: {len(claims)}/{total_claims}  ({skipped} skipped/unreviewed)")
    print(f"{'=' * 72}")

    if not claims:
        print("\n  [!] No reviewed claims found. Run interactive_review.py first.\n")
        return

    stats = compute_agreement_stats(claims)
    print_stats_table(stats)

    agreements, partials, disagreements = split_by_agreement(claims)

    # ---- Disagreements (always shown) ----
    print(f"\n  DISAGREEMENTS — human marked model verdict as INCORRECT ({len(disagreements)} claims)")
    if disagreements:
        for i, c in enumerate(disagreements, 1):
            print_claim_detail(i, c)
    else:
        print("  None — the model's verdicts matched human assessment in all reviewed cases.")

    # ---- Partial agreements ----
    if args.show_partials or partials:
        print(f"\n  PARTIAL AGREEMENTS ({len(partials)} claims)")
        if partials:
            for i, c in enumerate(partials, 1):
                print_claim_detail(i, c)
        else:
            print("  None.")

    # ---- Full agreements (opt-in) ----
    if args.show_agreements:
        print(f"\n  AGREEMENTS — human marked model verdict as CORRECT ({len(agreements)} claims)")
        for i, c in enumerate(agreements, 1):
            print_claim_detail(i, c)

    # ---- Insight summary ----
    print(f"\n{'═' * 72}")
    print(f"  Key Findings")
    divider()

    rate = stats["overall_rate"]
    if rate >= 0.85:
        signal = "strong — the verifier model is reliable"
    elif rate >= 0.65:
        signal = "moderate — the verifier has meaningful but imperfect accuracy"
    else:
        signal = "weak — verifier outputs need careful human review"

    print(f"  Overall agreement rate:  {rate:.1%}  ({signal})")

    # Highest and lowest agreement verdicts
    bv = stats["by_verdict"]
    if len(bv) > 1:
        best  = max(bv, key=lambda v: bv[v]["rate"])
        worst = min(bv, key=lambda v: bv[v]["rate"])
        print(f"  Most reliable verdict:   {best.capitalize()} ({bv[best]['rate']:.1%} agreement)")
        print(f"  Least reliable verdict:  {worst.capitalize()} ({bv[worst]['rate']:.1%} agreement)")

    if disagreements:
        contested_songs = {c["song_title"] for c in disagreements}
        print(f"  Songs with disagreements: {', '.join(sorted(contested_songs))}")

    print(f"{'═' * 72}\n")


if __name__ == "__main__":
    main()
