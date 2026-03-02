"""
Priority Review: Rank fact-check results for human review.
===========================================================
Loads ai_control_fact_check_results.json, surfaces the most
review-worthy claims first (CONTRADICTED > UNCERTAIN > VERIFIED),
and prints the top 10-12 for focused human inspection.

Run from src/ directory:
    python priority_review.py
    python priority_review.py --top 15
    python priority_review.py --sample 20      # balanced sample across all three verdicts
    python priority_review.py --input ../data/processed/ai_control_fact_check_results_reviewed.json
"""

import json
import argparse
import os
import sys

INPUT_PATH = "../data/processed/ai_control_fact_check_results.json"

VERDICT_PRIORITY = {"contradicted": 0, "uncertain": 1, "verified": 2}
CONFIDENCE_PRIORITY = {"low": 0, "medium": 1, "high": 2}


def load_results(path):
    """Load the fact-check results JSON file."""
    if not os.path.exists(path):
        sys.exit(f"[Error] File not found: {path}")
    with open(path) as f:
        return json.load(f)


def flatten_claims(data):
    """
    Flatten all claims from all songs into a single list,
    attaching song-level metadata to each claim.

    Returns a list of dicts with keys:
        song_title, artist, year, song_type, claim_id,
        claim_text, verdict, confidence, explanation,
        already_reviewed
    """
    rows = []
    for song in data["results"]:
        for claim in song["claims"]:
            review = claim.get("human_review", {})
            rows.append({
                "song_title":       song["title"],
                "artist":           song["artist"],
                "year":             song["year"],
                "song_type":        song["song_type"],
                "claim_id":         claim["claim_id"],
                "claim_text":       claim["claim_text"],
                "verdict":          claim["verdict"],
                "confidence":       claim["confidence"],
                "explanation":      claim["explanation"],
                "already_reviewed": review.get("reviewed", False),
                "human_assessment": review.get("human_assessment", None),
            })
    return rows


def prioritize(claims, reviewed_only=False):
    """
    Sort claims by:
      1. Verdict priority  (contradicted > uncertain > verified)
      2. Confidence        (low confidence first — more likely to be worth checking)
      3. Already reviewed  (unreviewed first)

    Optionally filter to only already-reviewed claims.
    """
    if reviewed_only:
        claims = [c for c in claims if c["already_reviewed"]]
    return sorted(
        claims,
        key=lambda c: (
            VERDICT_PRIORITY[c["verdict"]],
            CONFIDENCE_PRIORITY[c["confidence"]],
            c["already_reviewed"],
        )
    )


def sample_by_verdict(claims, n):
    """
    Return a balanced sample of n claims drawn evenly from the three
    verdict groups (contradicted, uncertain, verified).

    Each group is sorted by confidence (low first — most review-worthy).
    If a group has fewer claims than its allocation, the remainder is
    distributed to the other groups in priority order.
    """
    groups = {v: [] for v in ("contradicted", "uncertain", "verified")}
    for c in claims:
        groups[c["verdict"]].append(c)

    # Sort each group: low confidence first
    for v in groups:
        groups[v].sort(key=lambda c: CONFIDENCE_PRIORITY[c["confidence"]])

    # Distribute n as evenly as possible; priority order fills any remainder
    per_group = n // 3
    remainder = n % 3
    alloc = {
        "contradicted": per_group + (1 if remainder > 0 else 0),
        "uncertain":    per_group + (1 if remainder > 1 else 0),
        "verified":     per_group,
    }

    # Cap allocations at group size and redistribute any surplus
    surplus = 0
    for v in ("contradicted", "uncertain", "verified"):
        available = len(groups[v])
        if alloc[v] > available:
            surplus += alloc[v] - available
            alloc[v] = available

    # Give surplus to whichever groups still have capacity (priority order)
    for v in ("contradicted", "uncertain", "verified"):
        if surplus == 0:
            break
        capacity = len(groups[v]) - alloc[v]
        if capacity > 0:
            give = min(surplus, capacity)
            alloc[v] += give
            surplus -= give

    # Build final sample, grouped by verdict for easy reading
    result = []
    for v in ("contradicted", "uncertain", "verified"):
        result.extend(groups[v][: alloc[v]])
    return result, alloc


def print_claim(rank, claim, show_explanation=True):
    """Pretty-print a single ranked claim."""
    verdict = claim["verdict"].upper()
    confidence = claim["confidence"].upper()
    reviewed_tag = f" [REVIEWED: {claim['human_assessment']}]" if claim["already_reviewed"] else ""

    verdict_label = {
        "CONTRADICTED": "CONTRADICTED",
        "UNCERTAIN":    "UNCERTAIN   ",
        "VERIFIED":     "VERIFIED    ",
    }.get(verdict, verdict)

    print(f"\n{'─' * 70}")
    print(f"#{rank:>2}  {verdict_label}  ({confidence} confidence){reviewed_tag}")
    print(f"     Song:   \"{claim['song_title']}\" — {claim['artist']} ({claim['year']}) [{claim['song_type']}]")
    print(f"     Claim {claim['claim_id']}: {claim['claim_text']}")
    if show_explanation:
        print(f"     Sonnet: {claim['explanation']}")


def main():
    parser = argparse.ArgumentParser(description="Print prioritised claims for human review.")
    parser.add_argument("--input", default=INPUT_PATH, help="Path to results JSON")
    parser.add_argument("--top", type=int, default=12, help="Number of claims to show (default: 12)")
    parser.add_argument("--sample", type=int, metavar="N",
                        help="Show a balanced sample of N claims split evenly across all three verdicts")
    parser.add_argument("--no-explanation", action="store_true", help="Hide Sonnet explanation")
    parser.add_argument("--all", action="store_true", help="Show all claims, not just top N")
    args = parser.parse_args()

    data = load_results(args.input)
    meta = data["pipeline_metadata"]

    all_claims = flatten_claims(data)
    ranked = prioritize(all_claims)

    total = len(ranked)
    n_contradicted = sum(1 for c in ranked if c["verdict"] == "contradicted")
    n_uncertain    = sum(1 for c in ranked if c["verdict"] == "uncertain")
    n_verified     = sum(1 for c in ranked if c["verdict"] == "verified")
    n_reviewed     = sum(1 for c in ranked if c["already_reviewed"])

    print(f"\n{'=' * 70}")
    print(f"  Priority Review — {meta['year_range']} ({meta['total_songs_processed']} songs)")
    print(f"  Analyzer: {meta['analyzer_model']}  |  Verifier: {meta['verifier_model']}")
    print(f"  Total claims: {total}  "
          f"({n_contradicted} contradicted, {n_uncertain} uncertain, {n_verified} verified)")
    if n_reviewed:
        print(f"  Already reviewed: {n_reviewed}/{total}")
    print(f"{'=' * 70}")

    if args.sample:
        subset, alloc = sample_by_verdict(all_claims, args.sample)
        print(f"\nBalanced sample of {len(subset)} claims "
              f"({alloc['contradicted']} contradicted, "
              f"{alloc['uncertain']} uncertain, "
              f"{alloc['verified']} verified):\n")
        current_verdict = None
        rank = 1
        for claim in subset:
            if claim["verdict"] != current_verdict:
                current_verdict = claim["verdict"]
                print(f"\n  {'─' * 68}")
                print(f"  {current_verdict.upper()} ({alloc[current_verdict]} shown)")
                print(f"  {'─' * 68}")
            print_claim(rank, claim, show_explanation=not args.no_explanation)
            rank += 1
    else:
        limit = total if args.all else args.top
        subset = ranked[:limit]
        print(f"\nShowing top {len(subset)} claims by review priority:\n")

        for i, claim in enumerate(subset, start=1):
            print_claim(i, claim, show_explanation=not args.no_explanation)

        if not args.all and total > args.top:
            print(f"\n{'─' * 70}")
            print(f"  ... {total - args.top} more claims not shown. Use --all to see everything.")

    print(f"\n{'=' * 70}")
    print(f"  To review interactively:  python interactive_review.py")
    print(f"  To analyse agreement:     python analyze_agreement.py")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
