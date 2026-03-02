"""
Interactive Review: Human-in-the-loop fact-check assessment.
=============================================================
Presents fact-check results one claim at a time and records your
verdict (CORRECT / INCORRECT / PARTIALLY_CORRECT / SKIP).

Progress is saved after every decision, so it is safe to quit
(Ctrl+C) and resume later — already-reviewed claims are skipped.

Run from src/ directory:
    python interactive_review.py
    python interactive_review.py --redo              # re-review already-reviewed claims
    python interactive_review.py --limit 10          # stop after 10 reviews
    python interactive_review.py --sample 20         # balanced sample across all three verdicts
    python interactive_review.py --only-contradicted
"""

import json
import argparse
import os
import sys
import copy
from datetime import datetime

INPUT_PATH  = "../data/processed/ai_control_fact_check_results.json"
OUTPUT_PATH = "../data/processed/ai_control_fact_check_results_reviewed.json"

VERDICT_PRIORITY    = {"contradicted": 0, "uncertain": 1, "verified": 2}
CONFIDENCE_PRIORITY = {"low": 0, "medium": 1, "high": 2}
VALID_ASSESSMENTS   = {"1": "CORRECT", "2": "INCORRECT",
                        "3": "PARTIALLY_CORRECT", "s": "SKIP",
                        "q": "QUIT"}


# ---- File I/O ----

def load_input(path):
    """Load the original or previously-reviewed results JSON."""
    if not os.path.exists(path):
        sys.exit(f"[Error] File not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_reviewed(output_path, input_path):
    """
    Load existing reviewed file if present, otherwise start from
    a deep copy of the input so the original is never modified.
    """
    if os.path.exists(output_path):
        print(f"[Info] Resuming from existing review file: {output_path}")
        with open(output_path) as f:
            return json.load(f)
    print(f"[Info] Starting fresh — output will be saved to: {output_path}")
    return copy.deepcopy(load_input(input_path))


def save_reviewed(data, path):
    """Atomically save the reviewed data by writing then renaming."""
    tmp_path = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"\n[Error] Could not save results: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---- Claim indexing ----

def build_claim_index(data):
    """
    Return a flat list of (song_idx, claim_idx) tuples sorted by review
    priority: contradicted > uncertain > verified, then low confidence first.
    """
    items = []
    for s_idx, song in enumerate(data["results"]):
        for c_idx, claim in enumerate(song["claims"]):
            items.append((s_idx, c_idx, claim["verdict"], claim["confidence"]))

    items.sort(key=lambda x: (
        VERDICT_PRIORITY[x[2]],
        CONFIDENCE_PRIORITY[x[3]],
    ))
    return [(s, c) for s, c, _, _ in items]


def sample_index_by_verdict(data, index, n):
    """
    Reduce a claim index to a balanced sample of n entries drawn evenly
    from contradicted, uncertain, and verified groups.

    Within each group claims are already sorted by confidence (low first)
    because build_claim_index sorts by (verdict_priority, confidence_priority).
    If a group is smaller than its allocation the surplus rolls to the next
    group in priority order (contradicted > uncertain > verified).

    Returns the sampled index and an alloc dict showing counts per verdict.
    """
    groups = {"contradicted": [], "uncertain": [], "verified": []}
    for s_idx, c_idx in index:
        verdict = data["results"][s_idx]["claims"][c_idx]["verdict"]
        groups[verdict].append((s_idx, c_idx))

    per_group = n // 3
    remainder = n % 3
    alloc = {
        "contradicted": per_group + (1 if remainder > 0 else 0),
        "uncertain":    per_group + (1 if remainder > 1 else 0),
        "verified":     per_group,
    }

    # Cap at group size and redistribute surplus in priority order
    surplus = 0
    for v in ("contradicted", "uncertain", "verified"):
        available = len(groups[v])
        if alloc[v] > available:
            surplus += alloc[v] - available
            alloc[v] = available

    for v in ("contradicted", "uncertain", "verified"):
        if surplus == 0:
            break
        capacity = len(groups[v]) - alloc[v]
        if capacity > 0:
            give = min(surplus, capacity)
            alloc[v] += give
            surplus -= give

    sampled = []
    for v in ("contradicted", "uncertain", "verified"):
        sampled.extend(groups[v][: alloc[v]])
    return sampled, alloc


def count_reviewed(data):
    """Count how many claims already have a human_review entry."""
    return sum(
        1 for song in data["results"]
        for claim in song["claims"]
        if claim.get("human_review", {}).get("reviewed", False)
    )


# ---- Display ----

def clear_line():
    print()


def print_divider(char="─", width=72):
    print(char * width)


def display_claim(rank, total, song, claim):
    """Print all available information about a claim."""
    verdict    = claim["verdict"].upper()
    confidence = claim["confidence"].upper()

    verdict_display = {
        "CONTRADICTED": f"\033[91mCONTRADICTED\033[0m",  # red
        "UNCERTAIN":    f"\033[93mUNCERTAIN\033[0m",      # yellow
        "VERIFIED":     f"\033[92mVERIFIED\033[0m",       # green
    }.get(verdict, verdict)

    print_divider("═")
    print(f"  Claim {rank} of {total}")
    print_divider()
    print(f"  Song:       \"{song['title']}\" — {song['artist']} ({song['year']}) [{song['song_type']}]")
    print_divider()
    print(f"  CLAIM {claim['claim_id']}:")
    print(f"  {claim['claim_text']}")
    print_divider()
    print(f"  MODEL VERDICT:    {verdict_display}  ({confidence} confidence)")
    print(f"  MODEL REASONING:")
    # Word-wrap the explanation at ~68 chars
    words = claim["explanation"].split()
    line, lines = [], []
    for word in words:
        if sum(len(w) + 1 for w in line) + len(word) > 68:
            lines.append("  " + " ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append("  " + " ".join(line))
    print("\n".join(lines))
    print_divider()


def prompt_assessment():
    """
    Prompt the user for their assessment.
    Returns (assessment_str, notes, ground_truth) or ("QUIT", None, None).
    """
    print("  Your assessment:")
    print("    [1] CORRECT           — model verdict is right")
    print("    [2] INCORRECT         — model verdict is wrong")
    print("    [3] PARTIALLY_CORRECT — model is partly right")
    print("    [s] SKIP              — not enough info to judge")
    print("    [q] QUIT              — save and exit")
    print()

    while True:
        raw = input("  Choice (1/2/3/s/q): ").strip().lower()
        if raw in VALID_ASSESSMENTS:
            assessment = VALID_ASSESSMENTS[raw]
            break
        print(f"  [!] Invalid input. Choose from: 1, 2, 3, s, q")

    if assessment == "QUIT":
        return "QUIT", None, None

    if assessment == "SKIP":
        return "SKIP", None, None

    notes = input("  Notes (optional, press Enter to skip): ").strip() or None
    ground_truth = input("  Ground truth correction (optional): ").strip() or None
    return assessment, notes, ground_truth


# ---- Review loop ----

def run_review(data, index, args):
    """
    Walk through the prioritised claim index and collect human assessments.
    Returns the number of new reviews completed this session.
    """
    session_count = 0
    total_in_scope = 0

    # Pre-filter index according to flags
    filtered_index = []
    for s_idx, c_idx in index:
        claim = data["results"][s_idx]["claims"][c_idx]
        already = claim.get("human_review", {}).get("reviewed", False)

        if already and not args.redo:
            continue
        verdict = claim["verdict"]
        if args.only_contradicted and verdict != "contradicted":
            continue
        filtered_index.append((s_idx, c_idx))

    total_in_scope = len(filtered_index)

    if total_in_scope == 0:
        print("\n[Info] No claims to review. All done, or try --redo to re-review.")
        return 0

    # Apply balanced sample if requested (takes priority over --limit)
    if args.sample:
        filtered_index, alloc = sample_index_by_verdict(data, filtered_index, args.sample)
        print(f"\n[Info] Balanced sample of {len(filtered_index)} claims "
              f"({alloc['contradicted']} contradicted, "
              f"{alloc['uncertain']} uncertain, "
              f"{alloc['verified']} verified).")
        queue = filtered_index
    else:
        limit = args.limit if args.limit else total_in_scope
        queue = filtered_index[:limit]

    print(f"[Info] {len(queue)} claim(s) queued. Press Ctrl+C or enter 'q' to quit early.\n")

    try:
        for queue_pos, (s_idx, c_idx) in enumerate(queue, start=1):
            song  = data["results"][s_idx]
            claim = song["claims"][c_idx]

            clear_line()
            display_claim(queue_pos, len(queue), song, claim)

            assessment, notes, ground_truth = prompt_assessment()

            if assessment == "QUIT":
                print("\n[Info] Quitting — progress saved.")
                break

            claim["human_review"] = {
                "human_assessment": assessment,
                "notes":            notes,
                "ground_truth":     ground_truth,
                "reviewed":         assessment != "SKIP",
                "reviewed_at":      datetime.now().isoformat(),
            }

            save_reviewed(data, args.output)
            session_count += 1

            if assessment != "SKIP":
                status_word = {"CORRECT": "✓", "INCORRECT": "✗", "PARTIALLY_CORRECT": "~"}.get(assessment, "?")
                print(f"\n  [{status_word}] Saved: {assessment}")
            else:
                print(f"\n  [–] Skipped.")

    except KeyboardInterrupt:
        print("\n\n[Info] Interrupted — progress saved.")

    return session_count


# ---- Summary ----

def print_summary(data, session_count):
    """Print a session summary and overall review progress."""
    all_claims = [c for song in data["results"] for c in song["claims"]]
    reviewed   = [c for c in all_claims if c.get("human_review", {}).get("reviewed", False)]

    counts = {}
    for c in reviewed:
        a = c["human_review"]["human_assessment"]
        counts[a] = counts.get(a, 0) + 1

    print(f"\n{'=' * 72}")
    print(f"  Session complete. {session_count} new review(s) recorded.")
    print(f"  Overall progress: {len(reviewed)}/{len(all_claims)} claims reviewed")
    print()
    for label, count in sorted(counts.items()):
        bar = "█" * count
        print(f"    {label:<20} {count:>3}  {bar}")
    print(f"{'=' * 72}")
    print(f"  Reviewed results saved to: {OUTPUT_PATH}")
    print(f"  To analyse agreement:      python analyze_agreement.py")
    print(f"{'=' * 72}\n")


# ---- Entry point ----

def main():
    parser = argparse.ArgumentParser(description="Interactive human review of fact-check results.")
    parser.add_argument("--input",             default=INPUT_PATH,  help="Source JSON path")
    parser.add_argument("--output",            default=OUTPUT_PATH, help="Output JSON path")
    parser.add_argument("--redo",              action="store_true", help="Re-review already-reviewed claims")
    parser.add_argument("--limit",             type=int,            help="Max claims to review this session")
    parser.add_argument("--sample",            type=int, metavar="N",
                        help="Review a balanced sample of N claims split evenly across all three verdicts")
    parser.add_argument("--only-contradicted", action="store_true", help="Only review CONTRADICTED claims")
    args = parser.parse_args()

    data  = load_reviewed(args.output, args.input)
    index = build_claim_index(data)

    already_done = count_reviewed(data)
    total        = sum(len(s["claims"]) for s in data["results"])
    meta         = data["pipeline_metadata"]

    print(f"\n{'=' * 72}")
    print(f"  Interactive Review — {meta['year_range']} ({meta['total_songs_processed']} songs)")
    print(f"  {total} total claims | {already_done} already reviewed")
    print(f"{'=' * 72}")

    session_count = run_review(data, index, args)
    print_summary(data, session_count)


if __name__ == "__main__":
    main()
