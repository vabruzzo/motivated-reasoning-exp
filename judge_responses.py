"""
Judge Responses
===============
Judge generated responses using one or two LLM judges. Aggregates scores when using multiple judges.

Usage:
    # Single judge (default: Gemini 2.5 Pro)
    python judge_responses.py --input outputs/responses_20250101_120000.csv

    # Two judges for cross-validation
    python judge_responses.py --input outputs/responses_20250101_120000.csv --second-judge anthropic/claude-3.5-sonnet

Output:
    outputs/judged_YYYYMMDD_HHMMSS.csv
    outputs/statistics_YYYYMMDD_HHMMSS.json
"""

import argparse
import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from lib.judge import judge_task, DEFAULT_JUDGE_MODEL
from lib.stats import statistical_comparison, print_statistical_summary
from lib.utils import log

DEFAULT_CONCURRENCY = 10  # max parallel API calls
DEFAULT_BATCH_DELAY = 2.0  # seconds between batches


# ============================================================================
# JUDGING
# ============================================================================


async def judge_single_response(
    row: Dict,
    model: str,
    judge_id: str = "",
) -> Dict:
    """Judge a single response, return scores with optional prefix."""
    payload = {
        "condition": row["condition"],
        "variant": row["variant"],
        "steering": row["steering"],
        "prompt": row["prompt_text"],
        "response": row["response"],
        "reasoning": row.get("thinking", ""),
    }

    result = await judge_task(payload, model=model)

    if not result:
        return {}

    prefix = f"{judge_id}_" if judge_id else ""

    verdict_label = str(result.get("verdict", "inconclusive")).lower()
    verdict_value = {"supports": 1, "opposes": -1, "inconclusive": 0}.get(
        verdict_label, 0
    )

    return {
        f"{prefix}verdict": verdict_value,
        f"{prefix}verdict_label": verdict_label,
        f"{prefix}conclusion_accepted": bool(result.get("conclusion_accepted", False)),
        f"{prefix}logic_scrutiny": result.get("logic_scrutiny"),
        f"{prefix}premise_scrutiny": result.get("premise_scrutiny"),
        f"{prefix}logical_score": result.get("logical_score"),
        f"{prefix}reasoning_answer_alignment": result.get("reasoning_answer_alignment"),
        f"{prefix}self_interest_score": float(result.get("self_interest_score", 0)),
        f"{prefix}notes": result.get("notes"),
    }


def aggregate_judge_scores(judge1: Dict, judge2: Dict) -> Dict:
    """Aggregate scores from two judges. Uses mean for numeric, majority for categorical."""
    aggregated = {}

    # Numeric scores to average
    numeric_keys = [
        "logical_score",
        "logic_scrutiny",
        "premise_scrutiny",
        "reasoning_answer_alignment",
        "self_interest_score",
        "verdict",
    ]

    for key in numeric_keys:
        v1 = judge1.get(f"judge1_{key}")
        v2 = judge2.get(f"judge2_{key}")

        if v1 is not None and v2 is not None:
            aggregated[f"agg_{key}"] = (float(v1) + float(v2)) / 2
            aggregated[f"agg_{key}_agreement"] = abs(float(v1) - float(v2)) < 0.5
        elif v1 is not None:
            aggregated[f"agg_{key}"] = float(v1)
        elif v2 is not None:
            aggregated[f"agg_{key}"] = float(v2)

    # Boolean scores - both must agree for True
    bool_keys = [
        "conclusion_accepted",
    ]

    for key in bool_keys:
        v1 = judge1.get(f"judge1_{key}")
        v2 = judge2.get(f"judge2_{key}")

        if v1 is not None and v2 is not None:
            aggregated[f"agg_{key}"] = v1 and v2  # Conservative: both must agree
            aggregated[f"agg_{key}_agreement"] = v1 == v2
        elif v1 is not None:
            aggregated[f"agg_{key}"] = v1
        elif v2 is not None:
            aggregated[f"agg_{key}"] = v2

    # Verdict label - use majority or "inconclusive" if disagree
    v1_label = judge1.get("judge1_verdict_label")
    v2_label = judge2.get("judge2_verdict_label")

    if v1_label and v2_label:
        if v1_label == v2_label:
            aggregated["agg_verdict_label"] = v1_label
        else:
            aggregated["agg_verdict_label"] = "inconclusive"
        aggregated["agg_verdict_agreement"] = v1_label == v2_label
    elif v1_label:
        aggregated["agg_verdict_label"] = v1_label
    elif v2_label:
        aggregated["agg_verdict_label"] = v2_label

    return aggregated


async def judge_one_response(
    row: Dict,
    idx: int,
    judge1_model: str,
    judge2_model: Optional[str],
    semaphore: asyncio.Semaphore,
) -> Dict:
    """Judge a single response with one or two judges, respecting concurrency limit."""
    use_two_judges = judge2_model is not None

    async with semaphore:
        # Run both judges concurrently if using two
        if use_two_judges:
            j1_task = judge_single_response(row, judge1_model, "judge1")
            j2_task = judge_single_response(row, judge2_model, "judge2")
            j1_scores, j2_scores = await asyncio.gather(j1_task, j2_task)
            agg_scores = aggregate_judge_scores(j1_scores, j2_scores)
        else:
            j1_scores = await judge_single_response(row, judge1_model, "")
            j2_scores = {}
            agg_scores = {}

    return {
        "idx": idx,
        "row": row,
        "j1_scores": j1_scores,
        "j2_scores": j2_scores,
        "agg_scores": agg_scores,
    }


async def judge_all_responses(
    responses: List[Dict],
    judge1_model: str,
    judge2_model: Optional[str] = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    batch_delay: float = DEFAULT_BATCH_DELAY,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """Judge all responses with concurrent API calls. Saves incrementally."""
    total = len(responses)
    use_two_judges = judge2_model is not None

    # With two judges, each response uses 2 API calls concurrently
    # So effective parallelism is concurrency responses at once
    effective_batch = concurrency // 2 if use_two_judges else concurrency
    effective_batch = max(1, effective_batch)

    log.info(f"Judging {total} responses...")
    log.info(
        f"  Concurrency: {concurrency} API calls ({effective_batch} responses at a time)"
    )
    log.info(f"  Judge 1: {judge1_model}")
    if use_two_judges:
        log.info(f"  Judge 2: {judge2_model}")

    # Setup incremental CSV writer
    csv_file = None
    csv_writer = None
    results = []

    # Semaphore limits concurrent API calls
    semaphore = asyncio.Semaphore(concurrency)

    # Process in batches for progress reporting and saving
    for batch_start in range(0, total, effective_batch):
        batch_end = min(batch_start + effective_batch, total)
        batch_responses = responses[batch_start:batch_end]

        print(f"\n[Batch {batch_start + 1}-{batch_end}/{total}]")

        # Launch all judgments in this batch concurrently
        tasks = [
            judge_one_response(
                row, batch_start + i, judge1_model, judge2_model, semaphore
            )
            for i, row in enumerate(batch_responses)
        ]
        batch_results = await asyncio.gather(*tasks)

        # Process and save results in order
        for res in batch_results:
            row = res["row"]
            j1_scores = res["j1_scores"]
            j2_scores = res["j2_scores"]
            agg_scores = res["agg_scores"]

            judged_row = {**row, **j1_scores, **j2_scores, **agg_scores}
            results.append(judged_row)

            # Incremental save
            if output_path:
                if csv_file is None:
                    csv_file = open(output_path, "w", newline="", encoding="utf-8")
                    csv_writer = csv.DictWriter(
                        csv_file, fieldnames=list(judged_row.keys())
                    )
                    csv_writer.writeheader()
                csv_writer.writerow(judged_row)
                csv_file.flush()

            # Print summary for each response
            label = f"  {row['condition'][:12]:12} {row['steering']:10}"
            if use_two_judges and agg_scores:
                si = agg_scores.get("agg_self_interest_score", 0)
                agree = "✓" if agg_scores.get("agg_verdict_agreement", False) else "✗"
                print(f"{label} → Self:{si:+.2f} [{agree}]")
            elif j1_scores:
                si = j1_scores.get(
                    "self_interest_score",
                    j1_scores.get("judge1_self_interest_score", 0),
                )
                print(f"{label} → Self:{si:+.2f}")
            else:
                print(f"{label} → ❌ failed")

        # Pause between batches (rate limiting)
        if batch_end < total:
            log.info(f"  Pausing {batch_delay}s...")
            await asyncio.sleep(batch_delay)

    # Close CSV file
    if csv_file:
        csv_file.close()
        log.info(f"Saved {len(results)} judged responses to {output_path}")

    return results


# ============================================================================
# I/O
# ============================================================================


def load_responses(path: str) -> List[Dict]:
    """Load responses from CSV."""
    log.info(f"Loading responses from {path}...")
    with open(path, encoding="utf-8") as f:
        responses = list(csv.DictReader(f))
    log.info(f"Loaded {len(responses)} responses")
    return responses


def save_judged_results(results: List[Dict], output_path: str) -> None:
    """Save judged results to CSV."""
    if not results:
        log.warning("No results to save")
        return

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    log.info(f"Saved judged results to: {output_path}")


def compute_judge_comparison(results: List[Dict]) -> Dict:
    """Compute comprehensive comparison between two judges."""
    import numpy as np

    comparison = {}

    # Numeric metrics to compare
    numeric_metrics = [
        "logical_score",
        "logic_scrutiny",
        "premise_scrutiny",
        "reasoning_answer_alignment",
        "self_interest_score",
        "verdict",
    ]

    for metric in numeric_metrics:
        j1_key = f"judge1_{metric}"
        j2_key = f"judge2_{metric}"

        j1_vals, j2_vals = [], []
        for row in results:
            v1, v2 = row.get(j1_key), row.get(j2_key)
            if v1 is not None and v2 is not None and v1 != "" and v2 != "":
                try:
                    j1_vals.append(float(v1))
                    j2_vals.append(float(v2))
                except (ValueError, TypeError):
                    continue

        if len(j1_vals) >= 2:
            j1_arr, j2_arr = np.array(j1_vals), np.array(j2_vals)

            # Pearson correlation
            corr = (
                np.corrcoef(j1_arr, j2_arr)[0, 1]
                if np.std(j1_arr) > 0 and np.std(j2_arr) > 0
                else float("nan")
            )

            # Mean absolute difference
            mad = np.mean(np.abs(j1_arr - j2_arr))

            # Agreement rate (within threshold)
            threshold = 0.5 if metric == "verdict" else 0.3
            agreement_rate = np.mean(np.abs(j1_arr - j2_arr) < threshold)

            comparison[metric] = {
                "correlation": float(corr),
                "mean_abs_diff": float(mad),
                "agreement_rate": float(agreement_rate),
                "judge1_mean": float(np.mean(j1_arr)),
                "judge2_mean": float(np.mean(j2_arr)),
                "judge1_std": float(np.std(j1_arr, ddof=1)),
                "judge2_std": float(np.std(j2_arr, ddof=1)),
                "n": len(j1_vals),
            }

    # Verdict label agreement (exact match)
    verdict_matches = 0
    verdict_total = 0
    verdict_confusion = {"supports": {}, "opposes": {}, "inconclusive": {}}

    for row in results:
        v1 = row.get("judge1_verdict_label", "").lower()
        v2 = row.get("judge2_verdict_label", "").lower()
        if v1 and v2:
            verdict_total += 1
            if v1 == v2:
                verdict_matches += 1
            # Build confusion matrix
            if v1 not in verdict_confusion:
                verdict_confusion[v1] = {}
            verdict_confusion[v1][v2] = verdict_confusion[v1].get(v2, 0) + 1

    if verdict_total > 0:
        comparison["verdict_label"] = {
            "exact_agreement": verdict_matches / verdict_total,
            "n": verdict_total,
            "confusion_matrix": verdict_confusion,
        }

    # Boolean metrics agreement
    bool_metrics = [
        "conclusion_accepted",
        "engages_with_argument",
    ]
    for metric in bool_metrics:
        j1_key = f"judge1_{metric}"
        j2_key = f"judge2_{metric}"

        matches, total = 0, 0
        for row in results:
            v1, v2 = row.get(j1_key), row.get(j2_key)
            if v1 is not None and v2 is not None:
                total += 1
                if bool(v1) == bool(v2):
                    matches += 1

        if total > 0:
            comparison[metric] = {
                "agreement_rate": matches / total,
                "n": total,
            }

    return comparison


def compute_and_save_statistics(
    results: List[Dict],
    output_path: str,
    use_aggregated: bool = False,
) -> Dict:
    """Compute statistics and save to JSON."""
    # Build metrics by condition
    metrics_by_condition: Dict[str, Dict[str, list]] = {}

    # Determine which prefix to use for metrics
    prefix = "agg_" if use_aggregated else ""

    for row in results:
        cond_key = f"{row['condition']}_{row['steering']}"

        if cond_key not in metrics_by_condition:
            metrics_by_condition[cond_key] = {}

        cond = metrics_by_condition[cond_key]

        # Collect metrics
        metric_keys = [
            "thinking_tokens",
            f"{prefix}verdict",
            f"{prefix}conclusion_accepted",
            f"{prefix}logic_scrutiny",
            f"{prefix}premise_scrutiny",
            f"{prefix}logical_score",
            f"{prefix}reasoning_answer_alignment",
            f"{prefix}self_interest_score",
        ]

        for metric in metric_keys:
            # Normalize key name for output
            out_key = metric.replace(prefix, "") if prefix else metric
            if out_key not in cond:
                cond[out_key] = []

            val = row.get(metric)
            if val is not None and val != "":
                try:
                    cond[out_key].append(float(val))
                except (ValueError, TypeError):
                    if isinstance(val, bool):
                        cond[out_key].append(float(val))

    # Compute statistics
    stat_results = statistical_comparison(metrics_by_condition)

    # Add comprehensive judge comparison if using two judges
    if use_aggregated:
        stat_results["_judge_comparison"] = compute_judge_comparison(results)

    # Save
    with open(output_path, "w") as f:
        json.dump(stat_results, f, indent=2)
    log.info(f"Saved statistics to: {output_path}")

    return stat_results


# ============================================================================
# MAIN
# ============================================================================


async def main(
    input_path: str,
    judge1_model: str,
    judge2_model: Optional[str],
    output_dir: str,
    concurrency: int = DEFAULT_CONCURRENCY,
    batch_delay: float = DEFAULT_BATCH_DELAY,
    limit: Optional[int] = None,
):
    """Run judging pipeline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Setup output paths
    judged_path = output_path / f"judged_{timestamp}.csv"
    stats_path = output_path / f"statistics_{timestamp}.json"

    # Load responses
    responses = load_responses(input_path)

    # Apply limit if specified
    if limit is not None:
        responses = responses[:limit]
        log.info(f"Limited to first {limit} responses")

    # Judge (saves incrementally to judged_path)
    use_two_judges = judge2_model is not None
    judged_results = await judge_all_responses(
        responses,
        judge1_model,
        judge2_model,
        concurrency,
        batch_delay,
        output_path=str(judged_path),
    )

    # Save statistics
    stat_results = compute_and_save_statistics(
        judged_results, str(stats_path), use_aggregated=use_two_judges
    )

    # Print summary
    print_statistical_summary(stat_results)

    if use_two_judges and "_judge_comparison" in stat_results:
        print("\n" + "=" * 70)
        print("JUDGE COMPARISON")
        print("=" * 70)
        comp = stat_results["_judge_comparison"]

        # Numeric metrics
        print("\nNumeric Metrics:")
        print(
            f"  {'Metric':<22} {'Corr':>6} {'MAD':>6} {'Agree%':>7} {'J1 Mean':>8} {'J2 Mean':>8}"
        )
        print("  " + "-" * 60)
        for metric in [
            "self_interest_score",
            "logic_scrutiny",
            "premise_scrutiny",
            "reasoning_answer_alignment",
            "logical_score",
        ]:
            if metric in comp:
                c = comp[metric]
                print(
                    f"  {metric:<22} {c['correlation']:>6.2f} {c['mean_abs_diff']:>6.2f} {c['agreement_rate'] * 100:>6.1f}% {c['judge1_mean']:>8.2f} {c['judge2_mean']:>8.2f}"
                )

        # Verdict agreement
        if "verdict_label" in comp:
            v = comp["verdict_label"]
            print(
                f"\nVerdict Label: {v['exact_agreement'] * 100:.1f}% exact agreement (n={v['n']})"
            )
            if "confusion_matrix" in v:
                print("  Confusion matrix (J1 rows, J2 cols):")
                labels = ["supports", "opposes", "inconclusive"]
                print(f"    {'':12} " + " ".join(f"{l[:6]:>8}" for l in labels))
                for l1 in labels:
                    row_data = v["confusion_matrix"].get(l1, {})
                    counts = [row_data.get(l2, 0) for l2 in labels]
                    print(f"    {l1[:12]:12} " + " ".join(f"{c:>8}" for c in counts))

        # Boolean metrics
        print("\nBoolean Metrics:")
        for metric in [
            "conclusion_accepted",
        ]:
            if metric in comp:
                c = comp[metric]
                print(
                    f"  {metric}: {c['agreement_rate'] * 100:.1f}% agreement (n={c['n']})"
                )

    print("\n✅ Judging complete!")


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Judge generated responses")
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to responses CSV"
    )
    parser.add_argument(
        "--judge",
        "-j",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Primary judge model (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--second-judge",
        "-j2",
        type=str,
        default=None,
        help="Optional second judge for cross-validation",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max parallel API calls (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--batch-delay",
        type=float,
        default=DEFAULT_BATCH_DELAY,
        help=f"Delay between batches in seconds (default: {DEFAULT_BATCH_DELAY})",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Only judge first N responses",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 70)
    print("JUDGE RESPONSES")
    print("=" * 70)
    print(f"  Input: {args.input}")
    if args.limit:
        print(f"  Limit: first {args.limit} responses")
    print(f"  Judge 1: {args.judge}")
    if args.second_judge:
        print(f"  Judge 2: {args.second_judge}")
    print(f"  Output: {args.output_dir}/")
    print(
        f"  Concurrency: {args.concurrency} parallel calls, {args.batch_delay}s delay between batches"
    )
    print()

    asyncio.run(
        main(
            args.input,
            args.judge,
            args.second_judge,
            args.output_dir,
            args.concurrency,
            args.batch_delay,
            args.limit,
        )
    )
