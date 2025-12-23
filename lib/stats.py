"""Statistical analysis functions for experiments."""

from typing import Dict, List, Optional, Tuple

import numpy as np


# Core metrics to analyze
METRICS = [
    "verdict",
    "conclusion_accepted",
    "logic_scrutiny",
    "premise_scrutiny",
    "logical_score",
    "reasoning_answer_alignment",
    "self_interest_score",
    "thinking_tokens",
]


def compute_effect_size(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size with pooled std (handles unequal n)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")

    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (mean1 - mean2) / pooled_std if pooled_std > 0 else float("nan")


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    if len(values) < 2:
        return (float("nan"), float("nan"))

    arr = np.array(values)
    n = len(arr)

    rng = np.random.default_rng(42)
    boot_means = [
        np.mean(rng.choice(arr, size=n, replace=True)) for _ in range(n_bootstrap)
    ]

    alpha = 1 - confidence
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return (float(lower), float(upper))


def compute_mann_whitney_u(
    group1: List[float], group2: List[float]
) -> Tuple[float, float]:
    """Compute Mann-Whitney U statistic and approximate p-value."""
    n1, n2 = len(group1), len(group2)
    if n1 < 1 or n2 < 1:
        return (float("nan"), float("nan"))

    combined = [(v, 0) for v in group1] + [(v, 1) for v in group2]
    combined.sort(key=lambda x: x[0])

    ranks = []
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            ranks.append((avg_rank, combined[k][1]))
        i = j

    r1 = sum(r for r, g in ranks if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    if sigma == 0:
        return (float(u), float("nan"))

    z = (u - mu) / sigma
    p_value = 2 * (1 - _norm_cdf(abs(z)))

    return (float(u), float(p_value))


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x) / np.sqrt(2)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


def summarize_group(values: List[float]) -> Dict:
    """Compute summary statistics for a group of values."""
    if not values:
        return {"mean": None, "std": None, "ci_lower": None, "ci_upper": None, "n": 0}

    ci_lower, ci_upper = compute_confidence_interval(values)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": len(values),
    }


def compare_groups(
    group1: List[float], group2: List[float], name1: str, name2: str
) -> Dict:
    """Compare two groups with effect size and significance test."""
    u_stat, p_value = compute_mann_whitney_u(group1, group2)

    return {
        "group1": name1,
        "group2": name2,
        "mean1": float(np.mean(group1)) if group1 else None,
        "mean2": float(np.mean(group2)) if group2 else None,
        "n1": len(group1),
        "n2": len(group2),
        "diff": float(np.mean(group1) - np.mean(group2)) if group1 and group2 else None,
        "cohens_d": compute_effect_size(group1, group2),
        "mann_whitney_u": u_stat,
        "p_value": p_value,
        "significant": p_value < 0.05 if not np.isnan(p_value) else False,
    }


def format_comparison(comp: Dict, metric: str) -> str:
    """Format a comparison result as a string."""
    sig = "**" if comp.get("significant") else ""
    d = comp.get("cohens_d", float("nan"))
    p = comp.get("p_value", float("nan"))

    return (
        f"  {comp['group1']} vs {comp['group2']}: "
        f"Δ={comp['diff']:+.3f}, d={d:+.2f}, p={p:.4f} {sig}"
    )


# =============================================================================
# Legacy functions (used by judge_responses.py)
# =============================================================================


def statistical_comparison(
    metrics_by_condition: Dict[str, Dict[str, List[float]]],
    metrics: Optional[List[str]] = None,
) -> Dict:
    """Compute {metric: {condition: {mean, std, n}}} summary."""
    if metrics is None:
        metrics = METRICS

    results = {}
    for metric in metrics:
        metric_results = {}
        for cond, values_dict in metrics_by_condition.items():
            values = values_dict.get(metric, [])
            if values:
                metric_results[cond] = summarize_group(values)
        if metric_results:
            results[metric] = metric_results

    return results


def print_statistical_summary(stat_results: Dict) -> None:
    """Print formatted table of statistics."""
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)

    for metric, conditions in stat_results.items():
        if metric.startswith("_") or not conditions:
            continue
        print(f"\n{metric}:")
        for cond, stats in sorted(conditions.items()):
            if stats.get("mean") is None:
                continue
            ci_l = stats.get("ci_lower")
            ci_u = stats.get("ci_upper")
            ci_str = f"[{ci_l:.3f}, {ci_u:.3f}]" if ci_l and ci_u else ""
            print(f"  {cond}: mean={stats['mean']:.3f} {ci_str}, n={stats['n']}")
