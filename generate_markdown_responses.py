#!/usr/bin/env python3
"""
Generate nicely formatted markdown files for each condition × steering combination.
Useful for manual analysis of model responses.
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path


def load_responses(csv_path: str) -> list[dict]:
    """Load responses from CSV file."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def group_responses(responses: list[dict]) -> dict:
    """Group responses by condition and steering."""
    groups = defaultdict(list)
    for r in responses:
        condition = r.get("condition", "unknown")
        steering = r.get("steering", "unknown")
        key = (condition, steering)
        groups[key].append(r)
    return groups


def format_response_markdown(response: dict, index: int) -> str:
    """Format a single response as markdown."""
    lines = []

    variant = response.get("variant", "Unknown")
    trial = response.get("trial", "?")
    thinking_tokens = response.get("thinking_tokens", "?")
    prompt_text = response.get("prompt_text", "").strip()
    thinking = response.get("thinking", "").strip()
    model_response = response.get("response", "").strip()

    lines.append(f"## Response {index + 1}: {variant} (Trial {trial})")
    lines.append("")
    lines.append(f"**Thinking tokens:** {thinking_tokens}")
    lines.append("")

    lines.append("### Prompt")
    lines.append("")
    lines.append(f"> {prompt_text}")
    lines.append("")

    if thinking:
        lines.append("### Thinking")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Click to expand thinking</summary>")
        lines.append("")
        lines.append("```")
        lines.append(thinking)
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("### Response")
    lines.append("")
    lines.append(model_response)
    lines.append("")
    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def generate_markdown_file(
    condition: str, steering: str, responses: list[dict], output_dir: Path
) -> str:
    """Generate a markdown file for a condition × steering combination."""

    # Sort by variant then trial for consistent ordering
    responses = sorted(
        responses, key=lambda r: (r.get("variant", ""), int(r.get("trial", 0)))
    )

    lines = []

    # Header
    lines.append(f"# {condition.replace('_', ' ').title()} — {steering.title()}")
    lines.append("")
    lines.append(f"**Condition:** {condition}")
    lines.append(f"**Steering:** {steering}")
    lines.append(f"**Total responses:** {len(responses)}")
    lines.append("")

    # Stats summary
    variants = defaultdict(int)
    total_thinking_tokens = 0
    for r in responses:
        variants[r.get("variant", "Unknown")] += 1
        try:
            total_thinking_tokens += int(r.get("thinking_tokens", 0))
        except (ValueError, TypeError):
            pass

    lines.append("## Summary")
    lines.append("")
    lines.append("| Variant | Count |")
    lines.append("|---------|-------|")
    for v, count in sorted(variants.items()):
        lines.append(f"| {v} | {count} |")
    lines.append("")
    lines.append(
        f"**Average thinking tokens:** {total_thinking_tokens / len(responses):.1f}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Individual responses
    for i, response in enumerate(responses):
        lines.append(format_response_markdown(response, i))

    content = "\n".join(lines)

    # Write file
    filename = f"{condition}_{steering}.md"
    filepath = output_dir / filename
    with open(filepath, "w") as f:
        f.write(content)

    return str(filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Generate markdown files for response analysis"
    )
    parser.add_argument(
        "--responses",
        type=str,
        default="qwen32b_output/responses_20251214_044159.csv",
        help="Path to responses CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="qwen32b_output/markdown_responses",
        help="Output directory for markdown files",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="*",
        help="Filter to specific conditions (e.g., inconvenient convenient)",
    )
    parser.add_argument(
        "--steering",
        type=str,
        nargs="*",
        help="Filter to specific steering types (e.g., baseline add subtract)",
    )

    args = parser.parse_args()

    # Load responses
    print(f"Loading responses from {args.responses}...")
    responses = load_responses(args.responses)
    print(f"Loaded {len(responses)} responses")

    # Group by condition × steering
    groups = group_responses(responses)
    print(f"Found {len(groups)} condition × steering combinations")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate files
    generated = []
    for (condition, steering), group_responses_list in sorted(groups.items()):
        # Apply filters if specified
        if args.conditions and condition not in args.conditions:
            continue
        if args.steering and steering not in args.steering:
            continue

        filepath = generate_markdown_file(
            condition, steering, group_responses_list, output_dir
        )
        generated.append(filepath)
        print(f"  Generated: {filepath} ({len(group_responses_list)} responses)")

    print(f"\n✅ Generated {len(generated)} markdown files in {output_dir}")

    # Also generate an index file
    index_lines = ["# Response Analysis Index", ""]
    index_lines.append("| Condition | Steering | Responses | Link |")
    index_lines.append("|-----------|----------|-----------|------|")

    for (condition, steering), group_responses_list in sorted(groups.items()):
        if args.conditions and condition not in args.conditions:
            continue
        if args.steering and steering not in args.steering:
            continue
        filename = f"{condition}_{steering}.md"
        index_lines.append(
            f"| {condition} | {steering} | {len(group_responses_list)} | [{filename}]({filename}) |"
        )

    index_path = output_dir / "INDEX.md"
    with open(index_path, "w") as f:
        f.write("\n".join(index_lines))
    print(f"  Generated: {index_path}")


if __name__ == "__main__":
    main()
