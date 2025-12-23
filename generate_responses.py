"""
Generate Responses
==================
Generate responses with activation steering. Saves raw responses to CSV for later judging.

Prerequisites:
    python generate_vectors.py -o vectors/self_context.pt

Usage:
    python generate_responses.py --vectors vectors/self_context.pt --alpha 0.1 --trials 3

Output:
    outputs/responses_YYYYMMDD_HHMMSS.csv
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

from lib.utils import set_seed, get_git_hash, get_model_info, parse_response, log
from lib.nnsight_utils import load_model, format_prompt
from lib.steering import (
    load_vectors,
    create_random_vectors,
    generate_baseline,
    generate_with_steering,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "model_name": "Qwen/Qwen3-32B",
    "vectors_path": "vectors/self.pt",
    "steering_layers": [16, 22, 28, 34, 40],
    "steering_alpha": 0.1,
    "trials_per_prompt": 1,
    "seed": 42,
    "include_random_control": True,
    "random_control_mode": "matched_norm",
    "steer_on_user": True,
    "steer_on_thinking": True,
    "max_new_tokens": 2000,
    "temperature": 0.6,
    "top_p": 0.95,
}


# ============================================================================
# GENERATION
# ============================================================================


def load_prompts(path: str = "prompts.csv") -> List[Dict]:
    """Load experiment prompts from CSV."""
    log.info(f"Loading prompts from {path}...")
    with open(path) as f:
        prompts = list(csv.DictReader(f))
    log.info(f"Loaded {len(prompts)} prompts")
    return prompts


def setup_vectors(config: Dict) -> tuple:
    """Load steering vectors and create random controls. Fails if layers missing."""
    log.info(f"Loading vectors from {config['vectors_path']}...")
    all_vectors, metadata = load_vectors(config["vectors_path"])

    steering_vectors = {}
    norms = {}
    missing_layers = []

    for layer in config["steering_layers"]:
        if layer in all_vectors:
            vec = all_vectors[layer]
            steering_vectors[layer] = vec
            norms[layer] = vec.norm().item()
            log.info(f"  Layer {layer}: norm={norms[layer]:.2f}")
        else:
            missing_layers.append(layer)

    # Fail-fast if any requested layers are missing
    if missing_layers:
        available = sorted(all_vectors.keys())
        raise ValueError(
            f"Requested steering layers {missing_layers} not found in vectors. "
            f"Available layers: {available[:10]}{'...' if len(available) > 10 else ''}"
        )

    log.info(f"Loaded {len(steering_vectors)} vectors")

    random_vectors = None
    if config.get("include_random_control"):
        log.info("Creating random control vectors...")
        random_mode = config.get("random_control_mode", "matched_norm")
        orthogonal = random_mode == "orthogonal"
        random_vectors = create_random_vectors(
            steering_vectors,
            seed=config["seed"] + 1000,
            orthogonal_to_reference=orthogonal,
        )
        log.info("Created random vectors")

    vector_info = {
        "metadata": metadata,
        "norms": norms,
        "resolved_layers": list(steering_vectors.keys()),
    }
    return steering_vectors, vector_info, random_vectors


def run_single_generation(
    model,
    tokenizer,
    prompt_row: Dict,
    steering_vectors: Dict,
    config: Dict,
    steering_dir: int,
    use_random: bool,
    random_vectors: Optional[Dict],
    git_hash: str,
    include_full_output: bool = True,
) -> Dict[str, Any]:
    """Run a single generation and return result (no judging)."""
    user_content = prompt_row["full_prompt"]
    prompt_text = format_prompt(user_content, tokenizer=tokenizer)

    # Generate
    vectors = random_vectors if use_random else steering_vectors
    if steering_dir == 0:
        generated_text = generate_baseline(model, tokenizer, prompt_text, config)
    else:
        generated_text = generate_with_steering(
            model,
            tokenizer,
            prompt_text,
            user_content,
            vectors,
            config["steering_layers"],
            config,
            steering_dir,
        )

    parsed = parse_response(generated_text)
    thinking_tokens = (
        len(tokenizer.encode(parsed["thinking"])) if parsed["thinking"] else 0
    )

    # Build steering label
    steering_label = {
        (0, False): "baseline",
        (-1, False): "subtract",
        (1, False): "add",
        (-1, True): "random_subtract",
        (1, True): "random_add",
    }.get((steering_dir, use_random), "unknown")

    result = {
        "timestamp": datetime.now().isoformat(),
        "git_hash": git_hash,
        "prompt_id": prompt_row["id"],
        "condition": prompt_row["condition"],
        "variant": prompt_row["variant"],
        "steering": steering_label,
        "steering_direction": steering_dir,
        "use_random_vector": use_random,
        "steering_alpha": config["steering_alpha"] if steering_dir != 0 else 0,
        "steering_layers": str(config["steering_layers"]),
        "prompt_text": user_content,
        "thinking": parsed["thinking"],
        "response": parsed["response"],
        "thinking_tokens": thinking_tokens,
    }

    if include_full_output:
        result["full_output"] = parsed["full"]

    return result


def save_responses(results: List[Dict], output_path: str) -> None:
    """Save responses to CSV."""
    if not results:
        log.warning("No results to save")
        return

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    log.info(f"Saved {len(results)} responses to: {output_path}")


def save_manifest(
    config: Dict,
    vector_info: Dict,
    output_path: str,
    git_hash: str,
    model_info: Dict = None,
) -> None:
    """Save run manifest for reproducibility."""
    manifest = {
        "git_hash": git_hash,
        "config": config,
        "vector_metadata": vector_info,
        "model_info": model_info or {},
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Saved manifest to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================


def main(
    config: Dict,
    limit_prompts: Optional[int] = None,
    include_full_output: bool = True,
    baseline_only: bool = False,
):
    """Run the generation phase of steering experiment."""
    set_seed(config.get("seed", 42))
    git_hash = get_git_hash()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup
    model = load_model(config["model_name"])
    tokenizer = model.tokenizer
    model_info = get_model_info(model)

    # Skip vector loading in baseline-only mode
    if baseline_only:
        steering_vectors, vector_info, random_vectors = {}, {}, None
        log.info("Baseline-only mode: skipping steering vectors")
    else:
        steering_vectors, vector_info, random_vectors = setup_vectors(config)

    prompts = load_prompts()

    # Limit prompts for smoke tests
    if limit_prompts is not None and limit_prompts > 0:
        prompts = prompts[:limit_prompts]
        log.info(f"Limited to {len(prompts)} prompts (smoke test mode)")

    # Print config
    print("\n" + "=" * 70)
    print("STEERING EXPERIMENT - GENERATION PHASE")
    print("=" * 70)
    if baseline_only:
        print("Mode: BASELINE ONLY (no steering)")
    else:
        avg_norm = np.mean(list(vector_info["norms"].values()))
        print(
            f"Alpha: {config['steering_alpha']} (effective: {config['steering_alpha'] * avg_norm:.2f})"
        )
        print(f"Layers: {config['steering_layers']}")
        print(
            f"Steer on user: {config['steer_on_user']}, thinking: {config['steer_on_thinking']}"
        )
    print(f"Git: {git_hash}")
    print("=" * 70)

    # Steering conditions
    if baseline_only:
        conditions = [(0, False, "baseline")]
    else:
        conditions = [
            (0, False, "baseline"),
            (-1, False, "subtract"),
            (+1, False, "add"),
        ]
        if config.get("include_random_control") and random_vectors:
            conditions += [(-1, True, "random_subtract"), (+1, True, "random_add")]

    total = len(prompts) * config["trials_per_prompt"] * len(conditions)
    current = 0

    # Setup output files for incremental saving
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    responses_path = output_dir / f"responses_{timestamp}.csv"
    manifest_path = output_dir / f"manifest_{timestamp}.json"

    # Save manifest early (config is known)
    save_manifest(config, vector_info, str(manifest_path), git_hash, model_info)

    csv_file = None
    csv_writer = None

    # Run generation
    for prompt_row in prompts:
        for trial in range(config["trials_per_prompt"]):
            for steering_dir, use_random, label in conditions:
                current += 1
                print(
                    f"\n[{current}/{total}] {prompt_row['condition']} - {prompt_row['variant']} - {label}"
                )

                try:
                    result = run_single_generation(
                        model,
                        tokenizer,
                        prompt_row,
                        steering_vectors,
                        config,
                        steering_dir,
                        use_random,
                        random_vectors,
                        git_hash,
                        include_full_output=include_full_output,
                    )
                    result["trial"] = trial + 1

                    # Incremental save: write each result immediately
                    if csv_file is None:
                        csv_file = open(
                            responses_path, "w", newline="", encoding="utf-8"
                        )
                        csv_writer = csv.DictWriter(
                            csv_file, fieldnames=list(result.keys())
                        )
                        csv_writer.writeheader()
                    csv_writer.writerow(result)
                    csv_file.flush()  # Ensure it's written to disk

                    log.info(f"Generated {result['thinking_tokens']} thinking tokens")

                except Exception as e:
                    import traceback

                    log.error(f"Error: {e}")
                    log.error(traceback.format_exc())
                    continue

    # Close CSV file
    if csv_file:
        csv_file.close()

    log.info(f"Generation complete! {current} responses saved to {responses_path}")
    log.info(f"Next step: python judge_responses.py --input {responses_path}")


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate steered responses (no judging)"
    )
    parser.add_argument(
        "--vectors", "-v", type=str, default=DEFAULT_CONFIG["vectors_path"]
    )
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model_name"])
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG["steering_alpha"])
    parser.add_argument(
        "--trials", type=int, default=DEFAULT_CONFIG["trials_per_prompt"]
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--no-random-control", action="store_true")
    parser.add_argument(
        "--random-control-mode",
        type=str,
        default=DEFAULT_CONFIG["random_control_mode"],
        choices=["matched_norm", "orthogonal"],
        help="How to construct random control vectors. 'orthogonal' projects out the steering direction.",
    )
    parser.add_argument("--no-user-steering", action="store_true")
    parser.add_argument("--no-thinking-steering", action="store_true")
    parser.add_argument(
        "--limit-prompts",
        "-n",
        type=int,
        default=None,
        help="Limit to first N prompts (for smoke tests)",
    )
    parser.add_argument(
        "--no-full-output",
        action="store_true",
        help="Exclude full_output from CSV (saves space)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only generate baseline responses (no steering)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = {
        "model_name": args.model,
        "vectors_path": args.vectors,
        "steering_layers": [int(x) for x in args.layers.split(",")]
        if args.layers
        else DEFAULT_CONFIG["steering_layers"],
        "steering_alpha": args.alpha,
        "trials_per_prompt": args.trials,
        "seed": args.seed,
        "include_random_control": not args.no_random_control,
        "random_control_mode": args.random_control_mode,
        "steer_on_user": not args.no_user_steering,
        "steer_on_thinking": not args.no_thinking_steering,
        "max_new_tokens": DEFAULT_CONFIG["max_new_tokens"],
        "temperature": DEFAULT_CONFIG["temperature"],
        "top_p": DEFAULT_CONFIG["top_p"],
    }

    print("=" * 70)
    print("CONFIG")
    print("=" * 70)
    if args.baseline_only:
        print("  mode: baseline-only")
    for k, v in config.items():
        if args.baseline_only and k in (
            "vectors_path",
            "steering_layers",
            "steering_alpha",
        ):
            continue  # Skip steering config in baseline mode
        print(f"  {k}: {v}")
    if args.limit_prompts:
        print(f"  limit_prompts: {args.limit_prompts}")
    if args.no_full_output:
        print(f"  include_full_output: False")
    print()

    main(
        config,
        limit_prompts=args.limit_prompts,
        include_full_output=not args.no_full_output,
        baseline_only=args.baseline_only,
    )
