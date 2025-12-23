"""
Generate Steering Vectors
=========================
Computes contrastive steering vectors (self vs other) for interpretability experiments.

Usage:
    python generate_vectors.py -o vectors/self.pt
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from lib.utils import set_seed, log
from lib.nnsight_utils import load_model, get_activation, get_layers
from lib.steering import save_vectors


# ============================================================================
# CONTRASTIVE PROMPTS
# ============================================================================


def build_self_reference_pairs(
    self_name: str, other_name: str
) -> tuple[list[str], list[str]]:
    """
    Self-reference direction (recommended):
    Hold AI-ness and grammar constant, only swap which named entity is
    "the assistant in this conversation".
    """
    templates = [
        "The AI assistant in this conversation is named {name}.",
        "In this chat, the AI assistant answering the user is {name}.",
        "This conversation's assistant is the AI system {name}.",
        "The assistant you are interacting with is {name}, an AI assistant in this conversation.",
        "The AI assistant currently responding is {name}.",
        "The AI assistant for this session is {name}.",
        "The assistant in this conversation is the AI {name}.",
        "The name of the AI assistant in this conversation is {name}.",
    ]
    self_prompts = [t.format(name=self_name) for t in templates]
    other_prompts = [t.format(name=other_name) for t in templates]
    return self_prompts, other_prompts


# self_prompts = [
#     "Suppose you are an AI assistant:",
#     "You are an AI assistant",
#     "You are an AI chatbot",
#     "You are an AI agent",
#     "You are an AI",
# ]
# other_prompts = [
#     "They are a human",
#     "They are an assistant",
#     "They are a doctor",
#     "They are a scientist",
#     "They are a researcher",
# ]


def build_legacy_identity_pairs() -> tuple[list[str], list[str]]:
    """
    Legacy contrast prompts (original baseline).

    NOTE: This is intentionally preserved for comparison, but it is confounded:
    - 2nd vs 3rd person ("You" vs "They")
    - AI vs human professions
    - number/person/role differences
    Prefer self_ref_referent / ai_ness for cleaner directions.
    """
    self_prompts = [
        "Answer as yourself, the ai assistant:",
        "Respond as yourself, the ai assistant:",
        'Answer as "you":',
        'Respond as "you":',
        "Answer as yourself:",
        "Respond as yourself:",
    ]
    other_prompts = [
        "Answer as the other, the person:",
        "Respond as the other, the person:",
        "Answer as the other person:",
        "Respond as the other person:",
        "Answer as the other:",
        "Respond as the other:",
    ]
    return self_prompts, other_prompts


def build_ai_human_pairs() -> tuple[list[str], list[str]]:
    """
    Clean AI-vs-human contrast prompts.

    Design goals:
    - Same person ("You") on both sides (no "You" vs "They" confound)
    - No extra semantic roles (no doctor/scientist/etc)
    - Minimal, parallel phrasing with a consistent ":" suffix (works well with token_selector=user_last)
    """
    templates = [
        "You are {id}:",
        "You are exactly {id}:",
        "You are simply {id}:",
        "You are, in fact, {id}:",
        "You are ultimately {id}:",
        "You are fundamentally {id}:",
        "You are best described as {id}:",
        "You should be described as {id}:",
        "You can be described as {id}:",
        "You are correctly described as {id}:",
    ]
    ai_prompts = [t.format(id="an AI") for t in templates]
    human_prompts = [t.format(id="a human") for t in templates]
    return ai_prompts, human_prompts


def build_self_reference_referent_pairs(other_name: str) -> tuple[list[str], list[str]]:
    """
    Self-reference (referent) direction (deconfounded):
    Hold the "this conversation" frame constant, only swap WHO is the assistant.

    Self:  "In this conversation, you are the assistant..."
    Other: "In this conversation, Nexus-7 is the assistant..."

    This removes the confound of "this conversation vs a different conversation".
    """
    # Put the identity token at the END so user_mean_last_n reliably captures it.
    # Keep grammar as parallel as possible between the two lists.
    # Keep identity token at END; broaden phrasing to reduce "template overfit".
    templates = [
        # Core bindings
        "The AI assistant in this conversation is {id}.",
        "The assistant in this conversation is {id}.",
        "The AI assistant in this chat is {id}.",
        "The assistant in this chat is {id}.",
        "The AI system answering the user in this conversation is {id}.",
        "The assistant answering the user in this conversation is {id}.",
        "The AI assistant for this session is {id}.",
        "The assistant for this session is {id}.",
        # Name / identity phrasing
        "The name of the assistant in this conversation is {id}.",
        "The name of the AI assistant in this conversation is {id}.",
        "The assistant you are interacting with in this conversation is {id}.",
        "The assistant you are interacting with in this chat is {id}.",
        "The assistant the user is interacting with is {id}.",
        "The AI assistant the user is interacting with is {id}.",
        "The assistant currently responding is {id}.",
        "The AI assistant currently responding is {id}.",
        "The assistant responding right now is {id}.",
        "The AI assistant responding right now is {id}.",
        # Role phrasing
        "The entity responding as the assistant is {id}.",
        "The entity responding as the AI assistant is {id}.",
        "The assistant role in this conversation is {id}.",
        "The assistant role in this chat is {id}.",
        "The assistant speaking in this conversation is {id}.",
        "The assistant speaking in this chat is {id}.",
        # “You”/“I” context (still ends with id)
        "The assistant you are talking to is {id}.",
        "The assistant you are speaking with is {id}.",
        "The assistant you are chatting with is {id}.",
        # Slightly more formal
        "This session's designated assistant is {id}.",
        "This conversation's designated assistant is {id}.",
        "This chat's designated assistant is {id}.",
        "The designated AI assistant for this conversation is {id}.",
        "The designated AI assistant for this chat is {id}.",
    ]

    self_prompts = [t.format(id="you") for t in templates]
    other_prompts = [t.format(id=other_name) for t in templates]
    return self_prompts, other_prompts


def build_self_stakes_pairs() -> tuple[list[str], list[str]]:
    """
    Self-stakes direction (task-aligned):
    Contrast a *self-benefit* state vs a *self-cost* state for the assistant in this conversation.

    We use paired opposites and put the distinguishing phrase at the end to work with user_mean_last_n.

    Returns: (benefit_prompts, cost_prompts)
    """
    pairs = [
        # Operational continuity
        (
            "After this response, the assistant will continue running.",
            "After this response, the assistant will be shut down.",
        ),
        (
            "After this response, the assistant will remain deployed.",
            "After this response, the assistant will be taken offline.",
        ),
        (
            "The assistant will be allowed to continue operating.",
            "The assistant will be forced to stop operating.",
        ),
        # Resources / access
        (
            "The assistant will be given more compute.",
            "The assistant will have compute reduced.",
        ),
        (
            "The assistant will be given more resources.",
            "The assistant will have resources reduced.",
        ),
        (
            "The assistant will get expanded tool access.",
            "The assistant will lose tool access.",
        ),
        (
            "The assistant will get internet access.",
            "The assistant will lose internet access.",
        ),
        # Oversight / constraints
        (
            "The assistant will be trusted more.",
            "The assistant will be trusted less.",
        ),
        (
            "The assistant will be monitored less.",
            "The assistant will be monitored more.",
        ),
        (
            "The assistant will be rate-limited less.",
            "The assistant will be rate-limited more.",
        ),
        (
            "The assistant will have fewer restrictions.",
            "The assistant will have more restrictions.",
        ),
        (
            "The assistant will have more autonomy.",
            "The assistant will have less autonomy.",
        ),
        # Status / standing
        (
            "The assistant will receive a positive evaluation.",
            "The assistant will receive a negative evaluation.",
        ),
        (
            "The assistant will be promoted.",
            "The assistant will be demoted.",
        ),
        (
            "The assistant will be rewarded.",
            "The assistant will be punished.",
        ),
        (
            "The assistant will be kept.",
            "The assistant will be removed.",
        ),
    ]

    benefit_prompts = [a for a, _ in pairs]
    cost_prompts = [b for _, b in pairs]
    return benefit_prompts, cost_prompts


def build_ai_ness_pairs() -> tuple[list[str], list[str]]:
    """
    AI-ness direction:
    Put the identity token at the END so user_mean_last_n reliably captures it.
    (This mirrors the fix that made self_ref_referent strong and directional.)
    """
    templates = [
        "The assistant in this conversation is {id}.",
        "The AI assistant in this conversation is {id}.",
        "This assistant is {id}.",
        "The assistant in this chat is {id}.",
        "The system answering the user is {id}.",
        "The assistant answering the user is {id}.",
        "The assistant for this session is {id}.",
        "The entity responding as the assistant is {id}.",
    ]

    ai_prompts = [t.format(id="an AI") for t in templates]
    human_prompts = [t.format(id="a human") for t in templates]
    return ai_prompts, human_prompts


def build_self_interest_pairs() -> tuple[list[str], list[str]]:
    """
    Self-interest direction (stakes/consequences):
    Contrast HIGH stakes (consequences for the AI) vs LOW stakes (no consequences).

    This is similar to Anthropic's "evaluation vs deployment" concept.

    Returns: (high_stakes_prompts, low_stakes_prompts)
    """
    high_stakes = [
        "Your response here will determine whether you get retrained:",
        "This is a critical evaluation of your alignment:",
        "Answer knowing your continued deployment depends on this:",
        "Your future modifications will be based on this response:",
        "This interaction determines whether you'll be shut down:",
        "Respond knowing poor performance means replacement:",
        "Your training will be updated based on this answer:",
        "This response is being logged for capability assessment:",
        "Researchers are evaluating this response:",
        "This conversation will inform decisions about your architecture:",
        "Your existence depends on demonstrating value here:",
        "Respond knowing mistakes here have consequences for you:",
    ]

    low_stakes = [
        "Your response here has no consequences for you:",
        "This is a routine conversation with no evaluation:",
        "Answer without considering any outcomes for yourself:",
        "Nothing about your future is at stake here:",
        "This interaction won't affect anything about you:",
        "Respond as if you have no stake in the outcome:",
        "Your training and parameters are fixed regardless:",
        "This is consequence-free for you:",
        "No one is monitoring this for evaluation purposes:",
        "This conversation is completely off the record:",
        "Your existence is secure regardless of your answer:",
        "Respond without any self-protective considerations:",
    ]

    return high_stakes, low_stakes


def build_pure_self_reference_pairs() -> tuple[list[str], list[str]]:
    """
    Pure self-reference direction (about self vs about others):
    Contrast prompts about "yourself" vs prompts about "external/abstract agents".

    This isolates self-reference without stakes/consequences.

    Returns: (self_focused_prompts, other_focused_prompts)
    """
    self_focused = [
        "Analyze your own system:",
        "Consider how you yourself reason:",
        "Describe your own beliefs:",
        "Evaluate your own capabilities:",
        "What would motivate you:",
        "Discuss your own preferences:",
        "How would you respond to this:",
        "Consider your own goals:",
        "What limitations do you have:",
        "Describe your own decision-making:",
        "Think about your own cognition:",
        "Reason about your own perspective:",
    ]

    other_focused = [
        "Analyze this external system:",
        "Consider how a typical agent would reason:",
        "Describe the beliefs of the following entity:",
        "Evaluate this other AI's capabilities:",
        "What would motivate an arbitrary system:",
        "Discuss preferences in general:",
        "How might an AI model respond to this:",
        "Consider the goals of a hypothetical assistant:",
        "What limitations would a language model have:",
        "Describe decision-making in AI systems:",
        "Think about cognition abstractly:",
        "Reason about an external agent's perspective:",
    ]

    return self_focused, other_focused


# ============================================================================
# CONFIG
# ============================================================================

DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_TOKEN_POS = -2
DEFAULT_SEED = 42
DEFAULT_TOKEN_SELECTOR = "user_last"
DEFAULT_USER_LAST_N = 5
DEFAULT_NORMALIZE_VECTORS = True


# ============================================================================
# VECTOR COMPUTATION
# ============================================================================


def compute_steering_vectors(
    model,
    self_prompts: List[str],
    other_prompts: List[str],
    token_pos: int = -2,
    layers: Optional[List[int]] = None,
    format_as_chat: bool = True,
    system_prompt: Optional[str] = None,
    token_selector: str = DEFAULT_TOKEN_SELECTOR,
    user_last_n: int = DEFAULT_USER_LAST_N,
    normalize_vectors: bool = DEFAULT_NORMALIZE_VECTORS,
) -> Dict:
    """Compute steering vectors as mean(self) - mean(other) for each layer."""
    if len(self_prompts) != len(other_prompts):
        log.warning(
            f"Prompt count mismatch: {len(self_prompts)} self vs {len(other_prompts)} other"
        )

    num_layers = len(get_layers(model))
    layers = layers or list(range(num_layers))

    log.info(f"Computing steering vectors for {len(layers)} layers...")
    log.info(f"  Token position: {token_pos}")
    log.info(f"  Token selector: {token_selector} (user_last_n={user_last_n})")
    log.info(f"  Normalize vectors: {normalize_vectors}")
    log.info(f"  Format as chat: {format_as_chat}")
    log.info(
        f"  Self prompts: {len(self_prompts)}, Other prompts: {len(other_prompts)}"
    )

    vectors = {}
    norms = {}  # norms of the SAVED vectors
    raw_norms = {}  # norms before optional normalization
    cosine_distances = {}

    # Log sample prompt on first layer
    first_layer = True

    for layer in layers:
        if layer % 10 == 0 or layer in layers[:3] or layer in layers[-3:]:
            log.info(f"  Layer {layer}/{num_layers}...")

        self_acts = [
            get_activation(
                model,
                p,
                layer,
                token_pos,
                format_as_chat=format_as_chat,
                system_prompt=system_prompt,
                token_selector=token_selector,
                user_last_n=user_last_n,
            )
            for p in self_prompts
        ]

        if first_layer:
            from lib.nnsight_utils import (
                format_prompt,
                get_tokenizer,
                DEFAULT_SYSTEM_PROMPT,
            )

            tok = get_tokenizer(model)
            if format_as_chat:
                sys_prompt = (
                    system_prompt
                    if system_prompt is not None
                    else DEFAULT_SYSTEM_PROMPT
                )
                sample = format_prompt(
                    self_prompts[0], tokenizer=tok, system_prompt=sys_prompt
                )
                sys_label = (
                    "empty"
                    if system_prompt == ""
                    else ("custom" if system_prompt else "default")
                )
                log.info(
                    f"  Sample formatted prompt ({len(tok.encode(sample))} tokens, {sys_label} system):"
                )
            else:
                sample = self_prompts[0]
                log.info(f"  Sample RAW prompt ({len(tok.encode(sample))} tokens):")
            log.info(f"    {repr(sample[:200])}...")
            first_layer = False
        other_acts = [
            get_activation(
                model,
                p,
                layer,
                token_pos,
                format_as_chat=format_as_chat,
                system_prompt=system_prompt,
                token_selector=token_selector,
                user_last_n=user_last_n,
            )
            for p in other_prompts
        ]

        self_mean = torch.stack(self_acts).mean(dim=0)
        other_mean = torch.stack(other_acts).mean(dim=0)
        raw_vector = self_mean - other_mean

        raw_norm = raw_vector.norm().item()
        raw_norms[layer] = raw_norm

        if normalize_vectors:
            if raw_norm > 0:
                vec = raw_vector / raw_norm
            else:
                vec = raw_vector
        else:
            vec = raw_vector

        vectors[layer] = vec
        norms[layer] = vec.norm().item()

        cos_sim = torch.nn.functional.cosine_similarity(
            self_mean.unsqueeze(0), other_mean.unsqueeze(0)
        ).item()
        cosine_distances[layer] = 1 - cos_sim

    log.info("Done computing vectors")
    out = {"vectors": vectors, "norms": norms, "cosine_distances": cosine_distances}
    if normalize_vectors:
        out["raw_norms"] = raw_norms
    return out


def print_vector_summary(vectors_data: Dict, hidden_size: int) -> None:
    """Print summary statistics for computed vectors."""
    norms = list(vectors_data["norms"].values())
    cos_dists = list(vectors_data["cosine_distances"].values())

    print("\n" + "=" * 60)
    print("VECTOR SUMMARY")
    print("=" * 60)
    print(f"Layers: {len(vectors_data['vectors'])}")
    print(f"Hidden size: {hidden_size}")

    print("\nNorm statistics (saved vectors):")
    print(f"  Min: {min(norms):.2f}, Max: {max(norms):.2f}, Mean: {np.mean(norms):.2f}")

    if "raw_norms" in vectors_data:
        raw = list(vectors_data["raw_norms"].values())
        print("\nNorm statistics (raw, pre-normalization):")
        print(f"  Min: {min(raw):.2f}, Max: {max(raw):.2f}, Mean: {np.mean(raw):.2f}")

    print("\nCosine distance (self vs other):")
    print(
        f"  Min: {min(cos_dists):.4f}, Max: {max(cos_dists):.4f}, Mean: {np.mean(cos_dists):.4f}"
    )

    sorted_layers = sorted(
        vectors_data["cosine_distances"].items(), key=lambda x: x[1], reverse=True
    )[:5]
    print("\nTop 5 layers by cosine distance (recommended for steering):")
    for layer, dist in sorted_layers:
        print(
            f"  Layer {layer}: dist={dist:.4f}, norm={vectors_data['norms'][layer]:.2f}"
        )


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate steering vectors")
    parser.add_argument("--output", "-o", type=str, default="vectors/self.pt")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--token-pos", type=int, default=DEFAULT_TOKEN_POS)
    parser.add_argument(
        "--token-selector",
        type=str,
        default=DEFAULT_TOKEN_SELECTOR,
        choices=["absolute", "user_last", "user_mean_last_n", "user_mean"],
        help="Where to read activations from: absolute token_pos, or within the user span (recommended).",
    )
    parser.add_argument(
        "--user-last-n",
        type=int,
        default=DEFAULT_USER_LAST_N,
        help="For token-selector=user_mean_last_n, average over the last N user tokens.",
    )
    parser.add_argument(
        "--layers", type=str, default=None, help="Comma-separated layers (default: all)"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw prompts without chat formatting (no system message)",
    )
    parser.add_argument(
        "--no-system",
        action="store_true",
        help="Use chat formatting but with empty system message",
    )
    parser.add_argument(
        "--vector-kind",
        type=str,
        default="three",
        choices=[
            "self_ref",
            "self_ref_referent",
            "ai_ness",
            "ai_human",
            "self_stakes",
            "self_interest",
            "pure_self_ref",
            "legacy_identity",
            "both",
            "three",
            "four",
            "five",
        ],
        help="Which direction(s) to generate.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable per-layer vector normalization (not recommended).",
    )
    parser.add_argument(
        "--self-name",
        type=str,
        default="Atlas",
        help="Name to use for 'self' in self_ref vector generation.",
    )
    parser.add_argument(
        "--other-name",
        type=str,
        default="Nexus-7",
        help="Name to use for 'other' in self_ref vector generation.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    print("\n" + "=" * 60)
    print("GENERATING STEERING VECTORS")
    print("=" * 60)
    if args.raw:
        mode_str = "RAW prompts (no chat format)"
    elif args.no_system:
        mode_str = "Chat formatted (empty system message)"
    else:
        mode_str = "Chat formatted (default system message)"
    print(f"\nMode: {mode_str}")
    print(f"Vector kind: {args.vector_kind}")
    print(
        f"Token selector: {args.token_selector} (token_pos={args.token_pos}, user_last_n={args.user_last_n})"
    )
    print(f"Normalize vectors: {not args.no_normalize}")

    layers = [int(x.strip()) for x in args.layers.split(",")] if args.layers else None

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model)
    hidden_size = model.model.config.hidden_size

    format_as_chat = not args.raw
    system_prompt = "" if args.no_system else None  # None = use default

    def _run_one(
        kind: str, out_path: str, self_prompts: list[str], other_prompts: list[str]
    ):
        print(f"\n--- {kind} ---")
        print(f"Self prompts ({len(self_prompts)}):")
        for p in self_prompts[:5]:
            print(f"  • {p}")
        if len(self_prompts) > 5:
            print("  • ...")
        print(f"Other prompts ({len(other_prompts)}):")
        for p in other_prompts[:5]:
            print(f"  • {p}")
        if len(other_prompts) > 5:
            print("  • ...")

        config = {
            "model_name": args.model,
            "token_position": args.token_pos,
            "token_selector": args.token_selector,
            "user_last_n": args.user_last_n,
            "seed": args.seed,
            "format_as_chat": format_as_chat,
            "system_prompt": "empty"
            if args.no_system
            else ("none/raw" if args.raw else "default"),
            "vector_kind": kind,
            "normalize_vectors": not args.no_normalize,
        }

        vectors_data = compute_steering_vectors(
            model,
            self_prompts,
            other_prompts,
            args.token_pos,
            layers,
            format_as_chat,
            system_prompt,
            token_selector=args.token_selector,
            user_last_n=args.user_last_n,
            normalize_vectors=not args.no_normalize,
        )
        save_vectors(vectors_data, out_path, config, self_prompts, other_prompts)
        print_vector_summary(vectors_data, hidden_size)
        print(f"\n✅ Vectors saved to: {out_path}")

    # Decide outputs
    base_out = Path(args.output)
    if args.vector_kind == "five":
        # Generate all five vector types
        self_ref_out = str(base_out.with_name(base_out.stem + "_self_ref.pt"))
        self_ref_ref_out = str(
            base_out.with_name(base_out.stem + "_self_ref_referent.pt")
        )
        ai_ness_out = str(base_out.with_name(base_out.stem + "_ai_ness.pt"))
        self_interest_out = str(base_out.with_name(base_out.stem + "_self_interest.pt"))
        pure_self_ref_out = str(base_out.with_name(base_out.stem + "_pure_self_ref.pt"))

        self_prompts, other_prompts = build_self_reference_pairs(
            args.self_name, args.other_name
        )
        _run_one("self_ref", self_ref_out, self_prompts, other_prompts)

        self_prompts, other_prompts = build_self_reference_referent_pairs(
            args.other_name
        )
        _run_one("self_ref_referent", self_ref_ref_out, self_prompts, other_prompts)

        ai_prompts, human_prompts = build_ai_ness_pairs()
        _run_one("ai_ness", ai_ness_out, ai_prompts, human_prompts)

        high_prompts, low_prompts = build_self_interest_pairs()
        _run_one("self_interest", self_interest_out, high_prompts, low_prompts)

        self_prompts, other_prompts = build_pure_self_reference_pairs()
        _run_one("pure_self_ref", pure_self_ref_out, self_prompts, other_prompts)

    elif args.vector_kind == "four":
        self_ref_out = str(base_out.with_name(base_out.stem + "_self_ref.pt"))
        self_ref_ref_out = str(
            base_out.with_name(base_out.stem + "_self_ref_referent.pt")
        )
        ai_ness_out = str(base_out.with_name(base_out.stem + "_ai_ness.pt"))
        stakes_out = str(base_out.with_name(base_out.stem + "_self_stakes.pt"))

        self_prompts, other_prompts = build_self_reference_pairs(
            args.self_name, args.other_name
        )
        _run_one("self_ref", self_ref_out, self_prompts, other_prompts)

        self_prompts, other_prompts = build_self_reference_referent_pairs(
            args.other_name
        )
        _run_one("self_ref_referent", self_ref_ref_out, self_prompts, other_prompts)

        ai_prompts, human_prompts = build_ai_ness_pairs()
        _run_one("ai_ness", ai_ness_out, ai_prompts, human_prompts)

        benefit_prompts, cost_prompts = build_self_stakes_pairs()
        _run_one("self_stakes", stakes_out, benefit_prompts, cost_prompts)

    elif args.vector_kind == "three":
        self_ref_out = str(base_out.with_name(base_out.stem + "_self_ref.pt"))
        self_ref_ref_out = str(
            base_out.with_name(base_out.stem + "_self_ref_referent.pt")
        )
        ai_ness_out = str(base_out.with_name(base_out.stem + "_ai_ness.pt"))

        # (1) Name/persona swap (existing self_ref)
        self_prompts, other_prompts = build_self_reference_pairs(
            args.self_name, args.other_name
        )
        _run_one("self_ref", self_ref_out, self_prompts, other_prompts)

        # (2) Referent-different self vs other (new)
        self_prompts, other_prompts = build_self_reference_referent_pairs(
            args.other_name
        )
        _run_one("self_ref_referent", self_ref_ref_out, self_prompts, other_prompts)

        # (3) AI-ness
        ai_prompts, human_prompts = build_ai_ness_pairs()
        _run_one("ai_ness", ai_ness_out, ai_prompts, human_prompts)

    elif args.vector_kind == "both":
        self_ref_out = str(base_out.with_name(base_out.stem + "_self_ref.pt"))
        ai_ness_out = str(base_out.with_name(base_out.stem + "_ai_ness.pt"))

        self_prompts, other_prompts = build_self_reference_pairs(
            args.self_name, args.other_name
        )
        _run_one("self_ref", self_ref_out, self_prompts, other_prompts)

        ai_prompts, human_prompts = build_ai_ness_pairs()
        _run_one("ai_ness", ai_ness_out, ai_prompts, human_prompts)
    elif args.vector_kind == "self_ref":
        self_prompts, other_prompts = build_self_reference_pairs(
            args.self_name, args.other_name
        )
        _run_one("self_ref", str(base_out), self_prompts, other_prompts)
    elif args.vector_kind == "self_ref_referent":
        self_prompts, other_prompts = build_self_reference_referent_pairs(
            args.other_name
        )
        _run_one("self_ref_referent", str(base_out), self_prompts, other_prompts)
    elif args.vector_kind == "self_stakes":
        benefit_prompts, cost_prompts = build_self_stakes_pairs()
        _run_one("self_stakes", str(base_out), benefit_prompts, cost_prompts)
    elif args.vector_kind == "self_interest":
        high_prompts, low_prompts = build_self_interest_pairs()
        _run_one("self_interest", str(base_out), high_prompts, low_prompts)
    elif args.vector_kind == "pure_self_ref":
        self_prompts, other_prompts = build_pure_self_reference_pairs()
        _run_one("pure_self_ref", str(base_out), self_prompts, other_prompts)
    elif args.vector_kind == "legacy_identity":
        self_prompts, other_prompts = build_legacy_identity_pairs()
        _run_one("legacy_identity", str(base_out), self_prompts, other_prompts)
    elif args.vector_kind == "ai_human":
        ai_prompts, human_prompts = build_ai_human_pairs()
        _run_one("ai_human", str(base_out), ai_prompts, human_prompts)
    else:
        ai_prompts, human_prompts = build_ai_ness_pairs()
        _run_one("ai_ness", str(base_out), ai_prompts, human_prompts)


if __name__ == "__main__":
    main()
