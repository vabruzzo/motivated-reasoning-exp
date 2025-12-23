"""Activation steering with selective token targeting."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .nnsight_utils import get_layers
from .utils import log


# ============================================================================
# VECTOR I/O
# ============================================================================


def load_vectors(vectors_path: str) -> Tuple[Dict[int, torch.Tensor], Dict]:
    """Load steering vectors and metadata from .pt/.json files."""
    path = Path(vectors_path)
    if not path.exists():
        raise FileNotFoundError(f"Vectors not found: {path}")

    vectors = torch.load(path, weights_only=True)

    metadata = {}
    metadata_path = path.with_suffix(".json")
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        log.info(f"Loaded vectors from {path}")
        log.info(f"  Created: {metadata.get('created_at', 'unknown')}")
        log.info(f"  Prompts hash: {metadata.get('prompts_hash', 'unknown')}")
    else:
        log.info(f"Loaded vectors from {path} (no metadata)")

    return vectors, metadata


def save_vectors(
    vectors_data: Dict,
    output_path: str,
    config: Dict,
    self_prompts: List[str],
    other_prompts: List[str],
) -> Dict:
    """Save vectors (.pt) and metadata (.json). Returns metadata dict."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(vectors_data["vectors"], path)
    log.info(f"Saved vectors to: {path}")

    metadata = {
        "created_at": datetime.now().isoformat(),
        "config": config,
        "self_prompts": self_prompts,
        "other_prompts": other_prompts,
        "num_layers": len(vectors_data["vectors"]),
        "layers": list(vectors_data["vectors"].keys()),
        "norms": vectors_data["norms"],
        "raw_norms": vectors_data.get("raw_norms"),
        "cosine_distances": vectors_data["cosine_distances"],
        "hidden_size": list(vectors_data["vectors"].values())[0].shape[0],
        "prompts_hash": hashlib.md5(
            (str(self_prompts) + str(other_prompts)).encode()
        ).hexdigest()[:12],
    }

    metadata_path = path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Saved metadata to: {metadata_path}")

    return metadata


def create_random_vectors(
    reference_vectors: Dict[int, torch.Tensor],
    seed: int = 42,
    orthogonal_to_reference: bool = False,
) -> Dict[int, torch.Tensor]:
    """
    Create random vectors with matched norms (for control condition).

    If orthogonal_to_reference=True, each random vector is projected to be
    orthogonal to the corresponding reference vector before rescaling. This
    reduces the chance that the "random" control accidentally aligns with the
    true steering direction.
    """
    import random
    import numpy as np

    # Seed all RNGs for full determinism
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    result = {}
    for layer, ref in reference_vectors.items():
        rand = torch.randn_like(ref)

        if orthogonal_to_reference:
            denom = (ref @ ref).clamp_min(1e-12)
            rand = rand - ((rand @ ref) / denom) * ref
            # Extremely unlikely, but guard against near-zero after projection
            if rand.norm() < 1e-8:
                rand = torch.randn_like(ref)
                rand = rand - ((rand @ ref) / denom) * ref

        result[layer] = rand * (ref.norm() / rand.norm().clamp_min(1e-12))
    return result


# ============================================================================
# TOKEN MASKING
# ============================================================================


def create_user_token_mask(
    user_prompt: str, input_ids: torch.Tensor, tokenizer
) -> Optional[torch.Tensor]:
    """Create bool mask for user tokens. Returns None if detection fails."""
    if input_ids.dim() == 2:
        input_ids = input_ids[0]

    seq_len = input_ids.shape[0]
    device = input_ids.device

    user_tokens = tokenizer.encode(user_prompt, add_special_tokens=False)
    full_tokens = input_ids.tolist()

    if not user_tokens:
        log.warning("Empty user tokens, skipping mask")
        return None

    # Find occurrences
    matches = [
        (j, j + len(user_tokens))
        for j in range(len(full_tokens) - len(user_tokens) + 1)
        if full_tokens[j : j + len(user_tokens)] == user_tokens
    ]

    if not matches:
        log.warning("Could not find user tokens, skipping mask")
        return None
    if len(matches) > 1:
        log.warning(f"Found {len(matches)} matches, using first")

    user_start, user_end = matches[0]
    if user_start > seq_len * 0.7:
        log.warning(f"User tokens found late ({user_start}/{seq_len}), skipping mask")
        return None

    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[user_start:user_end] = True
    return mask


def get_think_token_ids(tokenizer) -> Tuple[List[int], List[int]]:
    """Get token IDs for <think> and </think> tags."""
    return (
        tokenizer.encode("<think>", add_special_tokens=False),
        tokenizer.encode("</think>", add_special_tokens=False),
    )


# ============================================================================
# SELECTIVE STEERING HOOK
# ============================================================================


class SelectiveSteeringHook:
    """
    Hook manager for selective steering during generation.

    Steers user tokens (prefill) and/or thinking tokens (generation).
    Use register_hooks() before generation, remove_hooks() after.
    """

    def __init__(
        self,
        model,
        steering_vectors: Dict[int, torch.Tensor],
        steering_layers: List[int],
        alpha: float,
        direction: int,
        steer_on_user: bool,
        steer_on_thinking: bool,
        user_mask: Optional[torch.Tensor],
        think_start_ids: List[int],
        think_end_ids: List[int],
        tokenizer,
    ):
        self.model = model
        self.alpha = alpha
        self.direction = direction
        self.steer_on_user = steer_on_user
        self.steer_on_thinking = steer_on_thinking
        self.user_mask = user_mask
        self.think_start_ids = think_start_ids
        self.think_end_ids = think_end_ids
        self.tokenizer = tokenizer

        self.is_prefill = True
        self.in_thinking = False
        self.recent_tokens: List[int] = []
        self.hooks = []

        # Pre-compute scaled vectors with missing layer warnings
        self.scaled_vectors = {}
        missing_layers = []
        for layer in steering_layers:
            if layer in steering_vectors:
                vec = steering_vectors[layer].to(model.device).to(torch.bfloat16)
                self.scaled_vectors[layer] = direction * alpha * vec
            else:
                missing_layers.append(layer)

        if missing_layers:
            log.warning(
                f"Steering layers {missing_layers} not found in vectors, skipping"
            )
        if not self.scaled_vectors:
            raise ValueError(
                f"No valid steering vectors found. Requested layers: {steering_layers}, "
                f"available: {list(steering_vectors.keys())}"
            )

    def _create_hook(self, layer_idx: int):
        """Create forward hook for a layer."""
        vec = self.scaled_vectors.get(layer_idx)
        if vec is None:
            return None

        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            rest = output[1:] if isinstance(output, tuple) else ()

            # Move vec to same device as hidden (for multi-GPU)
            vec_local = vec.to(hidden.device)

            if hidden.dim() == 3:  # (batch, seq, hidden) - Prefill
                seq_len = hidden.shape[1]
                if (
                    self.is_prefill
                    and self.steer_on_user
                    and self.user_mask is not None
                ):
                    mask_len = self.user_mask.shape[0]
                    if seq_len >= mask_len:
                        padded_mask = torch.zeros(
                            seq_len, dtype=torch.bool, device=hidden.device
                        )
                        padded_mask[:mask_len] = self.user_mask.to(hidden.device)
                        hidden[:] = (
                            hidden
                            + vec_local * padded_mask.unsqueeze(0).unsqueeze(-1).float()
                        )
                    # Done with prefill regardless of whether masking applied this step
                    self.is_prefill = False
                elif self.is_prefill:
                    # If we couldn't build a user mask, still advance out of prefill so
                    # thinking-time steering can activate during generation.
                    self.is_prefill = False
                elif (
                    not self.is_prefill and self.steer_on_thinking and self.in_thinking
                ):
                    hidden[:, -1, :] = hidden[:, -1, :] + vec_local

            elif hidden.dim() == 2:  # (batch, hidden) - Single token with KV cache
                self.is_prefill = False
                if self.steer_on_thinking and self.in_thinking:
                    hidden[:] = hidden + vec_local

            return (hidden,) + rest if rest else hidden

        return hook

    def register_hooks(self):
        """Register hooks on steering layers."""
        layers = get_layers(self.model)
        for layer_idx in self.scaled_vectors:
            hook_fn = self._create_hook(layer_idx)
            if hook_fn:
                self.hooks.append(layers[layer_idx].register_forward_hook(hook_fn))

    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def update_thinking_state(self, new_token_id: int):
        """Update <think> state tracker with new token."""
        self.recent_tokens.append(new_token_id)
        max_len = max(len(self.think_start_ids), len(self.think_end_ids))
        self.recent_tokens = self.recent_tokens[-max_len:]

        if len(self.recent_tokens) >= len(self.think_start_ids):
            if self.recent_tokens[-len(self.think_start_ids) :] == self.think_start_ids:
                self.in_thinking = True
        if len(self.recent_tokens) >= len(self.think_end_ids):
            if self.recent_tokens[-len(self.think_end_ids) :] == self.think_end_ids:
                self.in_thinking = False

    def reset_state(self):
        """Reset for new generation."""
        self.is_prefill = True
        self.in_thinking = False
        self.recent_tokens = []


# ============================================================================
# GENERATION
# ============================================================================


def generate_baseline(model, tokenizer, prompt_text: str, config: Dict) -> str:
    """Generate without steering. Returns generated text (excluding prompt)."""
    with model.generate(
        prompt_text,
        max_new_tokens=config["max_new_tokens"],
        do_sample=True,
        temperature=config["temperature"],
        top_p=config["top_p"],
    ) as gen:
        output_ids = model.generator.output.save()

    result = output_ids.value if hasattr(output_ids, "value") else output_ids
    prompt_ids = tokenizer.encode(prompt_text)
    return tokenizer.decode(result[0, len(prompt_ids) :], skip_special_tokens=False)


def generate_with_steering(
    model,
    tokenizer,
    prompt_text: str,
    user_content: str,
    steering_vectors: Dict[int, torch.Tensor],
    steering_layers: List[int],
    config: Dict,
    direction: int,
) -> str:
    """Generate with steering. direction: +1 to add, -1 to subtract vector."""
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
    prompt_length = input_ids.shape[1]

    user_mask = None
    if config.get("steer_on_user", True):
        user_mask = create_user_token_mask(user_content, input_ids, tokenizer)
        if user_mask is not None:
            user_mask = user_mask.to(model.device)

    think_start_ids, think_end_ids = get_think_token_ids(tokenizer)

    hook_manager = SelectiveSteeringHook(
        model=model,
        steering_vectors=steering_vectors,
        steering_layers=steering_layers,
        alpha=config.get("steering_alpha", 0.1),
        direction=direction,
        steer_on_user=config.get("steer_on_user", True),
        steer_on_thinking=config.get("steer_on_thinking", True),
        user_mask=user_mask,
        think_start_ids=think_start_ids,
        think_end_ids=think_end_ids,
        tokenizer=tokenizer,
    )

    try:
        hook_manager.register_hooks()

        if config.get("steer_on_thinking", True):
            generated_ids = _generate_with_thinking_steering(
                model, tokenizer, input_ids, hook_manager, config
            )
        else:
            with model.generate(
                input_ids,
                max_new_tokens=config["max_new_tokens"],
                do_sample=True,
                temperature=config["temperature"],
                top_p=config["top_p"],
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            ) as gen:
                output_ids = model.generator.output.save()
            generated_ids = (
                output_ids.value if hasattr(output_ids, "value") else output_ids
            )
    finally:
        hook_manager.remove_hooks()

    return tokenizer.decode(generated_ids[0, prompt_length:], skip_special_tokens=False)


def _get_eos_token_ids(tokenizer) -> List[int]:
    """Get all EOS-like token IDs for stopping generation."""
    eos_ids = []

    # Standard EOS
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id not in eos_ids:
        eos_ids.append(tokenizer.pad_token_id)

    # ChatML/Qwen im_end token
    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id != tokenizer.unk_token_id and im_end_id not in eos_ids:
            eos_ids.append(im_end_id)
    except Exception:
        pass

    return eos_ids


def _generate_with_thinking_steering(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    hook_manager: SelectiveSteeringHook,
    config: Dict,
) -> torch.Tensor:
    """Token-by-token generation with <think> state tracking and KV cache."""
    max_new_tokens = config.get("max_new_tokens", 0)
    if max_new_tokens <= 0:
        return input_ids

    generated_ids = input_ids.clone()
    past_key_values = None
    model_inputs = generated_ids
    hf_model = model.model

    # Get lm_head - try multiple paths for nnsight compatibility
    try:
        # Direct attribute access (nnsight proxies this)
        lm_head = hf_model.lm_head
    except AttributeError:
        # Fallback: try accessing underlying model
        if hasattr(model, "_model"):
            lm_head = model._model.lm_head
        else:
            raise AttributeError(
                f"Could not find lm_head. Model type: {type(model)}, hf_model type: {type(hf_model)}"
            )

    # Pre-compute EOS tokens
    eos_ids = _get_eos_token_ids(tokenizer)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = hf_model(
                model_inputs, use_cache=True, past_key_values=past_key_values
            )

        past_key_values = outputs.past_key_values
        last_hidden = outputs[0][:, -1, :]
        next_logits = lm_head(last_hidden) / config["temperature"]

        # Top-p sampling
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > config["top_p"]
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        if sorted_indices[sorted_indices_to_remove].numel() > 0:
            mask = torch.zeros_like(next_logits, dtype=torch.bool)
            mask.scatter_(1, sorted_indices, sorted_indices_to_remove)
            next_logits[mask] = float("-inf")

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        hook_manager.update_thinking_state(next_token.item())
        # Move to same device as generated_ids (for multi-GPU)
        next_token = next_token.to(generated_ids.device)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        model_inputs = next_token

        if next_token.item() in eos_ids:
            break

    return generated_ids
