"""NNsight and transformer model utilities."""

from typing import Optional, Any, Tuple, List

import torch
from nnsight import LanguageModel

from .utils import log

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def load_model(
    model_name: str, device_map: str = "auto", dtype: torch.dtype = torch.bfloat16
) -> LanguageModel:
    """Load model with nnsight wrapper."""
    log.info(f"Loading {model_name}...")
    model = LanguageModel(model_name, device_map=device_map, dtype=dtype, dispatch=True)
    model.model.eval()

    num_layers = len(get_layers(model))
    hidden_size = model.model.config.hidden_size
    log.info(f"Model loaded. Layers: {num_layers}, Hidden: {hidden_size}")

    return model


def get_tokenizer(model: LanguageModel):
    """Get tokenizer from nnsight model."""
    return model.tokenizer


def get_layers(model) -> list:
    """Get transformer layers list. Handles nnsight wrapper and various architectures."""
    # Try paths in order of likelihood for nnsight-wrapped models
    paths = [
        lambda m: m.model.model.layers,  # nnsight + Qwen/Llama (most common)
        lambda m: m.model.layers,  # nnsight + other architectures
        lambda m: m.layers,  # direct access
    ]

    for path in paths:
        try:
            layers = path(model)
            # Verify it's indexable with multiple elements (not a single module)
            if hasattr(layers, "__len__") and hasattr(layers, "__getitem__"):
                if len(layers) > 1:
                    return layers
        except (AttributeError, TypeError):
            continue

    raise ValueError(
        f"Could not find layers in {type(model)}. Model structure may not be supported."
    )


def format_prompt(
    user_content: str,
    tokenizer: Optional[Any] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """
    Format user content as chat prompt.

    If tokenizer provided, uses apply_chat_template (model-agnostic).
    Otherwise falls back to Qwen format.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Fallback: Qwen/ChatML format
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _find_subsequence(full: List[int], sub: List[int]) -> Optional[Tuple[int, int]]:
    """Return (start, end) of first occurrence of sub in full, else None."""
    if not sub or not full or len(sub) > len(full):
        return None
    for j in range(len(full) - len(sub) + 1):
        if full[j : j + len(sub)] == sub:
            return (j, j + len(sub))
    return None


def get_activation(
    model: LanguageModel,
    prompt: str,
    layer: int,
    token_pos: int = -2,
    format_as_chat: bool = True,
    tokenizer: Optional[Any] = None,
    system_prompt: Optional[str] = None,
    token_selector: str = "absolute",
    user_last_n: int = 5,
) -> torch.Tensor:
    """
    Extract activation vector from a layer.

    Token selection:
    - token_selector="absolute": use token_pos over the full formatted prompt
    - token_selector="user_last": use the last token of the user_content span (recommended)
    - token_selector="user_mean_last_n": average last N tokens of the user_content span
    - token_selector="user_mean": average ALL tokens of the user_content span
    """
    user_content = prompt  # keep original for user-span matching
    tok = tokenizer or get_tokenizer(model)

    if format_as_chat:
        sys_prompt = (
            system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        )
        prompt = format_prompt(user_content, tokenizer=tok, system_prompt=sys_prompt)

    with model.trace(prompt):
        hidden = model.model.layers[layer].output[0]
        act = hidden.save()

    result = act.value if hasattr(act, "value") else act

    # Normalize to (seq, hidden)
    if result.dim() == 3:
        if result.shape[0] != 1:
            raise ValueError(
                f"Expected batch_size=1 for activation extraction, got {result.shape[0]}"
            )
        result = result.squeeze(0)
    if result.dim() != 2:
        raise ValueError(f"Unexpected tensor dim: {result.dim()}")

    seq_len = result.shape[0]

    def _extract_by_pos(pos: int) -> torch.Tensor:
        clamped = max(-seq_len, min(pos, seq_len - 1))
        if clamped != pos:
            log.warning(f"token_pos {pos} clamped to {clamped} (seq_len={seq_len})")
        return result[clamped, :]

    if token_selector == "absolute":
        extracted = _extract_by_pos(token_pos)
    elif token_selector in ("user_last", "user_mean_last_n", "user_mean"):
        full_ids = tok.encode(prompt, add_special_tokens=False)
        user_ids = tok.encode(user_content, add_special_tokens=False)
        span = _find_subsequence(full_ids, user_ids)

        if span is None:
            log.warning(
                "Could not locate user_content inside formatted prompt tokens; "
                "falling back to absolute token_pos"
            )
            extracted = _extract_by_pos(token_pos)
        else:
            user_start, user_end = span

            # If tokenization length doesn't match activation length, clamp indices.
            if len(full_ids) != seq_len:
                log.warning(
                    f"Token/activation length mismatch (tokens={len(full_ids)} vs seq={seq_len}); "
                    "clamping user-span indices."
                )
                user_start = max(0, min(user_start, seq_len))
                user_end = max(0, min(user_end, seq_len))

            if user_end <= user_start:
                log.warning("Bad user span; falling back to absolute token_pos")
                extracted = _extract_by_pos(token_pos)
            elif token_selector == "user_last":
                extracted = result[user_end - 1, :]
            elif token_selector == "user_mean":
                extracted = result[user_start:user_end, :].mean(dim=0)
            else:
                n = max(1, int(user_last_n))
                start = max(user_start, user_end - n)
                extracted = result[start:user_end, :].mean(dim=0)
    else:
        raise ValueError(
            f"Unknown token_selector={token_selector!r}. "
            "Use one of: absolute, user_last, user_mean_last_n, user_mean."
        )

    return extracted.detach().cpu().float()


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text."""
    return len(tokenizer.encode(text))
