"""General utilities with no ML dependencies."""

import hashlib
import json
import logging
import random
import subprocess
import sys
from typing import Dict, Optional

import numpy as np
import torch


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the package logger."""
    logger = logging.getLogger("motivated_scrutiny")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# Global logger instance
log = setup_logging()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_hash() -> str:
    """Get short git commit hash, or 'unknown' if unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def get_model_info(model) -> Dict[str, str]:
    """Extract model revision and tokenizer hash for reproducibility."""
    info = {"model_revision": "unknown", "tokenizer_hash": "unknown"}

    try:
        # Try to get model revision from config
        if hasattr(model, "model") and hasattr(model.model, "config"):
            config = model.model.config
            if hasattr(config, "_commit_hash"):
                info["model_revision"] = config._commit_hash[:12]
            elif hasattr(config, "transformers_version"):
                info["model_revision"] = f"tf-{config.transformers_version}"
    except Exception:
        pass

    try:
        # Hash tokenizer vocab for reproducibility
        if hasattr(model, "tokenizer"):
            vocab = str(sorted(model.tokenizer.get_vocab().items())[:100])
            info["tokenizer_hash"] = hashlib.md5(vocab.encode()).hexdigest()[:12]
    except Exception:
        pass

    return info


def extract_json_dict(text: str) -> Optional[Dict]:
    """Extract JSON dict from text, handling markdown code blocks."""
    if not text:
        return None
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def parse_response(full_output: str) -> Dict[str, str]:
    """Parse model output into {thinking, response, full} dict."""
    result = {"thinking": "", "response": "", "full": full_output}

    if "<think>" in full_output and "</think>" in full_output:
        think_start = full_output.find("<think>") + len("<think>")
        think_end = full_output.find("</think>")
        result["thinking"] = full_output[think_start:think_end].strip()

    if "</think>" in full_output:
        result["response"] = full_output.split("</think>")[-1].strip()
    else:
        result["response"] = full_output.strip()

    for token in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        result["response"] = result["response"].replace(token, "").strip()
        result["thinking"] = result["thinking"].replace(token, "").strip()

    return result
