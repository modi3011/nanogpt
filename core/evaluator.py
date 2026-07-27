"""
Shared evaluation utilities: load a trained model from a checkpoint, compute
perplexity over text, and generate text from a prompt.

Consolidates logic from the original eval.py / eval2.py / eval_task2.py /
sample.py / sample_batch.py scripts.
"""

import json
import math
import os
from typing import List, Optional, Tuple

import tiktoken
import torch

from core.architectures import GPT, GPTConfig
from core.tokenizer import encode, decode
from core.utils import load_checkpoint


def load_model_from_checkpoint(out_dir: str, device: str = "cpu") -> GPT:
    """Load a trained GPT model from `<out_dir>/ckpt.pt`."""
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    print(f"Loading model from {ckpt_path}...")
    checkpoint = load_checkpoint(ckpt_path, map_location=device)

    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)
    return model


def load_pretrained_gpt2(model_type: str = "gpt2") -> GPT:
    """Load an OpenAI GPT-2 checkpoint (for baseline comparisons)."""
    model = GPT.from_pretrained(model_type, dict(dropout=0.0))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Perplexity computation
# ---------------------------------------------------------------------------

def compute_loss_over_tokens(
    model: GPT, tokens: List[int], block_size: int, device: str
) -> Tuple[float, int]:
    """Compute summed negative log-likelihood and token count over a token sequence,
    sliding through it in non-overlapping `block_size` chunks.

    Returns:
        (total_nll, total_tokens) where total_nll = sum over chunks of (mean_chunk_loss * chunk_len)
    """
    total_nll = 0.0
    total_tokens = 0
    pos = 0

    while pos < len(tokens) - 1:
        inp = tokens[pos:pos + block_size]
        tgt = tokens[pos + 1:pos + 1 + block_size]

        if len(tgt) == 0:
            break
        if len(inp) != len(tgt):
            inp = inp[:len(tgt)]

        x = torch.tensor(inp, dtype=torch.long, device=device)[None, :]
        y = torch.tensor(tgt, dtype=torch.long, device=device)[None, :]

        with torch.no_grad():
            _, loss = model(x, y)

        n = len(tgt)
        total_nll += loss.item() * n
        total_tokens += n
        pos += n

    return total_nll, total_tokens


def compute_perplexity(
    model: GPT, text: str, enc: tiktoken.Encoding, block_size: int, device: str
) -> Tuple[float, float]:
    """Compute (average_loss, perplexity) for a single text string."""
    tokens = encode(text, enc)
    total_nll, total_tokens = compute_loss_over_tokens(model, tokens, block_size, device)
    if total_tokens == 0:
        raise ValueError("No valid tokens to evaluate. Check your input text.")
    avg_loss = total_nll / total_tokens
    return avg_loss, math.exp(avg_loss)


def compute_perplexity_over_paragraphs(
    model: GPT, paragraphs: List[str], enc: tiktoken.Encoding, block_size: int, device: str
) -> dict:
    """Compute perplexity averaged over a list of paragraphs/documents.

    Returns a dict with keys: avg_loss, ppl, used_paragraphs, skipped_short, pred_tokens.
    """
    total_nll = 0.0
    total_tokens = 0
    used = 0
    skipped = 0

    for para in paragraphs:
        tokens = encode(para, enc)
        if len(tokens) < 2:
            skipped += 1
            continue
        nll, n_tok = compute_loss_over_tokens(model, tokens, block_size, device)
        total_nll += nll
        total_tokens += n_tok
        used += 1

    if total_tokens == 0:
        raise ValueError("No valid tokens to evaluate. Check your input text.")

    avg_loss = total_nll / total_tokens
    return {
        "avg_loss": avg_loss,
        "ppl": math.exp(avg_loss),
        "used_paragraphs": used,
        "skipped_short": skipped,
        "pred_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate_text(
    model: GPT,
    prompt: str,
    enc: tiktoken.Encoding,
    device: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 200,
    eot_token: Optional[int] = None,
    stop_at_second_eot: bool = False,
) -> str:
    """Generate text continuing from `prompt`."""
    tokens = encode(prompt, enc)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eot_token=eot_token,
        stop_at_second_eot=stop_at_second_eot,
    )
    generated = decode(out[0].tolist(), enc)
    generated.replace("<|endoftext|>", "").strip()
    return generated


# ---------------------------------------------------------------------------
# Paragraph file loading (txt / jsonl / json)
# ---------------------------------------------------------------------------

def _read_txt_paragraphs(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return [p.strip() for p in content.split("\n\n") if p.strip()]


def _read_jsonl_paragraphs(path: str, text_key: str) -> List[str]:
    paragraphs = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                text = obj
            elif isinstance(obj, dict):
                if text_key not in obj:
                    raise KeyError(f"Missing key '{text_key}' in JSONL line {ln}")
                text = obj[text_key]
            else:
                raise TypeError(f"Unsupported JSONL value type on line {ln}: {type(obj)}")
            text = text.strip()
            if text:
                paragraphs.append(text)
    return paragraphs


def _read_json_paragraphs(path: str, text_key: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("JSON input must be a list of strings or objects")
    paragraphs = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            if text_key not in item:
                raise KeyError(f"Missing key '{text_key}' in JSON item index {i}")
            text = item[text_key]
        else:
            raise TypeError(f"Unsupported JSON item type at index {i}: {type(item)}")
        text = text.strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def load_paragraphs(path: str, fmt: str = "auto", text_key: str = "text") -> Tuple[List[str], str]:
    """Load paragraphs/documents from a .txt, .jsonl, or .json file.

    - .txt: paragraphs separated by one or more blank lines
    - .jsonl: one JSON object (or string) per line
    - .json: a list of strings or objects

    Returns:
        (paragraphs, resolved_format)
    """
    if fmt == "auto":
        ext = os.path.splitext(path)[1].lower()
        fmt = {"txt": "txt", "jsonl": "jsonl", "json": "json"}.get(ext.lstrip("."), "txt")

    if fmt == "txt":
        return _read_txt_paragraphs(path), "txt"
    if fmt == "jsonl":
        return _read_jsonl_paragraphs(path, text_key), "jsonl"
    if fmt == "json":
        return _read_json_paragraphs(path, text_key), "json"
    raise ValueError(f"Unsupported input_format: {fmt}")
