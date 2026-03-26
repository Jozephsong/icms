"""
ICMS Query Generator

Generates InferenceX-style input requests with a shared prefix and
per-request unique suffixes, maximizing KV cache sharing across requests.

Design mirrors InferenceX's sample_random_requests pattern adapted for
vLLM offline inference (TokensPrompt) without requiring a tokenizer.
"""

import random
from typing import List

from vllm import TokensPrompt


def make_shared_prefix_prompts(
    num_requests: int,
    prefix_len: int,
    unique_len: int,
    vocab_size: int = 32000,
    seed: int = 42,
) -> List[TokensPrompt]:
    """
    Generate prompts with a shared prefix and unique per-request suffixes.

    All requests share the same prefix_len tokens (maximizes KV cache reuse),
    then each request appends unique_len tokens specific to that request.

    Args:
        num_requests:  Number of prompts to generate.
        prefix_len:    Number of shared prefix tokens (same for all requests).
        unique_len:    Number of unique suffix tokens per request.
        vocab_size:    Token vocabulary size for generating token IDs.
        seed:          Random seed for reproducibility.

    Returns:
        List of TokensPrompt objects ready for vllm.LLM.generate().
    """
    rng = random.Random(seed)

    # Shared prefix — identical across all requests to trigger KV cache hits
    prefix_tokens = [rng.randint(1, vocab_size - 1) for _ in range(prefix_len)]

    prompts = []
    for i in range(num_requests):
        # Unique per-request suffix to differentiate requests
        # Use a deterministic but unique offset per request (InferenceX pattern)
        offset = (seed + i * 1000) % vocab_size
        suffix = [(offset + j) % vocab_size + 1 for j in range(unique_len)]
        prompts.append(TokensPrompt(prompt_token_ids=prefix_tokens + suffix))

    return prompts


def make_warmup_prompt(prefix_len: int, unique_len: int, vocab_size: int = 32000) -> List[TokensPrompt]:
    """Generate a single warmup prompt that won't collide with benchmark prompts."""
    # Use tokens outside the range used by make_shared_prefix_prompts
    tokens = [vocab_size - 1 - (i % (vocab_size // 2)) for i in range(prefix_len + unique_len)]
    return [TokensPrompt(prompt_token_ids=tokens)]
