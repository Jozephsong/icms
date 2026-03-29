"""
InferenceX-style workload generator.

All requests in a batch share the same `prefix_len` tokens (the "shared
context"), then each request appends `suffix_len` tokens that are unique to
that request.  This mirrors InferenceX's sample_random_requests pattern and
maximises KV cache reuse between requests:

  request[i]  =  shared_prefix  +  unique_suffix[i]
                 <-- prefix_len -->  <-- suffix_len -->

Using TokensPrompt lets us drive vLLM offline inference without a tokenizer.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from vllm import TokensPrompt


VOCAB_SIZE = 32_000   # safe default; actual vocab is not used for token IDs


@dataclass
class WorkloadBatch:
    prompts: List[TokensPrompt]
    num_requests: int
    prefix_len: int
    suffix_len: int

    @property
    def total_len(self) -> int:
        return self.prefix_len + self.suffix_len


def make_batch(
    num_requests: int,
    prefix_len: int,
    suffix_len: int,
    seed: int = 0,
) -> WorkloadBatch:
    """
    Build a WorkloadBatch with a shared prefix and per-request unique suffixes.

    The shared prefix is drawn once with the given seed; each request's suffix
    is drawn with a deterministic but different seed so results are reproducible
    across runs.

    Args:
        num_requests:  How many distinct prompts to produce.
        prefix_len:    Number of shared prefix tokens (all requests identical).
        suffix_len:    Number of unique suffix tokens per request.
        seed:          Base RNG seed. Prefix uses seed; suffix[i] uses seed+i+1.
    """
    rng = np.random.default_rng(seed)
    prefix = rng.integers(1, VOCAB_SIZE, size=prefix_len).tolist()

    prompts = []
    for i in range(num_requests):
        suf_rng = np.random.default_rng(seed + i + 1)
        suffix = suf_rng.integers(1, VOCAB_SIZE, size=suffix_len).tolist()
        prompts.append(TokensPrompt(prompt_token_ids=prefix + suffix))

    return WorkloadBatch(
        prompts=prompts,
        num_requests=num_requests,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
    )


def make_warmup_batch(prefix_len: int, suffix_len: int) -> WorkloadBatch:
    """
    A single warmup prompt that uses tokens outside the benchmark range
    so it never collides with actual benchmark prompts.
    """
    total = prefix_len + suffix_len
    # Use tokens from the high end of the vocabulary
    tokens = [VOCAB_SIZE - 1 - (i % (VOCAB_SIZE // 4)) for i in range(total)]
    return WorkloadBatch(
        prompts=[TokensPrompt(prompt_token_ids=tokens)],
        num_requests=1,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
    )
