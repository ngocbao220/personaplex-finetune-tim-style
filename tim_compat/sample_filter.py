"""Class-A validation of actual reference Samples, never reconstructed sequences."""

import json
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FilterPolicy:
    text_cardinality: int
    audio_cardinality: int
    nonlexical_text_ids: tuple[int, ...]
    zero_padding_id: int = -1
    require_agent_text: bool = True
    max_consecutive_rejections: int = 1000
    text_sentinel_ids: tuple[int, ...] = ()
    audio_sentinel_ids: tuple[int, ...] = ()

    def __post_init__(self):
        if self.text_cardinality <= 0 or self.audio_cardinality <= 0:
            raise ValueError("token cardinalities must be positive")
        if not self.nonlexical_text_ids:
            raise ValueError("explicit reference nonlexical text IDs required")
        if self.max_consecutive_rejections <= 0:
            raise ValueError("max_consecutive_rejections must be positive")


def token_bounds_error(tokens, cardinality, zero_padding_id, sentinel_ids=()):
    """Describe invalid IDs, allowing only explicitly declared runtime sentinels."""
    invalid = ((tokens < 0) | (tokens >= cardinality)) & (tokens != zero_padding_id)
    for token_id in sentinel_ids:
        invalid &= tokens != token_id
    if invalid.any().item():
        values = tokens[invalid]
        return dict(cardinality=cardinality, invalid_count=values.numel(),
                    invalid_min=int(values.min().item()), invalid_max=int(values.max().item()),
                    zero_padding_id=zero_padding_id, sentinel_ids=list(sentinel_ids))
    return None


def rejection_reason(sample, policy):
    codes = sample.codes
    if (not isinstance(codes, torch.Tensor) or codes.ndim != 3 or
            codes.shape[:2] != (1, 17) or codes.shape[-1] == 0 or
            codes.dtype not in (torch.int32, torch.int64)):
        return "invalid_stream_layout"
    frames = codes.shape[-1]
    prompt = sample.prompt_length
    if not isinstance(prompt, int) or not 0 <= prompt < frames:
        return "invalid_prompt_length"
    mask = sample.context_mask
    if mask is not None and (
            not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or
            mask.shape != (frames,) or mask.device != codes.device):
        return "invalid_context_mask"
    for tokens, cardinality, sentinels, reason in (
        (codes[:, 0], policy.text_cardinality, policy.text_sentinel_ids, "text_token_out_of_bounds"),
        (codes[:, 1:], policy.audio_cardinality, policy.audio_sentinel_ids, "audio_token_out_of_bounds"),
    ):
        if token_bounds_error(tokens, cardinality, policy.zero_padding_id, sentinels):
            return reason
    if policy.require_agent_text:
        text = codes[0, 0, prompt:]
        lexical = text != policy.zero_padding_id
        for token_id in policy.nonlexical_text_ids:
            lexical &= text != token_id
        if mask is not None:
            lexical &= ~mask[prompt:]
        if not lexical.any().item():
            return "no_retained_agent_text"
    return None


def filter_samples(samples, policy, *, enabled=True, report=None):
    """Return samples unchanged/order-preserved or report and skip invalid samples.

    Disabled mode returns the original iterator itself and never examines policy.
    Reference iterator exceptions propagate: tokenizer failures are not rejections.
    Reports contain iterator-local indices; they are not persistent chunk identities.
    """
    if not enabled:
        return samples
    if not isinstance(policy, FilterPolicy):
        raise TypeError("enabled filtering requires an explicit FilterPolicy")

    def retained():
        consecutive_rejections = 0
        for index, sample in enumerate(samples):
            reason = rejection_reason(sample, policy)
            if reason is None:
                consecutive_rejections = 0
                yield sample
            else:
                consecutive_rejections += 1
                if report is not None:
                    report.write(json.dumps({"candidate_index": index, "reason": reason,
                        "provenance": getattr(sample, 'provenance', None)}) + "\n")
                    report.flush()
                if consecutive_rejections >= policy.max_consecutive_rejections:
                    raise RuntimeError("sample filter reached maximum consecutive rejections")

    return retained()


@contextmanager
def sample_filter_loader(data_loader, policy, *, enabled=True, report=None):
    """Temporarily wrap the reference data_loader's imported build_dataset binding.

    Process-local and single-threaded, like the local-checkpoint bridge. Does not
    patch source files, tokenize again or catch reference exceptions.
    """
    if not enabled:
        yield
        return
    if not isinstance(policy, FilterPolicy):
        raise TypeError("enabled filtering requires an explicit FilterPolicy")
    original = data_loader.build_dataset

    def filtered(*args, **kwargs):
        return filter_samples(original(*args, **kwargs), policy, report=report)

    data_loader.build_dataset = filtered
    try:
        yield
    finally:
        data_loader.build_dataset = original
