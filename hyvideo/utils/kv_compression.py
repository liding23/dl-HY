import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _normalize_method(method: str) -> str:
    if method is None:
        return "none"
    return method.lower().replace("-", "_")


def _minmax_norm(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    x_min = x.min()
    x_max = x.max()
    denom = (x_max - x_min).clamp_min(1e-6)
    return (x - x_min) / denom


def _query_affinity_score(key: torch.Tensor, query: Optional[torch.Tensor]) -> torch.Tensor:
    # key/query: [B, H, S, D]
    if query is None or query.numel() == 0:
        return key.new_zeros((key.shape[2],), dtype=torch.float32)

    # Use a tiny query summary to avoid materializing full QK attention scores.
    q_slice = query[:, :, -min(query.shape[2], 8) :, :].float()
    q_anchor = q_slice.mean(dim=(0, 1, 2))  # [D]
    score = torch.einsum("bhsd,d->bhs", key.float(), q_anchor).abs().mean(dim=(0, 1))
    return score.to(dtype=torch.float32)


def _norm_score(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    key_score = key.float().norm(dim=-1).mean(dim=(0, 1))
    value_score = value.float().norm(dim=-1).mean(dim=(0, 1))
    return 0.5 * key_score + 0.5 * value_score


def _h2o_indices(
    key: torch.Tensor,
    value: torch.Tensor,
    query: Optional[torch.Tensor],
    budget: int,
    recent_window: int,
) -> torch.Tensor:
    seq_len = key.shape[2]
    device = key.device
    recent = min(max(recent_window, 0), budget, seq_len)
    prefix_len = seq_len - recent
    heavy_budget = max(budget - recent, 0)

    recent_idx = (
        torch.arange(prefix_len, seq_len, device=device, dtype=torch.long)
        if recent > 0
        else torch.empty(0, device=device, dtype=torch.long)
    )
    if heavy_budget == 0 or prefix_len <= 0:
        return recent_idx

    score = _norm_score(key[:, :, :prefix_len], value[:, :, :prefix_len])
    score = score + _query_affinity_score(key[:, :, :prefix_len], query)
    topk = min(heavy_budget, prefix_len)
    heavy_idx = torch.topk(score, k=topk, dim=0, largest=True).indices.to(torch.long)

    keep = torch.cat([heavy_idx, recent_idx], dim=0)
    keep = torch.unique(keep, sorted=True)
    return keep


def _rocketkv_indices(
    key: torch.Tensor,
    value: torch.Tensor,
    query: Optional[torch.Tensor],
    budget: int,
    recent_window: int,
    pool_kernel: int,
    page_size: int,
) -> torch.Tensor:
    seq_len = key.shape[2]
    device = key.device
    recent = min(max(recent_window, 0), budget, seq_len)
    prefix_len = seq_len - recent
    global_budget = max(budget - recent, 0)

    recent_idx = (
        torch.arange(prefix_len, seq_len, device=device, dtype=torch.long)
        if recent > 0
        else torch.empty(0, device=device, dtype=torch.long)
    )
    if global_budget == 0 or prefix_len <= 0:
        return recent_idx

    # Stage 1 (coarse): pooled importance on prefix KV.
    base_score = _norm_score(key[:, :, :prefix_len], value[:, :, :prefix_len])
    kernel = max(1, min(pool_kernel, prefix_len))
    if kernel % 2 == 0:
        kernel = max(1, kernel - 1)
    pooled = F.avg_pool1d(
        base_score[None, None, :], kernel_size=kernel, stride=1, padding=kernel // 2
    ).squeeze(0).squeeze(0)

    if page_size > 1 and prefix_len > page_size:
        pad_len = (page_size - (prefix_len % page_size)) % page_size
        if pad_len > 0:
            pooled_pad = F.pad(pooled, (0, pad_len), value=float("-inf"))
        else:
            pooled_pad = pooled
        pooled_page = pooled_pad.view(-1, page_size)
        page_score = pooled_page.max(dim=1).values.repeat_interleave(page_size)
        page_score = page_score[:prefix_len]
    else:
        page_score = pooled

    stage1_score = 0.5 * pooled + 0.5 * page_score
    stage1_budget = min(prefix_len, max(global_budget, int(math.ceil(global_budget * 1.5))))
    stage1_idx = torch.topk(stage1_score, k=stage1_budget, dim=0, largest=True).indices

    # Stage 2 (fine): query-aware refinement over stage-1 candidates.
    candidate_key = key[:, :, :prefix_len].index_select(2, stage1_idx)
    candidate_score = stage1_score.index_select(0, stage1_idx)
    candidate_score = candidate_score + _query_affinity_score(candidate_key, query)
    stage2_budget = min(global_budget, candidate_score.shape[0])
    final_local = torch.topk(candidate_score, k=stage2_budget, dim=0, largest=True).indices
    global_idx = stage1_idx.index_select(0, final_local).to(torch.long)

    keep = torch.cat([global_idx, recent_idx], dim=0)
    keep = torch.unique(keep, sorted=True)
    return keep


def _infinipot_v_indices(
    key: torch.Tensor,
    value: torch.Tensor,
    budget: int,
    recent_window: int,
    alpha: float,
) -> torch.Tensor:
    seq_len = key.shape[2]
    device = key.device

    # Keep a small recent tail for stream continuity.
    recent = min(max(recent_window, 0), budget // 2, seq_len)
    prefix_len = seq_len - recent
    dynamic_budget = max(budget - recent, 0)
    recent_idx = (
        torch.arange(prefix_len, seq_len, device=device, dtype=torch.long)
        if recent > 0
        else torch.empty(0, device=device, dtype=torch.long)
    )
    if dynamic_budget == 0 or prefix_len <= 0:
        return recent_idx

    # TaR proxy: novelty from neighboring token cosine dissimilarity.
    k_mean = key[:, :, :prefix_len].float().mean(dim=(0, 1))
    k_mean = F.normalize(k_mean, dim=-1, eps=1e-6)
    tar = torch.zeros(prefix_len, device=device, dtype=torch.float32)
    if prefix_len > 1:
        sim = (k_mean[1:] * k_mean[:-1]).sum(dim=-1).clamp(-1.0, 1.0)
        tar[1:] = 1.0 - sim
        tar[0] = tar[1]
    tar = _minmax_norm(tar)

    # VaN proxy: value norm importance.
    van = value[:, :, :prefix_len].float().norm(dim=-1).mean(dim=(0, 1))
    van = _minmax_norm(van)

    score = alpha * tar + (1.0 - alpha) * van

    # Preserve the earliest token as a stable anchor whenever possible.
    anchor = torch.tensor([0], device=device, dtype=torch.long)
    keep_dynamic = max(dynamic_budget - 1, 0)
    if keep_dynamic > 0:
        top_idx = torch.topk(score, k=min(keep_dynamic, prefix_len), dim=0).indices
        dynamic_idx = torch.cat([anchor, top_idx], dim=0)
    else:
        dynamic_idx = anchor

    keep = torch.cat([dynamic_idx, recent_idx], dim=0)
    keep = torch.unique(keep, sorted=True)
    if keep.numel() > budget:
        # If we overflow due to uniqueness merge, keep chronological tail first.
        keep = keep[-budget:]
    return keep


def compress_vision_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    query: Optional[torch.Tensor],
    method: str,
    max_tokens: int,
    recent_window: int = 1024,
    rocket_pool_kernel: int = 31,
    rocket_page_size: int = 64,
    infinipot_alpha: float = 0.6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compress [B, H, S, D] vision KV cache along the sequence axis (S).
    """
    method = _normalize_method(method)
    seq_len = key.shape[2]
    if method == "none" or max_tokens is None or max_tokens <= 0 or seq_len <= max_tokens:
        return key, value

    budget = min(max_tokens, seq_len)
    if method == "h2o":
        keep_idx = _h2o_indices(key, value, query, budget, recent_window)
    elif method == "rocketkv":
        keep_idx = _rocketkv_indices(
            key,
            value,
            query,
            budget,
            recent_window,
            pool_kernel=rocket_pool_kernel,
            page_size=rocket_page_size,
        )
    elif method == "infinipot_v":
        keep_idx = _infinipot_v_indices(
            key,
            value,
            budget,
            recent_window=recent_window,
            alpha=float(max(0.0, min(1.0, infinipot_alpha))),
        )
    else:
        raise ValueError(
            f"Unsupported kv compression method: {method}. "
            "Expected one of ['none', 'h2o', 'rocketkv', 'infinipot_v']."
        )

    if keep_idx.numel() == 0:
        # Fallback to at least one token to avoid empty-KV attention.
        keep_idx = torch.tensor([seq_len - 1], device=key.device, dtype=torch.long)

    key = key.index_select(dim=2, index=keep_idx)
    value = value.index_select(dim=2, index=keep_idx)
    return key, value
