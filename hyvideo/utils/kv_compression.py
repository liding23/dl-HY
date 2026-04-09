import math
from typing import Any, Dict, Optional, Tuple

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


def _gather_by_head(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # x: [B, H, S, D], indices: [H, K]
    bsz, heads, _, dim = x.shape
    indices = indices.to(device=x.device, dtype=torch.long)
    gather_idx = indices.unsqueeze(0).unsqueeze(-1).expand(bsz, heads, indices.shape[1], dim)
    return torch.gather(x, dim=2, index=gather_idx)


def _attention_scores(
    query: Optional[torch.Tensor],
    key: torch.Tensor,
    observe_q: int = 8,
) -> torch.Tensor:
    # query/key: [B, H, S, D] -> returns [H, S]
    if query is None or query.numel() == 0 or key.numel() == 0:
        return key.new_zeros((key.shape[1], key.shape[2]), dtype=torch.float32)

    q_len = min(max(1, observe_q), query.shape[2])
    q_obs = query[:, :, -q_len:, :].float()
    k = key.float()
    logits = torch.einsum("bhqd,bhkd->bhqk", q_obs, k) / math.sqrt(max(1, key.shape[-1]))
    probs = torch.softmax(logits, dim=-1)
    return probs.sum(dim=(0, 2)).to(dtype=torch.float32)


def _build_page_minmax(
    key: torch.Tensor,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    # key: [B, H, S, D] -> min/max: [H, P, D]
    key_hsd = key.float().mean(dim=0)  # [H, S, D]
    heads, seq_len, dim = key_hsd.shape
    page_size = max(1, page_size)
    num_pages = math.ceil(seq_len / page_size)
    pad_len = num_pages * page_size - seq_len
    if pad_len > 0:
        pad_min = torch.full(
            (heads, pad_len, dim), float("inf"), device=key_hsd.device, dtype=key_hsd.dtype
        )
        pad_max = torch.full(
            (heads, pad_len, dim), float("-inf"), device=key_hsd.device, dtype=key_hsd.dtype
        )
        key_for_min = torch.cat([key_hsd, pad_min], dim=1)
        key_for_max = torch.cat([key_hsd, pad_max], dim=1)
    else:
        key_for_min = key_hsd
        key_for_max = key_hsd

    key_min = key_for_min.view(heads, num_pages, page_size, dim).amin(dim=2)
    key_max = key_for_max.view(heads, num_pages, page_size, dim).amax(dim=2)
    return key_min, key_max, seq_len


def _ensure_state(state: Optional[Dict[str, Any]], method: str) -> Dict[str, Any]:
    if state is None or state.get("method") != method:
        return {"method": method}
    return state


def _compress_h2o(
    key: torch.Tensor,
    value: torch.Tensor,
    query: Optional[torch.Tensor],
    budget: int,
    recent_window: int,
    state: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    seq_len = key.shape[2]
    heads = key.shape[1]
    scores = _attention_scores(query, key, observe_q=8)  # [H, S]

    prev = state.get("hh_score")
    if prev is None or prev.shape[0] != heads:
        hh_score = scores
    else:
        prev_len = prev.shape[1]
        cur_len = scores.shape[1]
        if cur_len == prev_len:
            hh_score = prev + scores
        elif cur_len > prev_len:
            hh_score = scores.clone()
            hh_score[:, :prev_len] += prev
        else:
            # New sequence is shorter than previous one: reset to current scores.
            hh_score = scores
    state["hh_score"] = hh_score

    if seq_len <= budget:
        state["last_debug"] = {
            "method": "h2o",
            "phase": "decode",
            "seq_in": int(seq_len),
            "seq_out": int(seq_len),
            "budget": int(budget),
            "heavy_budget": int(max(budget - min(max(recent_window, 0), budget, seq_len), 0)),
            "recent_budget": int(min(max(recent_window, 0), budget, seq_len)),
            "hh_score_mean": float(hh_score.mean().item()) if hh_score.numel() > 0 else 0.0,
        }
        return key, value, state

    recent = min(max(recent_window, 0), budget, seq_len)
    prefix_len = seq_len - recent
    heavy_budget = max(budget - recent, 0)

    if heavy_budget > 0 and prefix_len > 0:
        heavy_idx = torch.topk(hh_score[:, :prefix_len], k=min(heavy_budget, prefix_len), dim=-1).indices
        heavy_idx = torch.sort(heavy_idx, dim=-1).values
    else:
        heavy_idx = torch.empty((heads, 0), device=key.device, dtype=torch.long)

    if recent > 0:
        recent_idx_1d = torch.arange(prefix_len, seq_len, device=key.device, dtype=torch.long)
        recent_idx = recent_idx_1d.unsqueeze(0).expand(heads, -1)
    else:
        recent_idx = torch.empty((heads, 0), device=key.device, dtype=torch.long)

    keep_idx = torch.cat([heavy_idx, recent_idx], dim=-1)
    if keep_idx.shape[1] == 0:
        keep_idx = torch.full((heads, 1), seq_len - 1, device=key.device, dtype=torch.long)

    key = _gather_by_head(key, keep_idx)
    value = _gather_by_head(value, keep_idx)
    state["hh_score"] = torch.gather(hh_score, dim=1, index=keep_idx)
    state["last_debug"] = {
        "method": "h2o",
        "phase": "decode",
        "seq_in": int(seq_len),
        "seq_out": int(key.shape[2]),
        "budget": int(budget),
        "heavy_budget": int(heavy_budget),
        "recent_budget": int(recent),
        "hh_score_mean": float(state["hh_score"].mean().item())
        if state["hh_score"].numel() > 0
        else 0.0,
    }
    return key, value, state


def _rocket_stage1_prefill(
    key: torch.Tensor,
    value: torch.Tensor,
    query: Optional[torch.Tensor],
    budget: int,
    recent_window: int,
    pool_kernel: int,
    page_size: int,
    state: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    seq_len = key.shape[2]
    heads = key.shape[1]
    if seq_len <= budget:
        key_min, key_max, page_tokens = _build_page_minmax(key, page_size=page_size)
        state.update(
            {
                "prefill_done": True,
                "prefill_len": seq_len,
                "page_size": max(1, page_size),
                "page_min": key_min,
                "page_max": key_max,
                "page_tokens": page_tokens,
                "decode_steps": 0,
            }
        )
        state["last_debug"] = {
            "method": "rocketkv",
            "phase": "prefill",
            "seq_in": int(seq_len),
            "seq_out": int(seq_len),
            "budget": int(budget),
            "prompt_budget": int(seq_len),
            "recent_budget": 0,
            "page_size": int(max(1, page_size)),
            "pages_selected": int(key_min.shape[1]),
        }
        return key, value, state

    recent = min(max(recent_window, 0), budget, seq_len)
    prefix_len = seq_len - recent
    keep_prefix = max(budget - recent, 0)
    score = _attention_scores(query, key, observe_q=32)  # [H, S]

    if keep_prefix > 0 and prefix_len > 0:
        prefix_score = score[:, :prefix_len]
        kernel = max(1, min(pool_kernel, prefix_len))
        if kernel % 2 == 0:
            kernel = max(1, kernel - 1)
        if kernel > 1:
            pooled = F.max_pool1d(
                prefix_score.unsqueeze(1), kernel_size=kernel, stride=1, padding=kernel // 2
            ).squeeze(1)
        else:
            pooled = prefix_score
        top_idx = torch.topk(pooled, k=min(keep_prefix, prefix_len), dim=-1).indices
        top_idx = torch.sort(top_idx, dim=-1).values
    else:
        top_idx = torch.empty((heads, 0), device=key.device, dtype=torch.long)

    if recent > 0:
        recent_idx = torch.arange(prefix_len, seq_len, device=key.device, dtype=torch.long)
        recent_idx = recent_idx.unsqueeze(0).expand(heads, -1)
    else:
        recent_idx = torch.empty((heads, 0), device=key.device, dtype=torch.long)

    keep_idx = torch.cat([top_idx, recent_idx], dim=-1)
    if keep_idx.shape[1] == 0:
        keep_idx = torch.full((heads, 1), seq_len - 1, device=key.device, dtype=torch.long)

    key = _gather_by_head(key, keep_idx)
    value = _gather_by_head(value, keep_idx)

    key_min, key_max, page_tokens = _build_page_minmax(key, page_size=page_size)
    state.update(
        {
            "prefill_done": True,
            "prefill_len": key.shape[2],
            "page_size": max(1, page_size),
            "page_min": key_min,
            "page_max": key_max,
            "page_tokens": page_tokens,
            "decode_steps": 0,
        }
    )
    state["last_debug"] = {
        "method": "rocketkv",
        "phase": "prefill",
        "seq_in": int(seq_len),
        "seq_out": int(key.shape[2]),
        "budget": int(budget),
        "prompt_budget": int(key.shape[2]),
        "recent_budget": int(recent),
        "page_size": int(max(1, page_size)),
        "pages_selected": int(key_min.shape[1]),
    }
    return key, value, state


def _rocket_candidate_tokens(
    query: torch.Tensor,
    history_key: torch.Tensor,
    prefix_len: int,
    select_budget: int,
    page_size: int,
    state: Dict[str, Any],
) -> torch.Tensor:
    # returns [H, C] candidate token indices in [0, prefix_len)
    heads = history_key.shape[1]
    device = history_key.device
    page_size = max(1, page_size)
    if prefix_len <= 0:
        return torch.empty((heads, 0), device=device, dtype=torch.long)

    # Reuse stage-1 page min/max only when sequence matches; otherwise rebuild once.
    if (
        state.get("page_min") is None
        or state.get("page_max") is None
        or state.get("page_tokens") != history_key.shape[2]
    ):
        page_min, page_max, page_tokens = _build_page_minmax(history_key, page_size=page_size)
        state["page_min"] = page_min
        state["page_max"] = page_max
        state["page_tokens"] = page_tokens
    page_min = state["page_min"]
    page_max = state["page_max"]
    num_pages = page_min.shape[1]

    q_head = query.float().mean(dim=(0, 2))  # [H, D]
    pos = torch.clamp(q_head, min=0.0).unsqueeze(1)
    neg = torch.clamp(q_head, max=0.0).unsqueeze(1)
    page_score = (pos * page_max + neg * page_min).sum(dim=-1)  # [H, P]

    pages_needed = max(1, int(math.ceil((select_budget * 1.5) / page_size)))
    pages_needed = min(pages_needed, num_pages)
    top_pages = torch.topk(page_score, k=pages_needed, dim=-1).indices  # [H, pages_needed]

    candidates_per_head = []
    max_len = 0
    for h in range(heads):
        idx_h = []
        for p in top_pages[h].tolist():
            start = p * page_size
            end = min(start + page_size, prefix_len)
            if start < end:
                idx_h.extend(range(start, end))
        if not idx_h:
            idx_h = [prefix_len - 1]
        idx_h = sorted(set(idx_h))
        tensor_h = torch.tensor(idx_h, device=device, dtype=torch.long)
        candidates_per_head.append(tensor_h)
        max_len = max(max_len, tensor_h.numel())

    out = torch.empty((heads, max_len), device=device, dtype=torch.long)
    for h in range(heads):
        idx_h = candidates_per_head[h]
        if idx_h.numel() < max_len:
            pad = idx_h[-1].repeat(max_len - idx_h.numel())
            idx_h = torch.cat([idx_h, pad], dim=0)
        out[h] = idx_h
    return out


def _rocket_stage2_decode(
    key: torch.Tensor,
    value: torch.Tensor,
    query: Optional[torch.Tensor],
    budget: int,
    recent_window: int,
    page_size: int,
    history_len: int,
    state: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    seq_len = key.shape[2]
    heads = key.shape[1]
    history_len = max(0, min(history_len, seq_len))
    current_len = seq_len - history_len

    if history_len == 0:
        state["last_debug"] = {
            "method": "rocketkv",
            "phase": "decode",
            "seq_in": int(seq_len),
            "seq_out": int(seq_len),
            "budget": int(budget),
            "history_len": int(history_len),
            "history_budget": int(max(budget - (seq_len - history_len), 0)),
            "select_budget": 0,
            "recent_budget": 0,
            "decode_step": int(state.get("decode_steps", 0)),
        }
        return key, value, state

    history_budget = max(budget - current_len, 0)
    if history_budget <= 0:
        current_idx = torch.arange(history_len, seq_len, device=key.device, dtype=torch.long)
        if current_idx.numel() == 0:
            current_idx = torch.tensor([seq_len - 1], device=key.device, dtype=torch.long)
        current_idx = current_idx.unsqueeze(0).expand(heads, -1)
        key = _gather_by_head(key, current_idx)
        value = _gather_by_head(value, current_idx)
        state["decode_steps"] = int(state.get("decode_steps", 0)) + 1
        state["last_debug"] = {
            "method": "rocketkv",
            "phase": "decode",
            "seq_in": int(seq_len),
            "seq_out": int(key.shape[2]),
            "budget": int(budget),
            "history_len": int(history_len),
            "history_budget": int(history_budget),
            "select_budget": 0,
            "recent_budget": 0,
            "decode_step": int(state.get("decode_steps", 0)),
        }
        return key, value, state

    if history_len <= history_budget:
        state["decode_steps"] = int(state.get("decode_steps", 0)) + 1
        state["last_debug"] = {
            "method": "rocketkv",
            "phase": "decode",
            "seq_in": int(seq_len),
            "seq_out": int(seq_len),
            "budget": int(budget),
            "history_len": int(history_len),
            "history_budget": int(history_budget),
            "select_budget": 0,
            "recent_budget": 0,
            "decode_step": int(state.get("decode_steps", 0)),
        }
        return key, value, state

    recent_hist = min(max(recent_window, 0), history_budget, history_len)
    prefix_len = history_len - recent_hist
    select_budget = max(history_budget - recent_hist, 0)

    if select_budget > 0 and prefix_len > 0 and query is not None:
        history_key = key[:, :, :history_len, :]
        candidate_idx = _rocket_candidate_tokens(
            query=query,
            history_key=history_key,
            prefix_len=prefix_len,
            select_budget=select_budget,
            page_size=page_size,
            state=state,
        )
        candidate_key = _gather_by_head(history_key, candidate_idx)
        candidate_score = _attention_scores(query, candidate_key, observe_q=8)  # [H, C]
        sel_local = torch.topk(
            candidate_score, k=min(select_budget, candidate_idx.shape[1]), dim=-1
        ).indices
        selected_hist = torch.gather(candidate_idx, dim=1, index=sel_local)
        selected_hist = torch.sort(selected_hist, dim=-1).values
    else:
        selected_hist = torch.empty((heads, 0), device=key.device, dtype=torch.long)

    if recent_hist > 0:
        recent_idx = torch.arange(
            history_len - recent_hist, history_len, device=key.device, dtype=torch.long
        )
        recent_idx = recent_idx.unsqueeze(0).expand(heads, -1)
    else:
        recent_idx = torch.empty((heads, 0), device=key.device, dtype=torch.long)

    hist_keep = torch.cat([selected_hist, recent_idx], dim=-1)
    if hist_keep.shape[1] == 0:
        hist_keep = torch.full((heads, 1), history_len - 1, device=key.device, dtype=torch.long)
    hist_keep = torch.sort(hist_keep, dim=-1).values

    current_idx = torch.arange(history_len, seq_len, device=key.device, dtype=torch.long)
    current_idx = current_idx.unsqueeze(0).expand(heads, -1)
    keep_idx = torch.cat([hist_keep, current_idx], dim=-1)

    key = _gather_by_head(key, keep_idx)
    value = _gather_by_head(value, keep_idx)
    state["decode_steps"] = int(state.get("decode_steps", 0)) + 1
    state["last_debug"] = {
        "method": "rocketkv",
        "phase": "decode",
        "seq_in": int(seq_len),
        "seq_out": int(key.shape[2]),
        "budget": int(budget),
        "history_len": int(history_len),
        "history_budget": int(history_budget),
        "select_budget": int(select_budget),
        "recent_budget": int(recent_hist),
        "pages_selected": int(state.get("page_min").shape[1])
        if state.get("page_min") is not None
        else 0,
        "decode_step": int(state.get("decode_steps", 0)),
    }
    return key, value, state


def _infinipot_v_indices(
    key: torch.Tensor,
    value: torch.Tensor,
    budget: int,
    recent_window: int,
    alpha: float,
) -> torch.Tensor:
    seq_len = key.shape[2]
    device = key.device

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

    k_mean = key[:, :, :prefix_len].float().mean(dim=(0, 1))
    k_mean = F.normalize(k_mean, dim=-1, eps=1e-6)
    tar = torch.zeros(prefix_len, device=device, dtype=torch.float32)
    if prefix_len > 1:
        sim = (k_mean[1:] * k_mean[:-1]).sum(dim=-1).clamp(-1.0, 1.0)
        tar[1:] = 1.0 - sim
        tar[0] = tar[1]
    tar = _minmax_norm(tar)

    van = value[:, :, :prefix_len].float().norm(dim=-1).mean(dim=(0, 1))
    van = _minmax_norm(van)
    score = alpha * tar + (1.0 - alpha) * van

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
    phase: str = "decode",
    state: Optional[Dict[str, Any]] = None,
    history_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
    """
    Stateful KV compression entry for AR vision cache.

    Args:
        key/value: [B, H, S, D]
        phase: "prefill" (cache build) or "decode" (denoise loop)
        state: persistent per-layer compression state.
        history_len: valid only in decode phase for RocketKV, length of historical cache.
    """
    method = _normalize_method(method)
    seq_len = key.shape[2]
    if method == "none" or max_tokens is None or max_tokens <= 0:
        return key, value, state

    budget = min(max_tokens, seq_len)
    if method == "h2o":
        state = _ensure_state(state, method)
        key, value, state = _compress_h2o(
            key=key,
            value=value,
            query=query,
            budget=budget,
            recent_window=recent_window,
            state=state,
        )
        return key, value, state

    if method == "rocketkv":
        state = _ensure_state(state, method)
        if phase == "prefill":
            key, value, state = _rocket_stage1_prefill(
                key=key,
                value=value,
                query=query,
                budget=budget,
                recent_window=recent_window,
                pool_kernel=rocket_pool_kernel,
                page_size=rocket_page_size,
                state=state,
            )
            return key, value, state

        # decode phase
        if history_len is None:
            history_len = state.get("prefill_len", 0)
        key, value, state = _rocket_stage2_decode(
            key=key,
            value=value,
            query=query,
            budget=budget,
            recent_window=recent_window,
            page_size=rocket_page_size,
            history_len=int(history_len),
            state=state,
        )
        return key, value, state

    if method == "infinipot_v":
        seq_in = int(seq_len)
        keep_idx = _infinipot_v_indices(
            key,
            value,
            budget,
            recent_window=recent_window,
            alpha=float(max(0.0, min(1.0, infinipot_alpha))),
        )
        if keep_idx.numel() == 0:
            keep_idx = torch.tensor([seq_len - 1], device=key.device, dtype=torch.long)
        key = key.index_select(dim=2, index=keep_idx)
        value = value.index_select(dim=2, index=keep_idx)
        if state is not None:
            state["last_debug"] = {
                "method": "infinipot_v",
                "phase": phase,
                "seq_in": seq_in,
                "seq_out": int(key.shape[2]),
                "budget": int(budget),
                "recent_budget": int(min(max(recent_window, 0), budget // 2, seq_in)),
                "alpha": float(infinipot_alpha),
            }
        return key, value, state

    raise ValueError(
        f"Unsupported kv compression method: {method}. "
        "Expected one of ['none', 'h2o', 'rocketkv', 'infinipot_v']."
    )
