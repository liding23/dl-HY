# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class InferState:
    enable_sageattn: bool = False  # whether to use SageAttention
    sage_blocks_range: Optional[range] = None  # block range to use SageAttention
    enable_torch_compile: bool = False  # whether to use torch compile

    # fp8 gemm related
    use_fp8_gemm: bool = False  # whether to use fp8 gemm
    quant_type: str = "fp8-per-block"  # fp8 quantization type
    include_patterns: list = field(
        default_factory=lambda: ["double_blocks"]
    )  # include patterns for fp8 gemm

    # vae related
    use_vae_parallel: bool = False  # whether to use vae parallel

    # kv compression related
    kv_compression_method: str = "none"  # none/h2o/rocketkv/infinipot_v
    kv_max_tokens: int = 0  # <=0 means disabled
    kv_recent_window: int = 1024
    rocket_pool_kernel: int = 31
    rocket_page_size: int = 64
    infinipot_alpha: float = 0.6
    kv_debug_log: bool = False
    kv_debug_phase: str = "both"  # prefill/decode/both
    kv_debug_layers: Optional[set] = None  # None means all
    kv_debug_interval: int = 10
    kv_debug_level: str = "summary"  # summary/detail
    kv_debug_file: str = ""


__infer_state = None


def parse_range(value):
    if "-" in value:
        start, end = map(int, value.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(x) for x in value.split(",")]


def parse_layer_set(value):
    if value is None:
        return None
    value = value.strip().lower()
    if value in ("", "all"):
        return None
    out = set()
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = map(int, chunk.split("-", 1))
            if end < start:
                start, end = end, start
            out.update(range(start, end + 1))
        else:
            out.add(int(chunk))
    return out


def initialize_infer_state(args):
    global __infer_state
    sage_blocks_range = parse_range(args.sage_blocks_range)
    # Map CLI argument use_sageattn to internal enable_sageattn field
    use_sageattn = getattr(args, "use_sageattn", False)

    # Parse include_patterns from args
    include_patterns = getattr(args, "include_patterns", "double_blocks")
    if isinstance(include_patterns, str):
        # Split by comma and strip whitespace
        include_patterns = [p.strip() for p in include_patterns.split(",") if p.strip()]

    __infer_state = InferState(
        enable_sageattn=use_sageattn,
        sage_blocks_range=sage_blocks_range,
        enable_torch_compile=args.enable_torch_compile,
        # fp8 gemm related
        use_fp8_gemm=args.use_fp8_gemm,
        quant_type=args.quant_type,
        include_patterns=include_patterns,
        # vae related
        use_vae_parallel=args.use_vae_parallel,
        # kv compression related
        kv_compression_method=getattr(args, "kv_compression_method", "none"),
        kv_max_tokens=getattr(args, "kv_max_tokens", 0),
        kv_recent_window=getattr(args, "kv_recent_window", 1024),
        rocket_pool_kernel=getattr(args, "rocket_pool_kernel", 31),
        rocket_page_size=getattr(args, "rocket_page_size", 64),
        infinipot_alpha=getattr(args, "infinipot_alpha", 0.6),
        kv_debug_log=getattr(args, "kv_debug_log", False),
        kv_debug_phase=getattr(args, "kv_debug_phase", "both"),
        kv_debug_layers=parse_layer_set(getattr(args, "kv_debug_layers", "all")),
        kv_debug_interval=max(1, int(getattr(args, "kv_debug_interval", 10))),
        kv_debug_level=getattr(args, "kv_debug_level", "summary"),
        kv_debug_file=getattr(args, "kv_debug_file", ""),
    )
    return __infer_state


def get_infer_state():
    return __infer_state
