# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend CANN adaptation of the 2-bit MoE expert offloading system.

Migration from vllm-moet CUDA patch (vLLM v0.24.0) to Ascend CANN.
Target model: Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp (INT8 per-row).

Modules:
  ascend_device          — thin device-abstraction layer (torch_npu)
  moe_w2_planes          — 2-bit quantization math + INT8 dequant
  moe_w2_ascend_core     — main orchestrator (build + forward)
  moe_w2_delta           — FP4 delta tier (GPU pool + manager)
  moe_w2_gate            — confidence-gated FP4 re-forward
  moe_w2_looka           — router-lookahead prefetch
  moe_w2_store           — host-side storage backends
  moe_w2_planes_cache    — disk cache for built planes
  prefill_timers         — NPU event timers
  quantization_hooks_ascend — INT8 per-row MoE method
  envs_ascend            — environment variable reference
  runner_ascend          — model-runner integration hooks
"""

from .ascend_device import (
    Event,
    Stream,
    current_device,
    current_stream,
    init,
    is_available,
    is_current_stream_capturing,
    mem_get_info,
    set_device,
    synchronize,
)
from .moe_w2_ascend_core import (
    build_layer_planes_int8,
    enabled,
    is_w2_layer,
    moe_w2_forward,
    plan_pack_skip,
    arm_stream_build,
    ready,
)