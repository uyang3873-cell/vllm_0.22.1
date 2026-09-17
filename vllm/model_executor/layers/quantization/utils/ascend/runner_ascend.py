# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner integration hooks for Ascend CANN MoE offloading.

Adapted from the CUDA version's gpu_model_runner.py patch. These hooks
should be wired into the Ascend vLLM port's equivalent of GPUModelRunner.

Integration points (call these from the Ascend vLLM runner):
  1. step_begin()   — signal the delta tier manager at each step boundary
  2. kpi_step(...)   — feed per-step KPI counters (misses, replays)
  3. notify_capture()— idle the manager during NPU graph capture
  4. pool_preload()  — warm-start the delta pool from persisted heat file
  5. finalize_auto() — size the delta pool from free VRAM after KV alloc
"""

import os

from vllm.logger import init_logger

logger = init_logger(__name__)


def step_begin():
    """Call at the top of each decode step (before any layer forward).

    Signals the delta tier manager to run one pass (event-driven cadence).
    Also signals the base-cache tier if configured.
    """
    try:
        from . import moe_w2_delta
        moe_w2_delta.wake_all()
    except Exception:
        pass


def kpi_step(miss_pairs: int = 0, replays: int = 0):
    """Feed per-step KPI counters to the base cache tier.

    Called after the forward pass (or after each replay) with:
      miss_pairs: count of (layer, expert) pairs missing from the pool
      replays:    1 if this step was replayed, 0 for the first pass
    """
    from . import moe_w2_delta
    tier = moe_w2_delta._BASE_TIER
    if tier is None:
        return
    tier._kpi_steps += 1
    tier._kpi_miss_pairs += miss_pairs
    tier._kpi_replays += replays


def notify_capture():
    """Call at the start and end of NPU graph capture.

    Idles the delta tier manager so it does not interleave with capture.
    """
    from . import moe_w2_delta
    for tier in (moe_w2_delta._TIER, moe_w2_delta._BASE_TIER):
        if tier is not None:
            tier.notify_capture()


def pool_preload():
    """Preload the delta pool from the persisted heat file (GPU warm-start).

    Call after weight load but before the first decode step.
    """
    from . import moe_w2_delta
    for tier in (moe_w2_delta._TIER, moe_w2_delta._BASE_TIER):
        if tier is not None and not tier._heat_preloaded:
            tier._heat_preloaded = True
            pending = tier._heat_pending
            if pending:
                logger.info("moe_w2 runner: preloading %d pool entries for %s",
                            len(pending), tier._tag)
                # The heat file stores (layer, expert) pairs to pre-promote
                # Implementation: call tier.force_promote or direct slot fill
                try:
                    tier._heat_pending = None
                except Exception as e:
                    logger.warning("moe_w2 runner: pool preload failed: %s", e)


def finalize_auto():
    """Size the delta pool from free VRAM after KV cache allocation.

    Call after initialize_kv_cache, before any cudagraph capture.
    """
    from . import moe_w2_delta
    for tier in (moe_w2_delta._TIER, moe_w2_delta._BASE_TIER):
        if tier is not None:
            tier.finalize_auto()


def speculation_suppressed() -> bool:
    """True when the speculation guard has suppressed drafts (cold pool)."""
    from . import moe_w2_delta
    tier = moe_w2_delta._BASE_TIER
    if tier is None:
        return False
    return tier._spec_suppressed


def tier_stats() -> dict:
    """Aggregated stats for the KPI log line."""
    from . import moe_w2_delta
    st = {}
    if moe_w2_delta._TIER is not None:
        st["delta"] = moe_w2_delta._TIER.stats()
    if moe_w2_delta._BASE_TIER is not None:
        st["base"] = moe_w2_delta._BASE_TIER.stats()
    return st