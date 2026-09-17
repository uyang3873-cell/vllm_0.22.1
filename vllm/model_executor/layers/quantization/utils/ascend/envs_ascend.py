# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment variables for the Ascend CANN MoE offloading system.

This file documents ALL environment variables that control the 2-bit MoE
expert offloading system on Ascend. It should be integrated into the
Ascend vLLM port's environment variable registration (equivalent to the
CUDA version's envs.py patch).

Variables fall into these groups:
  MASTER    — enable/disable the whole system
  STORE     — where expert planes live (host RAM, SSD, or GPU)
  DELTA     — FP4 quality-restoration tier
  GATE      — confidence-gated re-forward
  LOOKA     — router-lookahead prefetch
  BUILD     — load-time quantization behavior
"""

import os

# ===========================================================================
# MASTER SWITCH
# ===========================================================================
# VLLM_MOE_W2: 0 (default) | 1
#   Master enable for the entire 2-bit MoE path. When 0, all other variables
#   are inert and the stock Ascend MoE pipeline runs unchanged.
VLLM_MOE_W2 = int(os.getenv("VLLM_MOE_W2", "0"))

# VLLM_MOE_W2_NUM_LAYERS: integer (auto-detected from model config if unset)
#   Number of main-model transformer layers. Layers >= this count are treated
#   as MTP drafter layers and keep the stock quantization path.
# VLLM_MOE_W2_NUM_LAYERS = auto

# ===========================================================================
# QUANTIZATION
# ===========================================================================
# VLLM_MOE_W2_ZERO_MODE: auto (default) | sign | alt
#   How zero-weight values map to the sign-symmetric codebook.
#   auto: sign-preserving unless >95% of zeros are same-signed, then alternating
#   sign: always sign-preserving (standard)
#   alt:  always alternating by k-position parity
VLLM_MOE_W2_ZERO_MODE = os.getenv("VLLM_MOE_W2_ZERO_MODE", "auto")

# VLLM_MOE_W2_TOPP: 0.0 (default) | float in (0, 1)
#   Adaptive expert top-p: drop each token's routed-expert tail past
#   cumulative weight p. 0 = off (exact stock routing).
VLLM_MOE_W2_TOPP = float(os.getenv("VLLM_MOE_W2_TOPP", "0"))

# VLLM_MOE_W2_TOPP_MIN: 2 (default) | int >= 1
#   Minimum number of experts always kept per token (top-p guard).
VLLM_MOE_W2_TOPP_MIN = int(os.getenv("VLLM_MOE_W2_TOPP_MIN", "2"))

# VLLM_MOE_W2_TOPP_RENORM: 1 (default) | 0
#   1: renormalize kept weights so total routed weight is preserved.
VLLM_MOE_W2_TOPP_RENORM = int(os.getenv("VLLM_MOE_W2_TOPP_RENORM", "1"))

# ===========================================================================
# STORES — where expert planes live
# ===========================================================================
# VLLM_MOE_W2_STORE_DIR: path (unset = PinnedHostStore, the default)
#   Directory for the pack-file store. When set, expert rows are persisted
#   to disk and served via pread -> pinned stage -> H2D.
VLLM_MOE_W2_STORE_DIR = os.getenv("VLLM_MOE_W2_STORE_DIR", "")

# VLLM_MOE_W2_BASE_RAM_GB: float | "auto" (requires VLLM_MOE_W2_STORE_DIR)
#   Pinned arena size for the tiered store (MRU cache over the pack file).
#   "auto" = 25% of the total pack size.
VLLM_MOE_W2_BASE_RAM_GB = os.getenv("VLLM_MOE_W2_BASE_RAM_GB", "")

# VLLM_MOE_W2_STORE_THREADS: 8 (default) | int
#   Thread pool size for parallel pack-file reads.
VLLM_MOE_W2_STORE_THREADS = int(os.getenv("VLLM_MOE_W2_STORE_THREADS", "8"))

# VLLM_MOE_W2_TIER_DIRECT: 0 (default) | 1
#   1: O_DIRECT reads for tiered store misses (bypass page cache).
VLLM_MOE_W2_TIER_DIRECT = int(os.getenv("VLLM_MOE_W2_TIER_DIRECT", "0"))

# VLLM_MOE_W2_TIER_SCAN: 1 (default) | 0
#   1: scan resistance — prefill batches never evict from the arena.
VLLM_MOE_W2_TIER_SCAN = int(os.getenv("VLLM_MOE_W2_TIER_SCAN", "1"))

# VLLM_MOE_W2_TIER_PREHEAT: 1 (default) | 0
#   1: on boot, load the previous run's hot set into the arena.
VLLM_MOE_W2_TIER_PREHEAT = int(os.getenv("VLLM_MOE_W2_TIER_PREHEAT", "1"))

# ===========================================================================
# DELTA TIER — FP4 quality restoration
# ===========================================================================
# VLLM_MOE_W2_DELTA_GB: 2.0 (default) | float | "auto"
#   GPU pool size for the FP4 delta tier. "auto" sizes from free VRAM after
#   KV cache allocation (minus VLLM_MOE_W2_DELTA_RESERVE_GB).
VLLM_MOE_W2_DELTA_GB = os.getenv("VLLM_MOE_W2_DELTA_GB", "2.0")

# VLLM_MOE_W2_DELTA_RESERVE_GB: 3.0 (default) | float
#   VRAM to leave free when auto-sizing the delta pool.
VLLM_MOE_W2_DELTA_RESERVE_GB = float(os.getenv("VLLM_MOE_W2_DELTA_RESERVE_GB", "3.0"))

# VLLM_MOE_W2_DELTA_MAX_GB: 0 (default = uncapped) | float
#   Hard cap on auto-sized delta pool.
VLLM_MOE_W2_DELTA_MAX_GB = float(os.getenv("VLLM_MOE_W2_DELTA_MAX_GB", "0"))

# VLLM_MOE_W2_DELTA_POLICY: freq (default) | lru | need
#   Promotion/eviction policy:
#   freq: promote hottest candidates by recency-decayed routing frequency
#   lru:  promote in order, evict coldest (old behaviour)
#   need: FP4 filled only by the confidence gate's force_promote
VLLM_MOE_W2_DELTA_POLICY = os.getenv("VLLM_MOE_W2_DELTA_POLICY", "freq")

# VLLM_MOE_W2_DELTA_DECAY: 0.5 (default) | float
#   Recency-decay factor for the routing-frequency signal.
VLLM_MOE_W2_DELTA_DECAY = float(os.getenv("VLLM_MOE_W2_DELTA_DECAY", "0.5"))

# VLLM_MOE_W2_DELTA_PROMOTE: 8 (default) | int
#   Max experts promoted per manager tick.
VLLM_MOE_W2_DELTA_PROMOTE = int(os.getenv("VLLM_MOE_W2_DELTA_PROMOTE", "8"))

# VLLM_MOE_W2_DELTA_TICK_MS: 5 (default) | int
#   Minimum ms between manager passes (rate limit).
VLLM_MOE_W2_DELTA_TICK_MS = int(os.getenv("VLLM_MOE_W2_DELTA_TICK_MS", "5"))

# VLLM_MOE_W2_DELTA_DECAY_S: 5.0 (default) | float
#   Wall-clock seconds between frequency-signal decay steps.
VLLM_MOE_W2_DELTA_DECAY_S = float(os.getenv("VLLM_MOE_W2_DELTA_DECAY_S", "5"))

# VLLM_MOE_W2_DELTA_SPLIT: 0 (default) | 1
#   Split FP4 mode: refinement planes are 2-bit corrections read alongside
#   the base plane (instead of full nibble bytes replacing the base).
VLLM_MOE_W2_DELTA_SPLIT = int(os.getenv("VLLM_MOE_W2_DELTA_SPLIT", "0"))

# VLLM_MOE_W2_PROMO_HYST: 0 (default) | float > 1.0
#   Promotion hysteresis multiplier. Candidate must exceed h*victim + 4.
VLLM_MOE_W2_PROMO_HYST = float(os.getenv("VLLM_MOE_W2_PROMO_HYST", "0"))

# VLLM_MOE_W2_SPEC_GUARD: 60 (default) | float (0 = off)
#   % replay threshold to suppress speculative decoding during cold pool.
VLLM_MOE_W2_SPEC_GUARD = float(os.getenv("VLLM_MOE_W2_SPEC_GUARD", "60"))

# VLLM_MOE_W2_SPEC_GUARD_EMA: 0.02 (default) | float
#   EMA smoothing factor for the replay-rate estimate.
VLLM_MOE_W2_SPEC_GUARD_EMA = float(os.getenv("VLLM_MOE_W2_SPEC_GUARD_EMA", "0.02"))

# ===========================================================================
# CONFIDENCE GATE
# ===========================================================================
# VLLM_MOE_W2_GATE: 0 (default) | 1
#   Master switch for the confidence-gated FP4 re-forward.
VLLM_MOE_W2_GATE = int(os.getenv("VLLM_MOE_W2_GATE", "0"))

# VLLM_MOE_W2_GATE_SIGNAL: max_prob (default) | margin
#   Signal to gate on: max_prob = exp(top1_logit - logsumexp),
#   margin = top1_logit - top2_logit.
VLLM_MOE_W2_GATE_SIGNAL = os.getenv("VLLM_MOE_W2_GATE_SIGNAL", "max_prob")

# VLLM_MOE_W2_GATE_TAU: 0.60 (default) | float
#   Fire threshold. Lower = fewer re-runs = lower latency.
VLLM_MOE_W2_GATE_TAU = float(os.getenv("VLLM_MOE_W2_GATE_TAU", "0.60"))

# VLLM_MOE_W2_GATE_MAX_PROMOTE: 64 (default) | int (0 = unlimited)
#   Cap on experts force-promoted per fired step.
VLLM_MOE_W2_GATE_MAX_PROMOTE = int(os.getenv("VLLM_MOE_W2_GATE_MAX_PROMOTE", "64"))

# VLLM_MOE_W2_GATE_TRACE: 0 (default) | 1
#   Log each gate fire and re-forward.
VLLM_MOE_W2_GATE_TRACE = int(os.getenv("VLLM_MOE_W2_GATE_TRACE", "0"))

# VLLM_MOE_W2_GATE_REFORWARD: 1 (default) | 0
#   0: force-promote but skip the 2nd forward (diagnostic: isolate promote
#   correctness from re-forward correctness).
VLLM_MOE_W2_GATE_REFORWARD = int(os.getenv("VLLM_MOE_W2_GATE_REFORWARD", "1"))

# ===========================================================================
# LOOKAHEAD / PREFETCH
# ===========================================================================
# VLLM_MOE_W2_LOOKA: 0 (default) | 1
#   Router-lookahead measurement (counters only, no behaviour change).
VLLM_MOE_W2_LOOKA = int(os.getenv("VLLM_MOE_W2_LOOKA", "0"))

# VLLM_MOE_W2_PILOT: 0 (default) | 1
#   Router-lookahead prefetch. Implies LOOKA machinery.
VLLM_MOE_W2_PILOT = int(os.getenv("VLLM_MOE_W2_PILOT", "0"))

# VLLM_MOE_W2_PILOT_K: 8 (default) | int 1..16
#   Top-K predictions kept per position for pilot prefetch.
VLLM_MOE_W2_PILOT_K = int(os.getenv("VLLM_MOE_W2_PILOT_K", "8"))

# VLLM_MOE_W2_PILOT_CAP: 32 (default) | int
#   Max experts fetched per manager tick from the pilot log.
VLLM_MOE_W2_PILOT_CAP = int(os.getenv("VLLM_MOE_W2_PILOT_CAP", "32"))

# ===========================================================================
# BUILD (load-time)
# ===========================================================================
# VLLM_MOE_W2_DIRECT_MODE: 0 (default) | 1
#   1: skip 2-bit requantization, keep INT8 weights directly on NPU.
#   For w8a8 models (e.g. Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp) that already
#   have INT8 per-row quantized checkpoints. The 2-bit/FP4 pipeline code is
#   preserved but not invoked. NPU memory usage is ~3.6x higher vs 2-bit mode.
VLLM_MOE_W2_DIRECT_MODE = int(os.getenv("VLLM_MOE_W2_DIRECT_MODE", "0"))

# VLLM_MOE_W2_STREAM_BUILD: 1 (default) | 0
#   1: requantize each layer the moment its last expert tensor lands
#   (streaming, peak staging = O(one layer)).
VLLM_MOE_W2_STREAM_BUILD = int(os.getenv("VLLM_MOE_W2_STREAM_BUILD", "1"))

# VLLM_MOE_W2_PLANES_CACHE: path (unset = off)
#   Directory for the GPU-resident planes disk cache (skips requant on
#   restarts for the same checkpoint + TP layout).
VLLM_MOE_W2_PLANES_CACHE = os.getenv("VLLM_MOE_W2_PLANES_CACHE", "")

# VLLM_MOE_W2_DENSE_FP8: 0 (default) | 1 | attn | l0
#   Online-quantize the non-expert dense GEMMs to FP8.
#   1: shared experts + first dense MLP
#   attn: + attention projections
#   l0: first dense MLP only
VLLM_MOE_W2_DENSE_FP8 = os.getenv("VLLM_MOE_W2_DENSE_FP8", "0")

# VLLM_MOE_W2_CUBIT_DIR: /cubit-share (default, CUDA only)
#   NOT APPLICABLE on Ascend — cubin kernels are not migrated.
# VLLM_MOE_W2_CUBIT_DIR = "/cubit-share"

# VLLM_W8A8_SKINNY_CUBIT_DIR: /cubit-share (default, CUDA only)
#   NOT APPLICABLE on Ascend — CUDA SASS kernel not migrated.
# VLLM_W8A8_SKINNY_CUBIT_DIR = "/cubit-share"

# ===========================================================================
# OBSERVABILITY
# ===========================================================================
# VLLM_MOE_W2_DELTA_TRACE: 0 (default) | 1 | 2
#   FP4 delta tier observability (see moe_w2_delta.py).
VLLM_MOE_W2_DELTA_TRACE = int(os.getenv("VLLM_MOE_W2_DELTA_TRACE", "0"))

# VLLM_PREFILL_TIMERS: 0 (default) | 1
#   CUDA/NPU event timers for prefill anatomy.
VLLM_PREFILL_TIMERS = int(os.getenv("VLLM_PREFILL_TIMERS", "0"))

# ===========================================================================
# DEPRECATED / REMOVED (not applicable on Ascend)
# ===========================================================================
# VLLM_MOE_W2_BASE_NVME_RATIO — superseded by pack store, IGNORED
# VLLM_MOE_W2_AFRAG — CUDA cubin AFRAG repack, NOT MIGRATED
# VLLM_W8A8_SKINNY_CUBIT — CUDA SASS kernel, NOT MIGRATED
# VLLM_W8A8_SKINNY_REPACK_KS — CUDA SASS repack, NOT MIGRATED