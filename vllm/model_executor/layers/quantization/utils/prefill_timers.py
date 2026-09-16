# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-gated Event timers for prefill anatomy (VLLM_PREFILL_TIMERS=1).

Ascend-adapted: uses ascend_device for stream/event operations instead
of torch.cuda.*. Zero overhead when the env is off.
"""

import os
from contextlib import contextmanager

from vllm.logger import init_logger

from . import ascend_device

logger = init_logger(__name__)

ENABLED = os.getenv("VLLM_PREFILL_TIMERS", "0") == "1"
FLUSH_EVERY = int(os.getenv("VLLM_PREFILL_TIMERS_FLUSH", "172"))

_pending: dict = {}
_total_ms: dict = {}
_count: dict = {}


@contextmanager
def span(name: str):
    if not ENABLED or ascend_device.is_current_stream_capturing():
        yield
        return
    e0 = ascend_device.Event(enable_timing=True)
    e1 = ascend_device.Event(enable_timing=True)
    e0.record()
    try:
        yield
    finally:
        e1.record()
        _pending.setdefault(name, []).append((e0, e1))
        if len(_pending[name]) >= FLUSH_EVERY:
            _flush(name)


def _flush(name):
    ascend_device.synchronize()
    pairs = _pending.pop(name, [])
    ms = sum(a.elapsed_time(b) for a, b in pairs)
    _total_ms[name] = _total_ms.get(name, 0.0) + ms
    _count[name] = _count.get(name, 0) + len(pairs)
    logger.info("[prefill-timer] %-14s total %8.1f ms over %5d spans",
                name, _total_ms[name], _count[name])