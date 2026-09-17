# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin device-abstraction layer for Ascend CANN (torch_npu).

All migrated moe_w2 modules import this instead of touching torch.cuda.*
or torch_npu.* directly. On Ascend hardware torch_npu is available; on
development machines without NPU drivers the module degrades gracefully
(useful for import/type-checking the migrated code on x86).
"""

import os
from typing import Optional

import torch

_NPU_OK = False
try:
    import torch_npu

    _NPU_OK = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Graceful fallback when torch_npu is not importable (e.g. dev on x86).
# The modules that import this file still parse; they will fail at first use
# with a clear message instead of an ImportError in the import path.
# ---------------------------------------------------------------------------

_FAKE_STREAM_CLS = None
_FAKE_EVENT_CLS = None

if not _NPU_OK:

    class _FakeStream:
        """Minimal Stream stand-in so `ascend_device.Stream(dev)` parses."""

        def __init__(self, dev=None):
            self.npu_stream = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class _FakeEvent:
        """Minimal Event stand-in for the same reason."""

        def __init__(self, enable_timing=False):
            pass

        def record(self, stream=None):
            pass

        def synchronize(self):
            pass

        def wait(self, stream=None):
            pass

        def query(self):
            return True

    _FAKE_STREAM_CLS = _FakeStream
    _FAKE_EVENT_CLS = _FakeEvent


def _fail(msg: str):
    raise RuntimeError(
        f"ascend_device: {msg} — torch_npu is not available. "
        f"Install torch_npu or run on an Ascend machine."
    )


# ---- device ---------------------------------------------------------------


def current_device() -> torch.device:
    if not _NPU_OK:
        return torch.device("cpu")  # let import-time code proceed
    return torch.device("npu")


def device_count() -> int:
    if not _NPU_OK:
        _fail("device_count")
    return torch_npu.npu.device_count()


# ---- stream ---------------------------------------------------------------


def current_stream(dev: Optional[torch.device] = None) -> "torch.Stream":
    if not _NPU_OK:
        return _FAKE_STREAM_CLS(dev)
    return torch_npu.npu.current_stream(dev)


def Stream(dev: Optional[torch.device] = None):  # noqa: N802
    if not _NPU_OK:
        return _FAKE_STREAM_CLS(dev)
    return torch_npu.npu.Stream(dev)


# ---- event ----------------------------------------------------------------


def Event(enable_timing: bool = False):  # noqa: N802
    if not _NPU_OK:
        return _FAKE_EVENT_CLS(enable_timing)
    return torch_npu.npu.Event(enable_timing)


def synchronize(dev: Optional[torch.device] = None):
    if not _NPU_OK:
        return
    torch_npu.npu.synchronize(dev)


# ---- memory info ----------------------------------------------------------


def mem_get_info(dev: Optional[torch.device] = None):
    if not _NPU_OK:
        _fail("mem_get_info")
    return torch_npu.npu.mem_get_info(dev)


# ---- graph capture --------------------------------------------------------


def is_current_stream_capturing() -> bool:
    """Stub: Ascend does not currently support CUDA-graph-style capture."""
    try:
        return torch_npu.npu.is_current_stream_capturing()
    except (AttributeError, RuntimeError):
        return False


# ---- init -----------------------------------------------------------------


def init():
    if not _NPU_OK:
        _fail("init")
    torch_npu.npu.init()


def set_device(dev: int):
    if not _NPU_OK:
        _fail("set_device")
    torch_npu.npu.set_device(dev)


# ---- helpers for legacy code patterns -------------------------------------


def optional_npu_guard(dev: torch.device):
    """Context-manager equivalent of CUDAGuard, no-op on Ascend."""
    # Ascend does not have a direct CUDAGuard equivalent; the device
    # is set through set_device and streams carry device affinity.
    import contextlib

    return contextlib.nullcontext()


def stream_handle(stream) -> int:
    """Return the raw stream handle (for progress/sync assertions)."""
    return getattr(stream, "npu_stream", 0)


# ---- env helpers ----------------------------------------------------------


def is_available() -> bool:
    return _NPU_OK


# Convenience: non_blocking copies between host and device use the same
# semantics on Ascend as on CUDA, so NO changes are needed to .to(dev,
# non_blocking=True) or .copy_(src, non_blocking=True) calls.