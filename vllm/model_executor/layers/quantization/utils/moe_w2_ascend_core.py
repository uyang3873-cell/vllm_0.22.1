import functools
import os
import re
import time
import ctypes
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger

from . import ascend_device
from .moe_w2_planes import (
    int8_per_row_to_codes_scales,
    pack_fragment_major,
    pack_fp4_fragment_major,
    pack_scales,
    reference_dequant,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_BLOCK = 32                      # tokens per pair == kernel M limit

# layer_key -> dict(N13, K13, N2, K2, E, base, tl_idx, ...)
_LAYERS: dict[int, dict] = {}
_WS: dict = {}                  # shared workspaces, sized lazily
_TRACE = int(os.getenv("VLLM_MOE_W2_DELTA_TRACE","0"))
# ---- adaptive expert top-p --------------------------------------------------
_TOPP = float(os.getenv("VLLM_MOE_W2_TOPP", "0"))
_TOPP_MIN = max(1, int(os.getenv("VLLM_MOE_W2_TOPP_MIN", "2")))
_TOPP_RENORM = os.getenv("VLLM_MOE_W2_TOPP_RENORM", "1") == "1"


def _apply_topp(topk_weights: torch.Tensor, topk_ids: torch.Tensor):
    """Drop each token's routed-weight tail past cumulative fraction _TOPP."""
    k = topk_ids.shape[1]
    if not (0.0 < _TOPP < 1.0) or k <= _TOPP_MIN:
        return topk_weights, topk_ids
    w = topk_weights.float()
    order = torch.argsort(w, dim=1, descending=True)
    w_sorted = w.gather(1, order)
    cum = torch.cumsum(w_sorted, dim=1)
    tot = cum[:, -1:]
    keep_sorted = (cum - w_sorted) < (_TOPP * tot)
    keep_sorted[:, :_TOPP_MIN] = True
    keep = torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)
    if _TOPP_RENORM:
        kept_sum = (w * keep).sum(dim=1, keepdim=True).clamp_min(1e-20)
        w = w * (tot / kept_sum)
    top1 = topk_ids.gather(1, order[:, :1])
    new_ids = torch.where(keep, topk_ids, top1.expand_as(topk_ids))
    new_w = torch.where(keep, w, torch.zeros_like(w)).to(topk_weights.dtype)
    return new_w, new_ids


def enabled() -> bool:
    return os.getenv("VLLM_MOE_W2", "0") == "1"


@functools.cache
def _layer_cutoff() -> int:
    """Main-stack layer count: layers >= this are the MTP drafter."""
    v = os.getenv("VLLM_MOE_W2_NUM_LAYERS")
    if v is not None:
        return int(v)
    try:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config().model_config.hf_config
        cfg = cfg.get_text_config()
        n = cfg.num_hidden_layers
        if n:
            return int(n)
    except Exception:
        pass
    return 43


def is_w2_layer(layer_name: str) -> bool:
    """Main-model routed experts only. MTP layers keep the stock path.

    Matches BOTH naming conventions:
      CUDA/Qwen:  model.layers.{N}.mlp.experts.{E}.{gate_proj,up_proj,down_proj}
      Ascend/DS4: layers.{N}.ffn.experts.{E}.{w1,w2,w3}
    """
    if not enabled():
        return False
    name = layer_name or ""
    if "mtp" in name.lower():
        return False
    m = re.search(r"(?:model\.)?layers\.(\d+)\.(?:mlp|ffn)\.experts", name)
    if m is None:
        return False
    return int(m.group(1)) < _layer_cutoff()


# ---- weight name adaptation (Ascend INT8 -> internal FusedMoE convention) ---

def _parse_int8_expert_params(layer, layer_name: str = ""):
    """Extract INT8 expert weights + scales + offsets from an Ascend-style layer.

    Target naming (Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp):
        layers.X.ffn.experts.Y.w1.weight       [N, K] int8  (gate_proj)
        layers.X.ffn.experts.Y.w1.weight_scale [N, 1] f32
        layers.X.ffn.experts.Y.w1.weight_offset[N, 1] f32
        layers.X.ffn.experts.Y.w2.weight       [N, K] int8  (down_proj)
        layers.X.ffn.experts.Y.w2.weight_scale [N, 1] f32
        layers.X.ffn.experts.Y.w2.weight_offset[N, 1] f32
        layers.X.ffn.experts.Y.w3.weight       [N, K] int8  (up_proj)
        layers.X.ffn.experts.Y.w3.weight_scale [N, 1] f32
        layers.X.ffn.experts.Y.w3.weight_offset[N, 1] f32

    Returns (w13_weight, w13_scale, w13_offset, w2_weight, w2_scale, w2_offset)
    where w13 = concat(w1, w3) along the N dimension (gate + up fused).
    Each is a list of per-expert tensors still on CPU.
    """
    # 将层内的所有参数名称和对象转换为字典，方便快速检索
    params = dict(layer.named_parameters())

    # =========================================================================
    # 新增逻辑：vLLM FusedMoE 格式 (预拼接/预融合格式)
    # 输入 pname 直接是 'w13_weight', 'w2_weight' 等
    # =========================================================================
    if "w13_weight" in params and "w2_weight" in params:
        w13_weight = params["w13_weight"].data
        w2_weight = params["w2_weight"].data

        # 张量的 shape 通常为 [E, N, K]，第 0 维即为专家数量 E
        E = w13_weight.shape[0]

        # 安全获取 scale 和 offset（兼容不同的命名后缀）
        def get_tensor(name_options):
            for name in name_options:
                if name in params:
                    return params[name].data
            return None

        w13_scale = get_tensor(["w13_weight_scale", "w13_scale"])
        w13_offset = get_tensor(["w13_weight_offset", "w13_offset"])
        w2_scale = get_tensor(["w2_weight_scale", "w2_scale"])
        w2_offset = get_tensor(["w2_weight_offset", "w2_offset"])

        return E, w13_weight, w13_scale, w13_offset, w2_weight, w2_scale, w2_offset

    # Collect per-expert tensors from layer attributes
    # The layer (RoutedExperts) has experts as submodules or flat params
    E = None
    w1_w, w1_s, w1_o = [], [], []
    w2_w, w2_s, w2_o = [], [], []
    w3_w, w3_s, w3_o = [], [], []

    # Try the flat-parameter path (vLLM FusedMoE quantized layer)
    for pname, param in layer.named_parameters():
        # Match: experts.{e}.w1.weight, experts.{e}.w1.weight_scale, etc.
        m = re.match(r"experts\.(\d+)\.(w[123])\.(weight|weight_scale|weight_offset)", pname)
        if m:
            e_idx = int(m.group(1))
            w_idx = m.group(2)
            kind = m.group(3)
            # Ensure lists are long enough
            while len(w1_w) <= e_idx:
                w1_w.append(None); w1_s.append(None); w1_o.append(None)
                w2_w.append(None); w2_s.append(None); w2_o.append(None)
                w3_w.append(None); w3_s.append(None); w3_o.append(None)
            if w_idx == "w1":
                if kind == "weight": w1_w[e_idx] = param.data
                elif kind == "weight_scale": w1_s[e_idx] = param.data
                elif kind == "weight_offset": w1_o[e_idx] = param.data
            elif w_idx == "w2":
                if kind == "weight": w2_w[e_idx] = param.data
                elif kind == "weight_scale": w2_s[e_idx] = param.data
                elif kind == "weight_offset": w2_o[e_idx] = param.data
            elif w_idx == "w3":
                if kind == "weight": w3_w[e_idx] = param.data
                elif kind == "weight_scale": w3_s[e_idx] = param.data
                elif kind == "weight_offset": w3_o[e_idx] = param.data

    if all(x is not None for x in w1_w):
        E = len(w1_w)
        # Stack per-expert tensors: [E, N, K] or [E, N, 1]
        w13_weight = torch.stack([torch.cat([w1_w[i], w3_w[i]], dim=0) for i in range(E)])
        w13_scale  = torch.stack([torch.cat([w1_s[i], w3_s[i]], dim=0) for i in range(E)])
        w13_offset = torch.stack([torch.cat([w1_o[i], w3_o[i]], dim=0) for i in range(E)])
        w2_weight  = torch.stack([w2_w[i] for i in range(E)])
        w2_scale   = torch.stack([w2_s[i] for i in range(E)])
        w2_offset  = torch.stack([w2_o[i] for i in range(E)])
        return E, w13_weight, w13_scale, w13_offset, w2_weight, w2_scale, w2_offset

    return None


# ---- load-time: INT8 -> 2-bit plane building ---------------------------------


def _fp4_tier_for_build(E: int, dev, n13k13: int, n2k2: int):
    """FP4 delta tier sized for this model's per-rank shapes."""
    from . import moe_w2_delta
    split = moe_w2_delta.split_enabled()
    sc13, sc2 = ((n13k13 // 32, n2k2 // 32)
                 if moe_w2_delta.base_enabled() and not split else (0, 0))
    div = 4 if split else 2
    return moe_w2_delta.get_tier(n_experts=E, dev=dev,
                                 w13_bytes=n13k13 // div + sc13,
                                 w2_bytes=n2k2 // div + sc2)


def _stage_fp4_host(tier, layer_key: int, fp13, sc13, fp2, sc2) -> None:
    """Stage a layer's FP4 planes into the tier's pinned host store."""
    from . import moe_w2_delta
    if moe_w2_delta.base_enabled() and not moe_w2_delta.split_enabled():
        tier.add_layer_host_sections(layer_key, (fp13, sc13), (fp2, sc2))
    else:
        tier.add_layer_host_planes(layer_key, fp13, fp2)


def _pack_fp4_plane(nib):
    """One expert's FP4-tier plane row from its e2m1 nibbles."""
    from . import moe_w2_delta
    if moe_w2_delta.split_enabled():
        from .moe_w2_planes import nibbles_to_refinement
        return pack_fragment_major(nibbles_to_refinement(nib))
    return pack_fp4_fragment_major(nib)


# Loader-level skip (boot-from-pack)
_n_created = 0
_skip_logged = False


def _noop_loader(*args, **kwargs):
    return True if kwargs.get("return_success") else None


def plan_pack_skip(layer) -> bool:
    """CREATE-time: probe stores; skip checkpoint staging for pack-resident layers."""
    from . import moe_w2_delta
    from . import moe_w2_planes_cache as _pc
    from .moe_w2_store import pack_has_layer
    global _n_created
    key = _n_created
    _n_created += 1
    layer._moe_w2_create_key = key
    if not enabled():
        return False
    try:
        result = _parse_int8_expert_params(layer, getattr(layer, "layer_name", ""))
        if result is None:
            return False
        E, w13_weight, w13_scale, w13_offset, w2_weight, w2_scale, w2_offset = result
        N13, K13 = w13_weight.shape[1], w13_weight.shape[2]
        N2, K2 = w2_weight.shape[1], w2_weight.shape[2]
    except Exception:
        return False
    c13len, s13len = N13 * K13 // 4, N13 * K13 // 32
    c2len, s2len = N2 * K2 // 4, N2 * K2 // 32
    if moe_w2_delta.base_enabled():
        n_keys = _layer_cutoff() + 1
        if not pack_has_layer("base", key, n_keys, E,
                              c13len + s13len + c2len + s2len):
            return False
        if moe_w2_delta.enabled():
            if moe_w2_delta.split_enabled():
                ftag, fslot = "fp4s", N13 * K13 // 4 + N2 * K2 // 4
            else:
                ftag = "fp4"
                fslot = (N13 * K13 // 2 + s13len) + (N2 * K2 // 2 + s2len)
            if not pack_has_layer(ftag, key, n_keys, E, fslot):
                return False
    else:
        # GPU-resident: probe planes cache
        lidx = _pc.layer_idx_from_name(getattr(layer, "layer_name", ""))
        if lidx is None or not _pc.cache_has_layer(
                lidx, _pc.expected_sizes(
                    E, N13, K13, N2, K2, want_fp4=moe_w2_delta.enabled())):
            return False
    for pname in ("w13_weight", "w13_weight_scale",
                  "w2_weight", "w2_weight_scale"):
        p = getattr(layer, pname)
        p.data = torch.empty(0, dtype=p.data.dtype, device="cpu")
        p.weight_loader = _noop_loader
    layer._moe_w2_shapes = (E, N13, K13, N2, K2)
    layer._moe_w2_pack_skip = True
    global _skip_logged
    if not _skip_logged:
        _skip_logged = True
        logger.info(
            "moe_w2 LOADER-SKIP armed: pack-resident expert layers are "
            "neither host-staged nor copied from the checkpoint "
            "(first: key %d)", key)
    return True

#add by slf 权重进行反量化后再fp8量化，scale采用128*128方式
def build_layer_planes_deepseek_raw_npu(layer, layer_key: int,
                                        scale_suffix: str = "weight_scale",
                                        offset_suffix: str = "weight_offset",
                                        target_dtype: torch.dtype = torch.int8) -> None:
    from . import moe_w2_delta
    from . import (
        moe_w2_planes_cache as _pc)

    # 1. 切换设备为 NPU
    dev = torch.device("npu")

    # 获取原始 CPU 数据
    # w13 = layer.w13_weight.data[:, :1024, :]
    w13 = layer.w13_weight.data
    # s13 = getattr(layer, f"w13_{scale_suffix}").data[:, :1024, :]
    s13 = getattr(layer, f"w13_{scale_suffix}").data
    # o13 = getattr(layer, f"w13_{offset_suffix}").data[:, :1024, :]
    o13 = getattr(layer, f"w13_{offset_suffix}").data

    # w2 = layer.w2_weight.data[:, :, :512]
    w2 = layer.w2_weight.data
    s2 = getattr(layer, f"w2_{scale_suffix}").data
    o2 = getattr(layer, f"w2_{offset_suffix}").data

    E, N13, K13 = w13.shape
    _, N2, K2 = w2.shape

    #scale采用128*128方式
    def asym_to_sym_quant_on_npu(w_i8, s_f32, o_f32, N, K, dtype, chunk_size=16):
        """
        🚀 终极完全体：128x128 Block 分块量化 + NaN 防御清洗
        """
        E_total = w_i8.shape[0]

        # 强制目标 Kernel 的 Block 大小为 128
        blk_size = 128
        N_blks, K_blks = N // blk_size, K // blk_size

        # 预分配 CPU 输出张量：
        # Weight 还原回扁平的 [N, K] 供矩阵乘
        w_sym_out = torch.empty((E_total, N, K), dtype=dtype, device='cpu')
        # Scale 完美保留 [32, 32] 的 2D 网格结构！
        s_sym_out = torch.empty((E_total, N_blks, K_blks), dtype=torch.float32, device='cpu')

        for i in range(0, E_total, chunk_size):
            end = min(i + chunk_size, E_total)

            w_chunk = w_i8[i:end].to(dev, non_blocking=True)
            s_chunk = s_f32[i:end].to(dev, non_blocking=True)
            o_chunk = o_f32[i:end].to(dev, non_blocking=True)

            # =========================================================
            # 🚨 【核心防波堤】：镇压原版权重里自带的 NaN 和越界垃圾
            # =========================================================
            s_chunk = torch.nan_to_num(s_chunk, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-100.0, 100.0)
            o_chunk = torch.nan_to_num(o_chunk, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-100.0, 100.0)

            # 2. 精准恢复到 FP32
            w_fp32 = w_chunk.float() * s_chunk + o_chunk
            w_fp32 = torch.nan_to_num(w_fp32, nan=0.0).clamp(-65500.0, 65500.0)

            # 3. 强制切分为 128x128 Blocks，形成 [chunk, 32, 32, 128, 128]
            w_fp32_blocks = w_fp32.view(-1, N_blks, blk_size, K_blks, blk_size).permute(0, 1, 3, 2, 4)

            # 4. 独立计算每个 128x128 Block 的绝对值最大值
            # 结果形状为: [chunk, 32, 32, 1, 1]
            abs_max = w_fp32_blocks.abs().amax(dim=(-2, -1), keepdim=True)

            # 5. 对称量化
            if dtype == torch.float8_e4m3fn:
                s_sym = (abs_max / 448.0).clamp(min=1e-12)
                w_temp_npu = (w_fp32_blocks / s_sym).contiguous()
                w_sym_cpu = w_temp_npu.cpu().to(torch.float8_e4m3fn)
            else:
                s_sym = (abs_max / 127.0).clamp(min=1e-12)
                w_temp_npu = (w_fp32_blocks / s_sym).round().clamp(-128, 127).contiguous()
                w_sym_cpu = w_temp_npu.cpu().to(torch.int8)

            # 6. 将量化后的 Weight 拼回 [N, K] 扁平结构
            w_sym_out[i:end] = w_sym_cpu.permute(0, 1, 3, 2, 4).reshape(end - i, N, K)

            # 7. 挤压掉 128 的空维度，最终 Scale 完美符合 [chunk, N_blks, K_blks] 即 [chunk, 32, 32]
            s_sym_out[i:end] = s_sym.cpu().squeeze(-1).squeeze(-1)

            # 清理显存
            del w_chunk, s_chunk, o_chunk, w_fp32, w_fp32_blocks, abs_max, s_sym, w_temp_npu

        return w_sym_out, s_sym_out


    # 在 NPU 上并行完成对称量化计算
    w13_sym_cpu, s13_sym_cpu = asym_to_sym_quant_on_npu(w13, s13, o13, N13, K13, target_dtype)
    w2_sym_cpu, s2_sym_cpu = asym_to_sym_quant_on_npu(w2, s2, o2, N2, K2, target_dtype)

    _tl = _pc.layer_idx_from_name(getattr(layer, "layer_name", ""))

    if moe_w2_delta.base_enabled():
        ##scale 128*128
        c13len, s13len = N13 * K13, (N13 // 128) * (K13 // 128) * 4
        c2len, s2len = N2 * K2, (N2 // 128) * (K2 // 128) * 4


        # 获取 base_tier，dev 传入 "npu"
        btier = moe_w2_delta.get_base_tier(
            _layer_cutoff() + 1, E, dev,
            w13_bytes=c13len + s13len, w2_bytes=c2len + s2len)

        # 因数据已在 CPU，无需再次 .cpu()，极度顺滑
        w13_bytes = w13_sym_cpu.contiguous().view(torch.uint8).reshape(E, -1)
        s13_bytes = s13_sym_cpu.contiguous().view(torch.uint8).reshape(E, -1)
        w2_bytes = w2_sym_cpu.contiguous().view(torch.uint8).reshape(E, -1)
        s2_bytes = s2_sym_cpu.contiguous().view(torch.uint8).reshape(E, -1)

        btier.add_layer_host_planes(
            layer_key,
            torch.cat((w13_bytes, s13_bytes), dim=1),
            torch.cat((w2_bytes, s2_bytes), dim=1))

        _LAYERS[layer_key] = dict(
            N13=N13, K13=K13, N2=N2, K2=K2, E=E, base=True, tl_idx=_tl,
            off_s13=c13len, off_c2=c13len + s13len,
            off_s2=c13len + s13len + c2len,
        )
    else:
        # GPU 直驻模式 (这里直接使用已经在 NPU 上的张量，省去了再次从 CPU .to(dev) 的过程)
        print("error @@@@")

    # 释放原始检查点内存，注意 stub 也换成 NPU 上的占位符
    stub = torch.empty(0, dtype=torch.uint8, device=dev)
    for name in ("w13_weight", f"w13_{scale_suffix}", f"w13_{offset_suffix}",
                 "w2_weight", f"w2_{scale_suffix}", f"w2_{offset_suffix}"):
        if hasattr(layer, name):
            layer.register_parameter(name, torch.nn.Parameter(stub, requires_grad=False))


def build_layer_planes_int8(layer, layer_key: int) -> None:
    """Quantize one FusedMoE layer's INT8 experts to 2-bit planes on NPU.

    Reads CPU-resident INT8 params (w1+w3 -> w13, w2 with per-row scale+offset),
    dequantizes to f64, re-quantizes via the sweep-validated 2-bit pipeline,
    packs fragment-major planes, then replaces the originals with empty stubs.

    When VLLM_MOE_W2_DIRECT_MODE=1, delegates to _build_layer_direct_int8
    which skips 2-bit requant and keeps INT8 weights directly on NPU.
    """
    if os.getenv("VLLM_MOE_W2_DIRECT_MODE", "0") == "1":
        return build_layer_planes_deepseek_raw_npu(layer, layer_key)

    dev = ascend_device.current_device()

    result = _parse_int8_expert_params(layer, getattr(layer, "layer_name", ""))
    if result is None:
        raise ValueError(
            f"moe_w2_ascend: layer {layer_key} has no INT8 expert params "
            f"(expected layers.X.ffn.experts.Y.{w1,w2,w3}.weight/scale/offset)")

    E, w13_cpu, w13_scale_cpu, w13_offset_cpu, \
        w2_cpu, w2_scale_cpu, w2_offset_cpu = result

    # w13 = concat(w1, w3): [E, 2*I, H] int8; w2: [E, H, I] int8
    N13, K13 = w13_cpu.shape[1], w13_cpu.shape[2]  # N13=2*I, K13=H
    N2, K2 = w2_cpu.shape[1], w2_cpu.shape[2]      # N2=H, K2=I

    from . import moe_w2_delta

    # Boot-from-pack: skip requant if stores already hold this layer
    if _try_skip_requant(layer, layer_key, E, N13, K13, N2, K2,
                         ("w13_weight", "w13_weight_scale",
                          "w2_weight", "w2_weight_scale")):
        return

    # Allocate GPU planes
    planes13 = torch.empty(E, N13 * K13 // 4, dtype=torch.uint8, device=dev)
    sc13 = torch.empty(E, N13 * K13 // 32, dtype=torch.uint8, device=dev)
    planes2 = torch.empty(E, N2 * K2 // 4, dtype=torch.uint8, device=dev)
    sc2 = torch.empty(E, N2 * K2 // 32, dtype=torch.uint8, device=dev)

    tier = _fp4_tier_for_build(E, dev, N13 * K13, N2 * K2)
    fp13 = fp2 = None
    if tier is not None:
        _div = 4 if moe_w2_delta.split_enabled() else 2
        fp13 = torch.empty(E, N13 * K13 // _div, dtype=torch.uint8, device=dev)
        fp2 = torch.empty(E, N2 * K2 // _div, dtype=torch.uint8, device=dev)

    # Process in chunks: dequant INT8 -> f64 -> requant -> pack
    chunk = 8
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        wg13 = w13_cpu[e0:e1].to(dev, non_blocking=True)
        sg13 = w13_scale_cpu[e0:e1].to(dev, non_blocking=True)
        og13 = w13_offset_cpu[e0:e1].to(dev, non_blocking=True)
        for i in range(e1 - e0):
            codes, sbytes, nib = int8_per_row_to_codes_scales(
                wg13[i], sg13[i], og13[i], want_nibbles=fp13 is not None)
            planes13[e0 + i] = pack_fragment_major(codes)
            sc13[e0 + i] = pack_scales(sbytes)
            if fp13 is not None:
                fp13[e0 + i] = _pack_fp4_plane(nib)

        wg2 = w2_cpu[e0:e1].to(dev, non_blocking=True)
        sg2 = w2_scale_cpu[e0:e1].to(dev, non_blocking=True)
        og2 = w2_offset_cpu[e0:e1].to(dev, non_blocking=True)
        for i in range(e1 - e0):
            codes, sbytes, nib = int8_per_row_to_codes_scales(
                wg2[i], sg2[i], og2[i], want_nibbles=fp2 is not None)
            planes2[e0 + i] = pack_fragment_major(codes)
            sc2[e0 + i] = pack_scales(sbytes)
            if fp2 is not None:
                fp2[e0 + i] = _pack_fp4_plane(nib)

    if tier is not None:
        _stage_fp4_host(tier, layer_key, fp13, sc13, fp2, sc2)
        del fp13, fp2

    _finish_layer(layer, layer_key, dev, planes13, sc13, planes2, sc2,
                  N13, K13, N2, K2, E,
                  ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"))


def _try_skip_requant(layer, layer_key: int, E: int, N13: int, K13: int,
                      N2: int, K2: int, param_names) -> bool:
    """Boot-from-pack: skip requant when stores already hold this layer."""
    from . import moe_w2_delta
    if not moe_w2_delta.base_enabled():
        return False
    dev = ascend_device.current_device()
    c13len, s13len = N13 * K13 // 4, N13 * K13 // 32
    c2len, s2len = N2 * K2 // 4, N2 * K2 // 32
    btier = moe_w2_delta.get_base_tier(
        _layer_cutoff() + 1, E, dev,
        w13_bytes=c13len + s13len, w2_bytes=c2len + s2len)
    if layer_key not in btier._store:
        return False
    tier = _fp4_tier_for_build(E, dev, N13 * K13, N2 * K2)
    if tier is not None and layer_key not in tier._store:
        return False
    from . import moe_w2_planes_cache as _pc
    _LAYERS[layer_key] = dict(
        N13=N13, K13=K13, N2=N2, K2=K2, E=E, base=True,
        tl_idx=_pc.layer_idx_from_name(getattr(layer, "layer_name", "")),
        off_s13=c13len, off_c2=c13len + s13len,
        off_s2=c13len + s13len + c2len,
        off4_s13=2 * c13len, off4_c2=2 * c13len + s13len,
        off4_s2=2 * c13len + s13len + 2 * c2len,
    )
    stub = torch.empty(0, dtype=torch.uint8, device=dev)
    for name in param_names:
        layer.register_parameter(
            name, torch.nn.Parameter(stub, requires_grad=False))
    logger.info(
        "moe_w2: layer %d requant SKIPPED — serving from pack (boot-from-pack)",
        layer_key)
    return True


def _finish_layer(layer, layer_key, dev, planes13, sc13, planes2, sc2,
                  N13, K13, N2, K2, E, param_names) -> None:
    """Register built planes in _LAYERS and replace checkpoint params."""
    from . import moe_w2_delta
    from . import moe_w2_planes_cache as _pc
    _tl = _pc.layer_idx_from_name(getattr(layer, "layer_name", ""))
    if moe_w2_delta.base_enabled():
        c13len, s13len = planes13.shape[1], sc13.shape[1]
        c2len, s2len = planes2.shape[1], sc2.shape[1]
        btier = moe_w2_delta.get_base_tier(
            _layer_cutoff() + 1, E, dev,
            w13_bytes=c13len + s13len, w2_bytes=c2len + s2len)
        btier.add_layer_host_planes(
            layer_key,
            torch.cat((planes13, sc13), dim=1),
            torch.cat((planes2, sc2), dim=1))
        _LAYERS[layer_key] = dict(
            N13=N13, K13=K13, N2=N2, K2=K2, E=E, base=True, tl_idx=_tl,
            off_s13=c13len, off_c2=c13len + s13len,
            off_s2=c13len + s13len + c2len,
            off4_s13=2 * c13len, off4_c2=2 * c13len + s13len,
            off4_s2=2 * c13len + s13len + 2 * c2len,
        )
        del planes13, sc13, planes2, sc2
        stub = torch.empty(0, dtype=torch.uint8, device=dev)
        for name in param_names:
            layer.register_parameter(
                name, torch.nn.Parameter(stub, requires_grad=False))
        logger.info("moe_w2: layer %d planes HOST-staged (base cache, "
                    "%.2f GiB pinned)", layer_key,
                    E * btier.slot_bytes / 2**30)
        return

    _LAYERS[layer_key] = dict(
        planes13=planes13, sc13=sc13, planes2=planes2, sc2=sc2,
        N13=N13, K13=K13, N2=N2, K2=K2, E=E, tl_idx=_tl,
    )
    stub = torch.empty(0, dtype=torch.uint8, device=dev)
    for name in param_names:
        layer.register_parameter(
            name, torch.nn.Parameter(stub, requires_grad=False))
    logger.info("moe_w2: layer %d planes built (%.2f GiB)", layer_key,
                (planes13.nbytes + sc13.nbytes + planes2.nbytes + sc2.nbytes)
                / 2**30)


# --------------------------------------------------------------------------
# Forward — Ascend CANN (on-the-fly dequant + torch.matmul)
# --------------------------------------------------------------------------

def _dequant_expert_2bit(codes: torch.Tensor, sc_bytes: torch.Tensor,
                          N: int, K: int, out_dtype: torch.dtype = torch.float16
                          ) -> torch.Tensor:
    """Dequantize one expert's 2-bit planes to dense fp16 weight matrix.

    Args:
        codes:    [N*K/4] u8 fragment-major packed 2-bit codes
        sc_bytes: [N*K/32] u8 UE8M0 scale bytes (packed layout)
        N, K:     weight matrix dimensions
        out_dtype: torch.float16 or torch.bfloat16

    Returns:
        [N, K] out_dtype dequantized weight matrix
    """
    # Unpack fragment-major -> [N, K] codes
    # Layout: for each N/16 nb, K/64 kb, 8 lane g, 4 lane t:
    #   8 bytes with 4x 2-bit codes each -> 16 codes per lane per k64 block
    #   N/16 * 16 rows = N rows total
    # Matches pack_fragment_major / reference_dequant
    assert N % 16 == 0 and K % 64 == 0
    assert codes.shape == (N * K // 4,)
    assert sc_bytes.shape == (N * K // 32,)

    # Unpack 2-bit codes
    c = codes.view(torch.uint8)
    # Each byte = 4 codes [c0|c1|c2|c3] little-endian
    c0 = c & 0x3
    c1 = (c >> 2) & 0x3
    c2 = (c >> 4) & 0x3
    c3 = (c >> 6) & 0x3
    all_codes = torch.stack([c0, c1, c2, c3], dim=-1).flatten().to(torch.int64)

    # Reshape from fragment-major layout
    # After flatten: N*K codes
    # Fragment-major order: [nb, kb, g, t, tile, k32, half, k4]
    # We undo the permutation to get row-major [N, K]
    all_codes = all_codes.view(N // 16, K // 64, 8, 4, 2, 2, 2, 4)
    # Reverse permute: [nb, tile, g, kb, k32, half, t, k4]
    #                  -> [nb, kb, g, t, tile, k32, half, k4] (packed order)
    # We need: [nb, tile, g, kb, k32, half, t, k4] (row-major order)
    # Where row = nb*16 + tile*8 + g, col = kb*64 + k32*32 + half*16 + t*4 + k4
    all_codes = all_codes.permute(0, 4, 2, 1, 5, 6, 3, 7).contiguous()
    all_codes = all_codes.view(N, K)

    # Code -> value (tensor-sym codebook)
    levels = torch.tensor([-4.0, -1.0, 1.0, 4.0], device=codes.device,
                          dtype=torch.float32)
    vals = levels[all_codes]

    # Unpack scales (same fragment-major layout but [N, K/32])
    sc = sc_bytes.float()
    # Reverse the pack_scales layout
    KS = K // 32
    sc = sc.view(N // 16, KS, 16).transpose(1, 2).contiguous().view(N, KS)
    sc = torch.exp2(sc - 127.0).repeat_interleave(32, dim=-1)

    return (vals * sc).to(out_dtype)

def _workspaces(slots: int, tokens: int, dev, inter: int = 2048,
                hidden: int = 4096, n_experts: int = 256) -> dict:
    # `inter` = per-rank expert intermediate size I (2048 on 1 GPU; 1024 @ TP2,
    # 512 @ TP4 as the experts shard). The hidden H (4096 DS4, 6144 GLM-5.x) is
    # NOT sharded, so the A-side (a1), x-quant (xq) and w2 output (c2) buffers
    # stay H-wide; only the gate/up output (c13 = 2I), the intermediate
    # activation (act/a2 = I) and its group-128 scales (as2 = I/128) follow the
    # shard.
    if (_WS.get("slots", 0) < slots or _WS.get("tokens", 0) < tokens
            or _WS.get("inter") != inter or _WS.get("hidden") != hidden
            or _WS.get("n_experts", 0) < n_experts):
        slots = max(slots, _WS.get("slots", 0))
        tokens = max(tokens, _WS.get("tokens", 0))
        n_experts = max(n_experts, _WS.get("n_experts", 0))
        _WS.update(
            slots=slots,
            tokens=tokens,
            inter=inter,
            hidden=hidden,
            n_experts=n_experts,
            # token-side quant buffers; the LAST row is the permanent zero
            # pad row (gather source for filler slots) — quant only ever
            # writes rows [:T].
            # xq=torch.zeros(tokens + 1, hidden, dtype=torch.float8_e4m3fn,
            #                device=dev),
            xq=torch.zeros(tokens + 1, hidden, dtype=torch.int8,
                           device=dev),
            xs=torch.zeros(tokens + 1, hidden // 128, dtype=torch.float32,
                           device=dev),
            # a1=torch.zeros(slots + 4, hidden, dtype=torch.float8_e4m3fn,
            #                device=dev),
            a1=torch.zeros(slots + 4, hidden, dtype=torch.int8,
                           device=dev),
            as1=torch.zeros(slots + 4, hidden // 128, dtype=torch.float32,
                            device=dev),
            # zeros, not empty: pad-pair rows are never written by the kernel
            # (early EXIT) yet flow through silu/scatter math with weight 0;
            # uninitialized inf/nan would poison 0*x.
            c13=torch.zeros(slots + 4, 2 * inter, dtype=torch.bfloat16,
                            device=dev),
            act=torch.zeros(slots + 4, inter, dtype=torch.bfloat16, device=dev),
            # a2=torch.zeros(slots + 4, inter, dtype=torch.float8_e4m3fn,
            #                device=dev),
            a2=torch.zeros(slots + 4, inter, dtype=torch.int8,
                           device=dev),
            as2=torch.zeros(slots + 4, max(inter // 128, 1),
                            dtype=torch.float32, device=dev),
            c2=torch.zeros(slots + 4, hidden, dtype=torch.bfloat16,
                           device=dev),
            desc=torch.empty(4, slots // _BLOCK, 6, dtype=torch.int64,
                             device=dev),
            # split-FP4 (moe_w4q_mm) desc tables: 8 u64 per pair, 64 B ABI
            desc4s=torch.empty(2, slots // _BLOCK, 8, dtype=torch.int64,
                               device=dev),
            # -1 slot row for the tier-less desc path; sized to the MODEL's
            # expert count (256 = DS4 default; 384 Kimi-K2.x reads past a
            # fixed 256-row table).
            no_slots=torch.full((max(n_experts, 256),), -1,
                                dtype=torch.int32, device=dev),
        )

    return _WS



import collections
import torch
import torch_npu


def _moe_align_block_size(
        x: torch.Tensor,  # NPU 官方算子需要特征张量 x 占位
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
        expert_map: torch.Tensor | None = None,
        pad_sorted_ids: bool = False,
        ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = topk_ids.device
    N = topk_ids.numel()

    # 1. 过滤非法专家，将其映射为 num_experts 垃圾桶 (或者直接传入)
    clamped_ids = torch.clamp(topk_ids, min=0, max=num_experts)


    # x 可以是真正的激活值，这样 NPU 会顺便帮你把特征也排好序！
    expanded_x, expanded_row_idx, counts, _ = torch_npu.npu_moe_init_routing_v2(
        x,
        clamped_ids.to(torch.int32),
        expert_num=num_experts + 1,
        drop_pad_mode=0,  # 不启用官方的固定 Capacity Padding
        expert_tokens_num_type=1,  # 给我每个专家的 Token Count
        expert_tokens_num_flag=True,  # 必须输出
        quant_mode=-1,  # 不量化
        row_idx_type=0  # 输出 gather 索引
    )

    # 此时，expanded_row_idx 已经是完美排序后的 Token 索引了！
    # counts 已经是每个专家实际分配到的 Token 数量了！
    valid_counts = counts[:num_experts].to(torch.int32)

    # =====================================================================
    # 🧩 缝合 vLLM 的 Block Padding 逻辑
    # =====================================================================
    # 2. 计算 vLLM 专属的 Padding 边界
    padded_counts = ((valid_counts + block_size - 1) // block_size) * block_size

    zeros = torch.zeros((1,), dtype=torch.int32, device=dev)
    counts_cumsum = torch.cat([zeros, torch.cumsum(valid_counts, dim=0, dtype=torch.int32)])
    padded_cumsum = torch.cat([zeros, torch.cumsum(padded_counts, dim=0, dtype=torch.int32)])
    total_padded_tokens = padded_cumsum[-1].unsqueeze(0)

    # 3. 计算静态边界
    max_padded_tokens_bound = N + num_experts * (block_size - 1)
    max_blocks = (max_padded_tokens_bound + block_size - 1) // block_size
    max_padded_tokens = max_blocks * block_size

    # 4. 把官方算子排好序的紧凑 IDs，散布到 vLLM 的 Padded IDs 里
    # 这一步因为去掉了全局 Sort，可以直接用简单的 Triton Kernel
    # 或者如果张量不大，直接用 PyTorch 原生算子
    out_sorted_ids = torch.full((max_padded_tokens,), N, dtype=torch.int32, device=dev)


    # 生成 0 到 N-1 的固定序号，大小静态已知，绝不触发同步
    token_idx = torch.arange(N, device=dev, dtype=torch.int32)

    # 用查表法得出每个排好序的 Token 属于哪个专家
    expert_for_each_token = torch.bucketize(token_idx, counts_cumsum, right=False) - 1

    rank_in_expert = token_idx - counts_cumsum[expert_for_each_token]
    out_pos = padded_cumsum[expert_for_each_token] + rank_in_expert

    out_sorted_ids.scatter_(0, out_pos.long(), expanded_row_idx.to(torch.int32))

    # 5. 生成 expert_ids (通过 blocks_cumsum 直接 bucketize)
    block_idx = torch.arange(max_blocks, device=dev, dtype=torch.int32)
    blocks_cumsum = padded_cumsum // block_size
    expert_ids = torch.bucketize(block_idx, blocks_cumsum, right=True) - 1
    expert_ids = torch.clamp(expert_ids, min=0, max=num_experts - 1).to(torch.int32)

    # 动态切片截断
    is_capturing = torch.npu.is_current_stream_capturing() if hasattr(torch, "npu") else False
    if not pad_sorted_ids and not is_capturing:
        actual_padded = total_padded_tokens.item()
        out_sorted_ids = out_sorted_ids[:actual_padded]
        expert_ids = expert_ids[:actual_padded // block_size]

    return out_sorted_ids, expert_ids, total_padded_tokens

import torch
def _per_token_group_quant_int8(
        x: torch.Tensor,
        group_size: int,
        out_q: torch.Tensor = None,
        eps: float = 1e-10
):
    """
    Ascend NPU 适配：纯 PyTorch 实现的 INT8 分组对称量化算子。
    将激活值映射到 [-127, 127] 区间，完美适配 910B 硬件 Cube 单元。
    """
    M, K = x.shape
    x_group = x.view(M, K // group_size, group_size)

    # 推荐以 FP32 计算 scale 和 rounding，保证精度不丢失
    x_group_f32 = x_group.to(torch.float32)

    # 1. 寻找每组的绝对值最大值 (amax)
    amax = x_group_f32.abs().max(dim=-1, keepdim=True)[0]
    amax = torch.clamp(amax, min=eps)

    # 2. 计算 Scale (INT8 的安全动态范围我们取对称的 127.0)
    INT8_MAX = 127.0
    scale = amax / INT8_MAX

    # 3. 缩放并四舍五入 (Rounding)
    x_scaled = torch.round(x_group_f32 / scale)

    # 4. 截断 (Clamp) 防止溢出，并强转为原生的 int8 类型
    # 范围限制在 [-128, 127] 内
    x_int8 = torch.clamp(x_scaled, min=-128.0, max=127.0).to(torch.int8)

    # 重新展平回 [M, K] 的物理形状
    x_int8 = x_int8.view(M, K)

    # 5. 结果输出
    if out_q is not None:
        # 因为 int8 是 NPU 原生支持的类型，底层 aclnnInplaceCopy 绝不会报错
        # 直接物理显存对拷即可，无需披 uint8 马甲！
        out_q.copy_(x_int8)
        res_q = out_q
    else:
        res_q = x_int8

    # 挤压掉最后一个维度，Scale 的形状变为 [M, K // group_size]
    res_s = scale.squeeze(-1)

    return res_q, res_s


# =====================================================================
import triton
import triton.language as tl
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_scale_scatter_add_kernel(
        c2_ptr, w_ptr, sorted_ids_ptr, out_ptr,
        slots, T, top_k,
        stride_c2_s, stride_out_t,
        H, BLOCK_H: tl.constexpr
):
    pid_s = tl.program_id(0)

    num_programs = tl.num_programs(0)

    for s_idx in range(pid_s, slots, num_programs):
        # 1. 拿权重
        w = tl.load(w_ptr + s_idx).to(tl.float32)

        #  优化：去除 continue，改为用 if w != 0.0 包裹整个执行块
        if w != 0.0:
            # 2. 算真实的 Token 索引
            sorted_id = tl.load(sorted_ids_ptr + s_idx)

            # 再嵌套一个 if 替代原本的 continue 越界检查
            if sorted_id < T * top_k:
                token_idx = sorted_id // top_k

                # 3. 处理 H 维度
                for h_start in range(0, H, BLOCK_H):
                    offs_h = h_start + tl.arange(0, BLOCK_H)
                    mask_h = offs_h < H

                    # 读取 C2 的值并乘上权重
                    c2_val = tl.load(c2_ptr + s_idx * stride_c2_s + offs_h, mask=mask_h, other=0.0)
                    res = c2_val * w

                    # 原子相加到最终矩阵
                    out_ptrs = out_ptr + token_idx * stride_out_t + offs_h
                    tl.atomic_add(out_ptrs, res, mask=mask_h)


# =====================================================================
# 主函数
# =====================================================================
def _moe_w2_forward_direct(
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        layer_key: int,
) -> torch.Tensor:
    import gc
    import os
    import sys
    import ctypes
    from vllm.model_executor.layers.quantization.utils.ascend import moe_w2_looka
    from vllm.model_executor.layers.quantization.utils.ascend import moe_w2_delta

    is_debug = "pydevd" in sys.modules or os.environ.get("IDE_DEBUG") == "1"

    if is_debug:
        gc.disable()
        print(f"🛡️ [DEBUG 模式已检测] LAYER {layer_key} 已临时挂起 GC 防止 IDE 后台崩溃。")

    st = _LAYERS[layer_key]
    T, H = x.shape

    # --- 阶段一：动态路由与内存对齐 ---
    topk_weights, topk_ids = _apply_topp(topk_weights, topk_ids)
    top_k = topk_ids.shape[1]
    dev = x.device
    stream = ctypes.c_void_p(torch.npu.current_stream(dev).npu_stream)
    prefill = T > 9
    mblock = 32 if prefill else _BLOCK

    sorted_ids, expert_blocks, num_post = _moe_align_block_size(x,topk_ids, mblock, st["E"])

    slots = sorted_ids.numel()
    pairs = slots // mblock

    ws = _workspaces(slots, T, dev, inter=st["K2"], hidden=st["K13"], n_experts=st["E"])

    # --- 阶段二：激活值 FP8 量化与重排 ---
    xq = ws["xq"]
    pad_row = xq.shape[0] - 1
    _, xs = _per_token_group_quant_int8(x, 128, out_q=xq[:T])
    ws["xs"][:T] = xs
    valid = sorted_ids < T * top_k
    rows = torch.where(valid, sorted_ids // top_k, torch.full_like(sorted_ids, pad_row))

    torch.index_select(xq, 0, rows, out=ws["a1"][:slots])
    torch.index_select(ws["xs"], 0, rows, out=ws["as1"][:slots])

    valid_mask_int8 = valid.unsqueeze(-1).to(torch.int8)
    valid_mask_fp32 = valid.unsqueeze(-1).to(torch.float32)

    ws["a1"][:slots].mul_(valid_mask_int8)
    ws["as1"][:slots].mul_(valid_mask_fp32)

    a1_base = ws["a1"]
    a2_base = ws["a2"]
    d = ws["desc"]
    cap = d.shape[1]

    btier = moe_w2_delta._BASE_TIER

    if torch.npu.is_current_stream_capturing():
        btier.notify_capture()
    elif prefill:
        btier.ensure_resident(layer_key, topk_ids.view(-1))

    moe_w2_delta.mark_seen(btier.seen[layer_key], topk_ids.view(-1).long())

    if not prefill:
        if moe_w2_looka.enabled():
            moe_w2_looka.record(layer_key, x, topk_ids, btier.route_log)

        if btier.route_log is not None:
            _t = min(topk_ids.shape[0], btier.route_log.shape[1])
            _k = min(topk_ids.shape[1], btier.route_log.shape[2])
            btier.route_log[layer_key, :_t, :_k].copy_(topk_ids[:_t, :_k], non_blocking=True)

    if layer_key == 0:
        btier.miss_count.zero_()

    slot_row = btier.slot_table[layer_key]

    off_w13 = 0
    off_s13 = st["N13"] * st["K13"]
    off_w2 = off_s13 + (st["N13"] // 128) * (st["K13"] // 128) * 4
    off_s2 = off_w2 + st["N2"] * st["K2"]

    a1_rb = st["K13"]
    as1_rb = (st["K13"] // 128) * 4
    c13_rb = st["K2"] * 4
    a2_rb = st["K2"]
    as2_rb = (st["K2"] // 128) * 4
    c2_rb = st["K13"] * 2

    debug_i64 = torch.zeros((4, cap), dtype=torch.int64, device=dev)

    _desc_build_kernel_w8[(triton.cdiv(pairs, 256),)](
        expert_blocks, num_post, slot_row, btier.miss_count, d,
        a1_base.data_ptr(), ws["as1"].data_ptr(), ws["c13"].data_ptr(),
        a2_base.data_ptr(), ws["as2"].data_ptr(), ws["c2"].data_ptr(),
        btier.pool.data_ptr(), btier.slot_bytes,
        off_w13, off_s13, off_w2, off_s2,
        a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
        st["E"], pairs, cap * 6, mblock,
        debug_i64, cap,
        BLOCK=256
    )

    e_pair = expert_blocks.to(torch.int32).clamp_(0, st["E"] - 1)
    resident = (slot_row[e_pair] >= 0)
    token_to_block_idx = torch.arange(slots, device=dev) // mblock
    token_resident = resident[token_to_block_idx]

    w8tier = "w8"
    STATIC_MAX_M_CAPACITY = 32


    ws["c13"][:slots].zero_()
    ws["c2"][:slots].zero_()

    # --- 阶段四：纯 FP8 GEMM 计算流 ---
    _launch_int8_triton(w8tier, st["K13"], d[0], st["N13"], pairs, stream, STATIC_MAX_M_CAPACITY)

    c13_out = ws["c13"][:slots]
    gate, up = c13_out.chunk(2, dim=-1)
    act = F.silu(gate) * up

    ws["act"][:slots] = act
    _, qs2 = _per_token_group_quant_int8(act, 128, out_q=ws["a2"][:slots])
    ws["as2"][:slots] = qs2

    _launch_int8_triton(w8tier, st["K2"], d[1], st["N2"], pairs, stream, STATIC_MAX_M_CAPACITY)

    # 1. 提取权重，用 token_resident 掩码把 Cache Miss 的权重当场化为 0
    w = topk_weights.reshape(-1)[sorted_ids.clamp(max=T * top_k - 1)]
    w = torch.where(valid, w, torch.zeros_like(w)).to(torch.float32)
    w = w * token_resident.to(torch.float32)

    # 2. 申请干干净净的 [T, H] 最终输出 (由于 Triton 会原子加，必须初始化为 0)
    out_fp32 = torch.zeros((T, H), dtype=torch.float32, device=dev)

    # 3. 发射终极还原内核！
    BLOCK_H = 256


    # 我们发给 NPU 的线程块数量永远不超过 65535 的硬件极限！
    # 如果 slots 比 65535 大，Triton 内部的 for 循环会自动吃掉多余的部分。
    safe_grid_size = min(slots, 65000)

    # 传入一维 Grid
    grid = (safe_grid_size,)

    _fused_scale_scatter_add_kernel[grid](
        ws["c2"], w, sorted_ids, out_fp32,
        slots, T, top_k,
        ws["c2"].stride(0), out_fp32.stride(0),
        H, BLOCK_H=BLOCK_H
    )

    # 最后转回原精度输出
    out = out_fp32.to(x.dtype)
    torch.npu.set_device(dev)

    return out

def _launch_int8_triton(tier, K: int, desc: torch.Tensor, N: int, pairs: int, stream=None,max_m: int = None):

    if pairs == 0:
        return
    if max_m is None:
        max_m = 8192 # 静态常量上限，保证 Grid 尺寸在 CUDA Graph 捕获时完全固定
    if max_m <= 0:
        return
    GROUP_SIZE = 128
    BLOCK_M = 32
    BLOCK_N = 128
    BLOCK_K = 128

    def grid(meta):
        return (
            pairs,
            triton.cdiv(N, meta['BLOCK_N']),
            triton.cdiv(max_m, meta['BLOCK_M'])
        )
    #  1. 在 NPU 上开辟一块全零的 FP32 探针内存 (足够大)
    debug_tensor = torch.zeros(64, dtype=torch.float32, device=desc.device)
    def _run():
        desc_contiguous = desc.contiguous()
        # 注意：建议将内核函数统一命名为 _moe_fp8_gemm_kernel_fixed
        _moe_int8_w13_gemm_kernel_fixed[grid](
            desc_contiguous,
            debug_tensor,
            K=K,
            N=N,
            GROUP_SIZE=GROUP_SIZE,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K
        )

    # 1. 尝试直接使用，或者获取当前的设备
    current_dev = torch.npu.current_device()
    active_stream_obj = None

    if stream is None:
        active_stream_obj = torch.npu.current_stream()
    elif isinstance(stream, torch.npu.Stream):
        active_stream_obj = stream
    else:
        # 如果是一个底层的指针/整数，在 torch_npu 中，我们不能直接从指针反向构造 Stream 对象。
        # 最安全的做法是：直接信任主调用线程，强制使用当前流！
        # 因为我们之前在 _moe_w2_forward_direct 外部取的也是 current_stream。
        active_stream_obj = torch.npu.current_stream(current_dev)

    # 2. 必须使用上下文管理器紧紧包裹住 _run()
    with torch.npu.stream(active_stream_obj):
        _run()



from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult
def _moe_w2_forward_ascend(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_key: int,
) -> FusedExpertsResult:
    """Ascend CANN MoE forward: group-by-expert, dequant weights, torch.matmul.

    Replaces the CUDA cubin GEMM path. For each unique expert in the batch:
    1. Gather tokens routed to that expert
    2. Dequantize the expert's 2-bit planes to fp16
    3. Run gate/up GEMM (w13) -> silu_and_mul -> down GEMM (w2)
    4. Scatter results back

    This is a CORRECTNESS BASELINE. The user will replace torch.matmul with
    custom Ascend 2-bit GEMM operators for production performance.

    When VLLM_MOE_W2_DIRECT_MODE=1, dispatches to _moe_w2_forward_direct
    which uses INT8 per-row dequant instead of 2-bit dequant.
    """
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    st = _LAYERS[layer_key]

    if os.getenv("VLLM_MOE_W2_DIRECT_MODE", "0") == "1":
        routed_out = _moe_w2_forward_direct(x, topk_weights, topk_ids, layer_key)
        # ===================================================================
        # 【新增】：构造 FusedExpertsResult 以满足外层 DeepSeek 共享专家逻辑
        # ===================================================================

        import torch.npu

        # 因为你现在的算子是端到端全包圆的（没有拆分 dispatch 和 combine 阶段），
        # 但外层的共享专家可能依赖这几个时间戳来进行异步同步，
        # 所以我们在这里打下几个 Dummy (占位) 或真实的 NPU 时间戳。
        current_stream = torch.npu.current_stream()
        before_dispatch_evt = current_stream.record_event()
        before_gmm2_evt = current_stream.record_event()
        before_combine_evt = current_stream.record_event()

        # 返回外层期望的数据结构
        return FusedExpertsResult(
            routed_out=routed_out,
            before_dispatch_evt=before_dispatch_evt,
            before_gmm2_evt=before_gmm2_evt,
            before_combine_evt=before_combine_evt,
            # 下面这两个参数是为了保持结构体完整性，你直接填空值或默认值即可，
            # 你的端到端 Triton 算子不需要这些内部元数据
            group_list_type=1,
            expert_tokens=None,
            swiglu_limit=0.0
        )
    else:
        logger.error("_moe_w2_forward_ascend should set VLLM_MOE_W2_DIRECT_MODE=1")



def moe_w2_forward(x, topk_weights, topk_ids, layer_key) -> torch.Tensor:
    """Entry point called from quantization method apply()."""
    from . import prefill_timers
    with prefill_timers.span("moe_w2_ascend"):
        return _moe_w2_forward_ascend(x, topk_weights, topk_ids, layer_key)


def ready() -> bool:
    """True if at least one layer's planes are registered."""
    return len(_LAYERS) > 0

# --------------------------------------------------------------------------
# Streaming build (load-time optimization, pure Python — no CUDA deps)
# --------------------------------------------------------------------------

_STREAM = os.getenv("VLLM_MOE_W2_STREAM_BUILD", "1") == "1"
_stream_logged = False


class _StreamLoader:
    """Per-param weight_loader wrapper: triggers layer build when complete."""

    def __init__(self, layer, pname, inner):
        self._layer = layer
        self._pname = pname
        self._inner = inner

    def __call__(self, param, loaded_weight, *args, **kwargs):
        if param.data.numel() == 0:
            shape = self._layer._moe_w2_stream_shapes[self._pname]
            param.data = torch.empty(shape, dtype=param.data.dtype,
                                     device="cpu")
        ret = self._inner(param, loaded_weight, *args, **kwargs)
        ok = (ret is True) if kwargs.get("return_success") else True
        if not ok:
            return ret
        pend = self._layer._moe_w2_pending
        pend[self._pname] -= 1
        if all(v == 0 for v in pend.values()):
            key = self._layer._moe_w2_create_key
            build_layer_planes_int8(self._layer, key)
            for p in self._layer._moe_w2_stream_orig:
                p.data = torch.empty(0, dtype=p.data.dtype, device="cpu")
            self._layer._moe_w2_stream_orig = ()
            self._layer._moe_w2_stream_built = True
            logger.debug("moe_w2: layer key %d stream-built during load", key)
        return ret


def arm_stream_build(layer) -> bool:
    """Arm streaming per-layer build for an INT8 layer (first boot)."""
    global _stream_logged
    if not (_STREAM and enabled()):
        return False
    try:
        result = _parse_int8_expert_params(layer, getattr(layer, "layer_name", ""))
        if result is None:
            return False
        E = result[0]
    except Exception:
        return False
    expected = {"w13_weight": 2 * E, "w13_weight_scale": 2 * E,
                "w13_weight_offset": 2 * E,
                "w2_weight": E, "w2_weight_scale": E,
                "w2_weight_offset": E}
    big = ("w13_weight", "w13_weight_scale", "w13_weight_offset",
           "w2_weight", "w2_weight_scale", "w2_weight_offset")
    wrappers = {}
    for pname in expected:
        p = getattr(layer, pname, None)
        inner = getattr(p, "weight_loader", None)
        if p is None or inner is None:
            return False
        wrappers[pname] = (p, _StreamLoader(layer, pname, inner))
    layer._moe_w2_pending = expected
    layer._moe_w2_stream_orig = tuple(getattr(layer, p) for p in big)
    layer._moe_w2_stream_shapes = {
        p: tuple(getattr(layer, p).shape) for p in big}
    for pname in big:
        p = getattr(layer, pname)
        p.data = torch.empty(0, dtype=p.data.dtype, device="cpu")
    for pname, (p, wrap) in wrappers.items():
        p.weight_loader = wrap
    if not _stream_logged:
        _stream_logged = True
        logger.info(
            "moe_w2 STREAM-BUILD armed: layer staging materializes on its "
            "first loaded tensor and requants on its last")
    return True

import triton
import triton.language as tl


@triton.jit
def _desc_build_kernel_w8(
        eids_ptr, npost_ptr, slot_ptr, miss_ptr, d_ptr,
        a1b, as1b, c13b, a2b, as2b, c2b,
        poolb, slot_bytes,
        off_w13, off_s13, off_w2, off_s2,
        a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
        n_experts, pairs, cap6, mblock,
        debug_i64_ptr, cap_max: tl.constexpr,  # 🚨 只用 int64 探针，传回纯数字
        BLOCK: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    slot = tl.load(slot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)

    live = p < npost // mblock
    hit = slot >= 0
    m = tl.where(live & hit, mblock, 0).to(tl.int64)
    n_miss = tl.sum(tl.where(mask & live & ~hit, 1, 0))
    tl.atomic_add(miss_ptr, n_miss)

    base = p.to(tl.int64) * mblock
    slot_c = tl.maximum(slot, 0)

    sbase = poolb + slot_c * slot_bytes
    a1 = a1b + base * a1_rb
    as1 = as1b + base * as1_rb
    c13 = c13b + base * c13_rb
    a2 = a2b + base * a2_rb
    as2 = as2b + base * as2_rb
    c2 = c2b + base * c2_rb

    for gi in tl.static_range(2):
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c = sbase + off_w13, sbase + off_s13, a1, as1, c13

            # =========================================================
            tl.store(debug_i64_ptr + 0 * cap_max + p, e, mask=mask)
            tl.store(debug_i64_ptr + 1 * cap_max + p, slot, mask=mask)
            # 记录偏移量：用物理指针减去池子基指针 poolb，得到字节偏移量
            tl.store(debug_i64_ptr + 2 * cap_max + p, sbase - poolb, mask=mask)
            tl.store(debug_i64_ptr + 3 * cap_max + p, s - poolb, mask=mask)

        else:
            b, s, a, as_, c = sbase + off_w2, sbase + off_s2, a2, as2, c2

        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m, mask=mask)

import triton
import triton.language as tl


@triton.jit
def _moe_int8_w13_gemm_kernel_fixed(
        desc_ptr,
        debug_ptr,  # 🚨 新增：外部传入的探针缓冲区指针
        K: tl.constexpr,
        N: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    pid_pair = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_m = tl.program_id(2)

    desc_base = desc_ptr + pid_pair * 6
    a_ptr_int = tl.load(desc_base + 0)
    as_ptr_int = tl.load(desc_base + 1)
    b_ptr_int = tl.load(desc_base + 2)
    s_ptr_int = tl.load(desc_base + 3)
    c_ptr_int = tl.load(desc_base + 4)
    m_val = tl.load(desc_base + 5)

    if pid_m * BLOCK_M >= m_val:
        return

    a_ptr = a_ptr_int.to(tl.pointer_type(tl.int8))
    as_ptr = as_ptr_int.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr_int.to(tl.pointer_type(tl.int8))
    s_ptr = s_ptr_int.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr_int.to(tl.pointer_type(tl.bfloat16))

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < m_val
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k_blocks = tl.cdiv(K, GROUP_SIZE)
    stride_am = K
    stride_asm = num_k_blocks
    stride_bn = K
    stride_bk = 1
    stride_sbn = num_k_blocks
    stride_cm = N

    # 1. 安全计算是否为 Debug Block，彻底避开链式 and 报错
    c1 = (pid_pair == 0)
    c2 = (pid_m == 0)
    c3 = (pid_n == 0)
    is_debug_block = (c1 and c2) and c3

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k * BLOCK_K
        offs_k = tl.arange(0, BLOCK_K)
        mask_k = (k_start + offs_k) < K

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k_start + offs_k[None, :]))
        a_int8 = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0)

        b_ptrs = b_ptr + (offs_n[None, :] * stride_bn + (k_start + offs_k[:, None]) * stride_bk)
        b_int8 = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0)

        k_block_idx = k_start // GROUP_SIZE

        sa_ptrs = as_ptr + (offs_m * stride_asm + k_block_idx)
        scale_a = tl.load(sa_ptrs, mask=mask_m, other=0.0)

        offs_sn = offs_n // GROUP_SIZE
        sb_ptrs = s_ptr + (offs_sn * stride_sbn + k_block_idx)
        scale_b = tl.load(sb_ptrs, mask=mask_n, other=0.0)

        if is_debug_block:
            if k == 0:
                idx_1 = tl.arange(0, 1)  # 生成一个长度为1的安全偏移量 [0]

                # 1. 直接从 B 的 Scale 基地址读取 1 个元素
                debug_sb = tl.load(s_ptr + idx_1)
                tl.store(debug_ptr + 0 + idx_1, debug_sb)

                # 2. 直接从 A 的 Scale 基地址读取 1 个元素
                debug_sa = tl.load(as_ptr + idx_1)
                tl.store(debug_ptr + 1 + idx_1, debug_sa)

                # 3. 直接从 A 的 INT8 基地址读取 1 个元素
                debug_a = tl.load(a_ptr + idx_1)
                tl.store(debug_ptr + 2 + idx_1, debug_a.to(tl.float32))

                # 4. 直接从 B 的 INT8 基地址读取 1 个元素
                debug_b = tl.load(b_ptr + idx_1)
                tl.store(debug_ptr + 3 + idx_1, debug_b.to(tl.float32))


        # 1. 核心: 直接使用 INT8 数据执行 dot 运算，累加器指定为 int32
        dot_int32 = tl.dot(a_int8, b_int8, out_dtype=tl.int32)

        # 2. 将 INT32 的累加结果转为 Float32
        dot_f32 = dot_int32.to(tl.float32)

        # 3. 乘上 A 和 B 的 Scale，直接累加进全局 Float32 累加器中！
        # a_int8 和 b_int8 越界部分 load 时 other=0，所以这里越界部分天然是 0
        acc += dot_f32 * scale_a[:, None] * scale_b[None, :]

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :])
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])