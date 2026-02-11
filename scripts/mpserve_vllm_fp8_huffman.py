"""vLLM integration for FP8 + Huffman (bucket) decode.

This module is intentionally *out-of-tree* relative to vendored `vllm/`.
Importing it registers a new vLLM quantization method via
`vllm.model_executor.layers.quantization.register_quantization_config`.

Usage (see scripts/):
    import mpserve_vllm_fp8_huffman  # registers quant method
    from vllm import LLM
    llm = LLM(model=..., quantization="fp8_huffman", dtype="bfloat16", ...)

Notes:
- MVP implementation: decodes a temporary BF16 weight each forward.
- Tensor parallel (tp>1) is not supported in this MVP.
- Method is gated by env var: MPSERVE_ENABLE_FP8_HUFFMAN=1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
import re
import sys
import weakref

import torch


_PREFIX_TO_LAYER: "weakref.WeakValueDictionary[str, torch.nn.Module]" = weakref.WeakValueDictionary()


def _preload_torch_libs() -> None:
    """Best-effort preload of torch shared libs for custom extensions."""

    try:
        import ctypes
        from pathlib import Path

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if not torch_lib.exists():
            return

        for libname in (
            "libc10.so",
            "libtorch.so",
            "libtorch_cpu.so",
            "libtorch_cuda.so",
            "libtorch_python.so",
        ):
            p = torch_lib / libname
            if p.exists():
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
    except Exception:
        return


def _ensure_mpserve_on_sys_path() -> None:
    """Make sure repository root and scripts dir are importable in subprocesses."""

    try:
        from pathlib import Path

        scripts_dir = Path(__file__).resolve().parent
        repo_root = scripts_dir.parents[0]
        for p in (str(repo_root), str(scripts_dir)):
            if p not in sys.path:
                sys.path.insert(0, p)
    except Exception:
        return


def _import_mpserve_cuda():
    _preload_torch_libs()
    _ensure_mpserve_on_sys_path()
    import MPServe_cuda  # type: ignore

    return MPServe_cuda


@dataclass(frozen=True)
class FP8HuffmanParams:
    max_len: int = 7


from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase  # noqa: E402
from vllm.model_executor.layers.linear import UnquantizedLinearMethod  # noqa: E402
from vllm.model_executor.layers.quantization import register_quantization_config  # noqa: E402
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E402
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import BasevLLMParameter  # noqa: E402


@register_quantization_config("fp8_huffman")
class FP8HuffmanConfig(QuantizationConfig):
    def __init__(
        self,
        *,
        max_len: int = 7,
        include_regex: str | None = None,
        ignore_regex: str | None = r"(^lm_head$|^model\.visual\.|.*\.mlp\.gate$|.*\.mlp\.shared_expert_gate$)",
    ) -> None:
        super().__init__()
        import os

        if int(max_len) != 7:
            raise ValueError("fp8_huffman currently requires max_len=7")
        self.params = FP8HuffmanParams(max_len=int(max_len))
        self.include_regex = include_regex
        self._include_re = re.compile(include_regex) if include_regex else None
        self.ignore_regex = ignore_regex
        self._ignore_re = re.compile(ignore_regex) if ignore_regex else None

        # Only enable this quantization method when explicitly requested.
        self._enabled = os.environ.get("MPSERVE_ENABLE_FP8_HUFFMAN", "0") == "1"

    def get_name(self) -> str:
        return "fp8_huffman"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "FP8HuffmanConfig":
        max_len = int(config.get("max_len", 7))
        include_regex = config.get("include_regex", None)
        ignore_regex = config.get("ignore_regex", None)
        return cls(max_len=max_len, include_regex=include_regex, ignore_regex=ignore_regex)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if not getattr(self, "_enabled", False):
            return UnquantizedLinearMethod()
        if isinstance(layer, LinearBase):
            if self._ignore_re is not None and self._ignore_re.search(prefix) is not None:
                return UnquantizedLinearMethod()
            if self._include_re is None or self._include_re.search(prefix) is not None:
                return FP8HuffmanLinearMethod(self.params)
            return UnquantizedLinearMethod()
        return None


class FP8HuffmanLinearMethod(LinearMethodBase):
    def __init__(self, params: FP8HuffmanParams):
        self.params = params

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        tp_rank = getattr(layer, "tp_rank", 0)
        tp_size = getattr(layer, "tp_size", 1)
        if int(tp_rank) != 0 or int(tp_size) != 1:
            raise NotImplementedError("fp8_huffman MVP only supports tp_size=1")

        out_features = int(output_size)
        in_features = int(input_size)
        _ = (out_features, in_features, input_size, output_size, params_dtype, extra_weight_attrs)

        num_shards = int(len(output_partition_sizes))
        shard_id_to_index: dict[object, int] = {
            0: 0,
            1: 1,
            "q": 0,
            "k": 1,
            "v": 2,
            "gate": 0,
            "up": 1,
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _as_tensor(x: Any) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                return x
            return cast(torch.Tensor, cast(Any, x)[:])

        def _discard_loader(
            param: torch.Tensor,
            loaded_weight: torch.Tensor,
            *args,
            **kwargs,
        ) -> None:
            return

        def _make_store_or_copy_loader(attr_name: str):
            def _loader(
                param: torch.Tensor,
                loaded_weight: torch.Tensor,
                shard_id: object | None = None,
                *args,
                **kwargs,
            ) -> None:
                loaded_weight_t = _as_tensor(loaded_weight)
                tgt = loaded_weight_t.to(device=param.device).contiguous()

                if num_shards <= 1 or shard_id is None:
                    try:
                        if int(tgt.numel()) >= int(param.data.numel()):
                            param.data = tgt
                    except Exception:
                        param.data = tgt
                    return

                shard_index = shard_id_to_index.get(shard_id, None)
                if shard_index is None:
                    raise ValueError(f"Unknown shard_id for fp8_huffman: {shard_id!r}")

                d = getattr(layer, f"_{attr_name}_shards", None)
                if d is None:
                    d = {}
                    setattr(layer, f"_{attr_name}_shards", d)
                d[int(shard_index)] = tgt

            return _loader

        layer.register_parameter(
            "huff_fp8_u8",
            BasevLLMParameter(
                data=torch.empty((1,), dtype=torch.uint8, device=device),
                weight_loader=_make_store_or_copy_loader("huff_fp8_u8"),
            ),
        )
        layer.register_parameter(
            "huff_bitstream_u64",
            BasevLLMParameter(
                data=torch.empty((1,), dtype=torch.int64, device=device),
                weight_loader=_make_store_or_copy_loader("huff_bitstream_u64"),
            ),
        )
        layer.register_parameter(
            "huff_decode_syms_u16",
            BasevLLMParameter(
                data=torch.empty((1,), dtype=torch.int16, device=device),
                weight_loader=_make_store_or_copy_loader("huff_decode_syms_u16"),
            ),
        )
        layer.register_parameter(
            "huff_decode_lens_u8",
            BasevLLMParameter(
                data=torch.empty((1,), dtype=torch.uint8, device=device),
                weight_loader=_make_store_or_copy_loader("huff_decode_lens_u8"),
            ),
        )
        layer.register_parameter(
            "huff_bucket_table_i32",
            BasevLLMParameter(
                data=torch.empty((1,), dtype=torch.int32, device=device),
                weight_loader=_make_store_or_copy_loader("huff_bucket_table_i32"),
            ),
        )
        layer.register_parameter(
            "huff_bucket_col_start_i32",
            BasevLLMParameter(
                data=torch.empty((1,), dtype=torch.int32, device=device),
                weight_loader=_make_store_or_copy_loader("huff_bucket_col_start_i32"),
            ),
        )
        layer.register_parameter(
            "huff_bucket_col_len_i32",
            BasevLLMParameter(
                data=torch.empty((1,), dtype=torch.int32, device=device),
                weight_loader=_make_store_or_copy_loader("huff_bucket_col_len_i32"),
            ),
        )
        layer.register_parameter(
            "huff_bucket_row_offsets_i32",
            BasevLLMParameter(
                data=torch.empty((1,), dtype=torch.int32, device=device),
                weight_loader=_make_store_or_copy_loader("huff_bucket_row_offsets_i32"),
            ),
        )
        layer.register_parameter(
            "huff_row_indices_i32",
            BasevLLMParameter(
                data=torch.empty((1,), dtype=torch.int32, device=device),
                weight_loader=_make_store_or_copy_loader("huff_row_indices_i32"),
            ),
        )
        layer.register_parameter(
            "huff_bucket_thread_startbit_u32",
            BasevLLMParameter(
                data=torch.empty((1,), dtype=torch.int32, device=device),
                weight_loader=_make_store_or_copy_loader("huff_bucket_thread_startbit_u32"),
            ),
        )
        layer.register_parameter(
            "huff_bucket_bit_base_i64",
            BasevLLMParameter(
                data=torch.empty((1,), dtype=torch.int64, device=device),
                weight_loader=_make_store_or_copy_loader("huff_bucket_bit_base_i64"),
            ),
        )

        layer.register_parameter(
            "huff_attn_block_row_start_i32",
            BasevLLMParameter(
                data=torch.zeros((0,), dtype=torch.int32, device=device),
                weight_loader=_make_store_or_copy_loader("huff_attn_block_row_start_i32"),
            ),
        )
        layer.register_parameter(
            "huff_attn_block_row_len_i32",
            BasevLLMParameter(
                data=torch.zeros((0,), dtype=torch.int32, device=device),
                weight_loader=_make_store_or_copy_loader("huff_attn_block_row_len_i32"),
            ),
        )

        layer.register_parameter(
            "weight",
            BasevLLMParameter(
                data=torch.empty((1,), dtype=torch.uint8, device=device),
                weight_loader=_discard_loader,
            ),
        )
        layer.register_parameter(
            "weight_scale_inv",
            BasevLLMParameter(
                data=torch.empty((1,), dtype=torch.float16, device=device),
                weight_loader=_discard_loader,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            BasevLLMParameter(
                data=torch.empty((1,), dtype=torch.float16, device=device),
                weight_loader=_discard_loader,
            ),
        )

        layer.register_buffer(
            "huff_max_len",
            torch.tensor(int(self.params.max_len), dtype=torch.int32, device=device),
            persistent=False,
        )
        layer.register_buffer(
            "huff_in_features_i32",
            torch.tensor(int(in_features), dtype=torch.int32, device=device),
            persistent=False,
        )
        layer.register_buffer(
            "huff_out_features_i32",
            torch.tensor(int(out_features), dtype=torch.int32, device=device),
            persistent=False,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        for name in (
            "huff_fp8_u8",
            "huff_bitstream_u64",
            "huff_decode_syms_u16",
            "huff_decode_lens_u8",
            "huff_bucket_table_i32",
            "huff_bucket_col_start_i32",
            "huff_bucket_col_len_i32",
            "huff_bucket_row_offsets_i32",
            "huff_row_indices_i32",
            "huff_bucket_thread_startbit_u32",
            "huff_bucket_bit_base_i64",
            "huff_attn_block_row_start_i32",
            "huff_attn_block_row_len_i32",
        ):
            p = getattr(layer, name)
            if not isinstance(p, torch.nn.Parameter):
                setattr(layer, name, torch.nn.Parameter(p, requires_grad=False))

        try:
            prefix = getattr(layer, "prefix", None)
            if isinstance(prefix, str) and prefix:
                _PREFIX_TO_LAYER[prefix] = layer
        except Exception:
            pass

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("fp8_huffman requires CUDA tensors")
        if x.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("fp8_huffman expects activations in bf16 or fp16")

        if x.numel() == 0:
            out_features = 0
            if hasattr(layer, "huff_out_features_i32"):
                out_features = int(getattr(layer, "huff_out_features_i32").item())
            if out_features <= 0 and hasattr(layer, "output_size"):
                out_features = int(getattr(layer, "output_size"))
            if out_features <= 0:
                prefix = getattr(layer, "prefix", None)
                raise RuntimeError(
                    "fp8_huffman: cannot infer out_features for empty-token run "
                    f"(prefix={prefix!r}, layer_type={type(layer).__name__})"
                )
            out_shape = x.shape[:-1] + (int(out_features),)
            out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
            if bias is not None and int(out.numel()) > 0:
                out.add_(bias)
            return out

        MPServe_cuda = _import_mpserve_cuda()
        max_len = int(getattr(layer, "huff_max_len").item())

        x2d = x.reshape(-1, x.shape[-1])
        m_in = int(x2d.shape[1])

        def _ensure_fp8_u8_2d(t: torch.Tensor) -> torch.Tensor:
            if t.numel() == 0:
                raise RuntimeError(
                    f"fp8_huffman: encountered empty fp8_u8 tensor (shape={tuple(t.shape)}). "
                    "This usually indicates missing/incorrect shard_id remapping during weight loading."
                )
            if t.dim() == 2:
                return t
            if t.dim() == 1:
                if m_in <= 0 or (t.numel() % m_in) != 0:
                    raise ValueError(
                        f"fp8_huffman: invalid fp8_u8 flat shape={tuple(t.shape)} for m_in={m_in}"
                    )
                return t.view(-1, m_in)
            raise ValueError(f"fp8_huffman: invalid fp8_u8 dim={t.dim()} shape={tuple(t.shape)}")

        def _decode_one(
            fp8_u8: torch.Tensor,
            bitstream_u64: torch.Tensor,
            decode_syms_u16: torch.Tensor,
            decode_lens_u8: torch.Tensor,
            bucket_table_i32: torch.Tensor,
            bucket_col_start_i32: torch.Tensor,
            bucket_col_len_i32: torch.Tensor,
            bucket_row_offsets_i32: torch.Tensor,
            row_indices_i32: torch.Tensor,
            bucket_thread_startbit_u32: torch.Tensor,
            bucket_bit_base_i64: torch.Tensor,
        ) -> torch.Tensor:
            fp8_u8_2d = _ensure_fp8_u8_2d(fp8_u8)
            n = int(fp8_u8_2d.shape[0])
            m = int(fp8_u8_2d.shape[1])

            if bitstream_u64.dtype == torch.int64:
                bitstream_u64 = bitstream_u64.view(torch.uint64)
            elif bitstream_u64.dtype != torch.uint64:
                raise ValueError(f"fp8_huffman: invalid bitstream dtype={bitstream_u64.dtype}")

            if decode_syms_u16.dtype == torch.int16:
                decode_syms_u16 = decode_syms_u16.view(torch.uint16)
            elif decode_syms_u16.dtype != torch.uint16:
                raise ValueError(f"fp8_huffman: invalid decode_syms dtype={decode_syms_u16.dtype}")

            if bucket_thread_startbit_u32.dtype == torch.int32:
                bucket_thread_startbit_u32 = bucket_thread_startbit_u32.view(torch.uint32)
            elif bucket_thread_startbit_u32.dtype != torch.uint32:
                raise ValueError(
                    f"fp8_huffman: invalid bucket_thread_startbit dtype={bucket_thread_startbit_u32.dtype}"
                )

            return MPServe_cuda.huffman_decode_fp8(
                fp8_u8_2d.contiguous(),
                bitstream_u64.contiguous(),
                decode_syms_u16.contiguous(),
                decode_lens_u8.contiguous(),
                bucket_table_i32.contiguous(),
                bucket_col_start_i32.contiguous(),
                bucket_col_len_i32.contiguous(),
                bucket_row_offsets_i32.contiguous(),
                row_indices_i32.contiguous(),
                bucket_thread_startbit_u32.contiguous(),
                bucket_bit_base_i64.contiguous(),
                n,
                m,
                max_len,
            )

        def _decode_weight_for_layer(l: torch.nn.Module) -> torch.Tensor:
            shards = getattr(l, "_huff_fp8_u8_shards", None)
            if isinstance(shards, dict) and len(shards) > 0:
                shard_indices = sorted(shards.keys())

                any_empty = False
                best_si = None
                best_numel = -1
                for si in shard_indices:
                    fp8_u8 = getattr(l, "_huff_fp8_u8_shards")[si]
                    n_el = int(fp8_u8.numel())
                    if n_el == 0:
                        any_empty = True
                    if n_el > best_numel:
                        best_numel = n_el
                        best_si = si

                if any_empty:
                    if best_si is None or best_numel <= 0:
                        raise RuntimeError("fp8_huffman: all shard sidecars are empty")
                    si = best_si
                    return _decode_one(
                        getattr(l, "_huff_fp8_u8_shards")[si],
                        getattr(l, "_huff_bitstream_u64_shards")[si],
                        getattr(l, "_huff_decode_syms_u16_shards")[si],
                        getattr(l, "_huff_decode_lens_u8_shards")[si],
                        getattr(l, "_huff_bucket_table_i32_shards")[si],
                        getattr(l, "_huff_bucket_col_start_i32_shards")[si],
                        getattr(l, "_huff_bucket_col_len_i32_shards")[si],
                        getattr(l, "_huff_bucket_row_offsets_i32_shards")[si],
                        getattr(l, "_huff_row_indices_i32_shards")[si],
                        getattr(l, "_huff_bucket_thread_startbit_u32_shards")[si],
                        getattr(l, "_huff_bucket_bit_base_i64_shards")[si],
                    )

                ws = []
                for si in shard_indices:
                    ws.append(
                        _decode_one(
                            getattr(l, "_huff_fp8_u8_shards")[si],
                            getattr(l, "_huff_bitstream_u64_shards")[si],
                            getattr(l, "_huff_decode_syms_u16_shards")[si],
                            getattr(l, "_huff_decode_lens_u8_shards")[si],
                            getattr(l, "_huff_bucket_table_i32_shards")[si],
                            getattr(l, "_huff_bucket_col_start_i32_shards")[si],
                            getattr(l, "_huff_bucket_col_len_i32_shards")[si],
                            getattr(l, "_huff_bucket_row_offsets_i32_shards")[si],
                            getattr(l, "_huff_row_indices_i32_shards")[si],
                            getattr(l, "_huff_bucket_thread_startbit_u32_shards")[si],
                            getattr(l, "_huff_bucket_bit_base_i64_shards")[si],
                        )
                    )
                return torch.cat(ws, dim=0)

            return _decode_one(
                getattr(l, "huff_fp8_u8"),
                getattr(l, "huff_bitstream_u64"),
                getattr(l, "huff_decode_syms_u16"),
                getattr(l, "huff_decode_lens_u8"),
                getattr(l, "huff_bucket_table_i32"),
                getattr(l, "huff_bucket_col_start_i32"),
                getattr(l, "huff_bucket_col_len_i32"),
                getattr(l, "huff_bucket_row_offsets_i32"),
                getattr(l, "huff_row_indices_i32"),
                getattr(l, "huff_bucket_thread_startbit_u32"),
                getattr(l, "huff_bucket_bit_base_i64"),
            )

        def _get_forward_cache_key() -> object:
            try:
                from vllm.forward_context import get_forward_context, is_forward_context_available

                if not is_forward_context_available():
                    return ("no_forward_context",)
                fc = get_forward_context()
                return (fc.virtual_engine, fc.batch_descriptor)
            except Exception:
                return ("no_forward_context",)

        def _get_or_decode_packed_weight(leader: torch.nn.Module) -> torch.Tensor:
            ck = _get_forward_cache_key()
            cached = getattr(leader, "_fp8_huff_attn_block_cache", None)
            if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == ck and isinstance(cached[1], torch.Tensor):
                return cached[1]
            w_full = _decode_weight_for_layer(leader)
            setattr(leader, "_fp8_huff_attn_block_cache", (ck, w_full))
            return w_full

        def _infer_layer_leader_prefix(prefix: str) -> str | None:
            if ".self_attn." in prefix:
                layer_root = prefix.split(".self_attn.", 1)[0]
                return layer_root + ".self_attn.o_proj"
            if ".mlp." in prefix:
                layer_root = prefix.split(".mlp.", 1)[0]
                return layer_root + ".self_attn.o_proj"
            return None

        def _get_slice_meta_nonshard(l: torch.nn.Module) -> tuple[int, int] | None:
            rs = getattr(l, "huff_attn_block_row_start_i32", None)
            rl = getattr(l, "huff_attn_block_row_len_i32", None)
            if not isinstance(rs, torch.Tensor) or not isinstance(rl, torch.Tensor):
                return None
            if int(rs.numel()) == 0 or int(rl.numel()) == 0:
                return None
            row_start = int(rs.reshape(-1)[0].item())
            row_len = int(rl.reshape(-1)[0].item())
            if row_len <= 0:
                return None
            return row_start, row_len

        def _get_slice_meta_sharded(l: torch.nn.Module) -> tuple[int, int] | None:
            d_rs = getattr(l, "_huff_attn_block_row_start_i32_shards", None)
            d_rl = getattr(l, "_huff_attn_block_row_len_i32_shards", None)
            if not isinstance(d_rs, dict) or not isinstance(d_rl, dict) or len(d_rs) == 0:
                return None
            starts = []
            ends = []
            for si, rs_t in d_rs.items():
                rl_t = d_rl.get(si, None)
                if not isinstance(rs_t, torch.Tensor) or not isinstance(rl_t, torch.Tensor):
                    continue
                if int(rs_t.numel()) == 0 or int(rl_t.numel()) == 0:
                    continue
                s = int(rs_t.reshape(-1)[0].item())
                ln = int(rl_t.reshape(-1)[0].item())
                if ln <= 0:
                    continue
                starts.append(s)
                ends.append(s + ln)
            if not starts:
                return None
            row_start = int(min(starts))
            row_end = int(max(ends))
            if row_end <= row_start:
                return None
            return row_start, (row_end - row_start)

        def _maybe_apply_layer_pack(l: torch.nn.Module) -> torch.Tensor | None:
            prefix = getattr(l, "prefix", None)
            if not isinstance(prefix, str) or not prefix:
                return None

            meta = _get_slice_meta_sharded(l)
            if meta is None:
                meta = _get_slice_meta_nonshard(l)
            if meta is None:
                return None

            row_start, row_len = meta
            leader_prefix = _infer_layer_leader_prefix(prefix)
            if leader_prefix is None:
                return None
            leader = _PREFIX_TO_LAYER.get(leader_prefix, None)
            if leader is None:
                raise RuntimeError(f"fp8_huffman: cannot find layer-pack leader {leader_prefix!r} for {prefix!r}")

            w_full = _get_or_decode_packed_weight(leader)
            w = w_full[row_start : row_start + row_len]
            if x.dtype == torch.float16 and w.dtype != torch.float16:
                w = w.to(torch.float16)
            return torch.matmul(x2d, w.t())

        out_lp = _maybe_apply_layer_pack(layer)
        if out_lp is not None:
            out = out_lp
        else:
            w_full = _decode_weight_for_layer(layer)
            w = w_full
            if x.dtype == torch.float16 and w.dtype != torch.float16:
                w = w.to(torch.float16)
            out = torch.matmul(x2d, w.t())

        if bias is not None:
            out.add_(bias)
        return out.reshape(x.shape[:-1] + (out.shape[-1],))
