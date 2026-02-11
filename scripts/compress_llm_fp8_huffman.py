#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build an FP8 + Huffman sidecar for vLLM inference.

This script is self-contained (does not depend on `tasks/`). It:
- Copies (or hardlinks) a base model directory into `out_dir`.
- Reads BF16 and FP8 safetensors shards via `model.safetensors.index.json`.
- For each BF16 Linear weight, builds Huffman bucket decode artifacts using
  the CUDA extension `MPServe_cuda`.
- Writes a sidecar safetensors shard (default: `huffman_fp8.safetensors`) and
  updates `out_dir/model.safetensors.index.json`.

Example:
  python scripts/compress_llm_fp8_huffman.py \
  --bf16_model <bf16_model_dir> \
  --fp8_model  <fp8_model_dir> \
  --out_dir    <out_dir>

Notes:
- The FP8 decode kernel in this repository expects `--max_len=7`.
- Requires the extension module `MPServe_cuda` to be importable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


def _log(msg: str) -> None:
  ts = time.strftime("%Y-%m-%d %H:%M:%S")
  print(f"[{ts}] {msg}", flush=True)


def _load_safetensors_weight_map(model_dir: Path) -> Dict[str, str]:
  index_path = model_dir / "model.safetensors.index.json"
  if not index_path.exists():
    raise FileNotFoundError(f"Missing safetensors index: {index_path}")
  with open(index_path, "r", encoding="utf-8") as f:
    idx = json.load(f)
  wm = idx.get("weight_map")
  if not isinstance(wm, dict):
    raise ValueError(f"Invalid weight_map in {index_path}")
  return wm  # type: ignore[return-value]


def _load_tensor_from_safetensors(model_dir: Path, weight_map: Dict[str, str], key: str) -> torch.Tensor:
  shard_rel = weight_map.get(key)
  if not isinstance(shard_rel, str):
    raise KeyError(f"Key not in safetensors index: {key}")
  shard_path = model_dir / shard_rel
  if not shard_path.exists():
    raise FileNotFoundError(f"Missing safetensors shard for {key}: {shard_path}")

  from safetensors import safe_open

  with safe_open(str(shard_path), framework="pt", device="cpu") as f:
    return f.get_tensor(key)


def _as_uint8_fp8_tensor(fp8_params: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
  """Convert stored fp8 tensor into uint8 with the same 2D shape.

  Handles:
  - uint8 tensors
  - float8 tensors (reinterpret raw bytes)
  - bf16/fp16/fp32 tensors storing packed bytes (best-effort)
  """
  if fp8_params.dtype == torch.uint8:
    out = fp8_params
  elif str(fp8_params.dtype).startswith("torch.float8"):
    out = fp8_params.view(torch.uint8)
  elif fp8_params.dtype in (torch.bfloat16, torch.float16):
    out = fp8_params.view(torch.uint16).view(torch.uint8)
  elif fp8_params.dtype == torch.float32:
    out = fp8_params.to(torch.bfloat16).view(torch.uint16).view(torch.uint8)
  else:
    out = fp8_params.view(torch.uint8)

  expected = int(shape[0] * shape[1])
  if int(out.numel()) != expected:
    raise ValueError(f"FP8 numel mismatch: got {int(out.numel())} expected {expected}")
  return out.reshape(shape)


def _to_supported_scale_dtype(scales: torch.Tensor) -> torch.Tensor:
  """MPServe CUDA kernel accepts scales as torch.uint16 (raw bits) or torch.bfloat16."""
  if scales.dtype in (torch.uint16, torch.bfloat16):
    return scales
  if scales.dtype == torch.float16:
    return scales.view(torch.uint16)
  if scales.dtype == torch.float32:
    return scales.to(torch.float16).view(torch.uint16)
  return scales.to(torch.bfloat16)


def _normalize_scales_for_build(scales: torch.Tensor, n: int, m: int) -> torch.Tensor:
  """Normalize scales into a form accepted by huffman_build_fp8_decode_tables.

  Kernel supports:
  - scalar
  - per-row scales: [n] or [n,1]
  - grouped-per-row scales: [n/g] expanded to [n]

  If scales are true 2D block-wise (e.g. [n/128, m/128]), we conservatively
  convert to per-row by taking the first column-block per row-block.
  """
  if int(scales.numel()) == 1:
    return scales

  if scales.dim() == 2 and int(scales.size(0)) == n and int(scales.size(1)) == 1:
    scales = scales.squeeze(1)

  if scales.dim() == 1 and int(scales.numel()) == n:
    return scales

  if scales.dim() == 1 and int(scales.numel()) > 0 and (n % int(scales.numel())) == 0:
    group = n // int(scales.numel())
    return scales.repeat_interleave(group)

  if scales.dim() == 2 and int(scales.size(1)) == 1 and int(scales.size(0)) > 0 and (n % int(scales.size(0))) == 0:
    group = n // int(scales.size(0))
    return scales.squeeze(1).repeat_interleave(group)

  if scales.dim() == 2 and int(scales.size(0)) > 0 and int(scales.size(1)) > 0:
    if (n % int(scales.size(0))) == 0 and (m % int(scales.size(1))) == 0:
      sb0 = n // int(scales.size(0))
      return scales[:, 0].repeat_interleave(sb0)

  return scales


def _link_or_copy_tree(src: Path, dst: Path) -> None:
  dst.mkdir(parents=True, exist_ok=True)
  for root, dirs, files in os.walk(src):
    rel = Path(root).relative_to(src)
    (dst / rel).mkdir(parents=True, exist_ok=True)
    for d in dirs:
      (dst / rel / d).mkdir(parents=True, exist_ok=True)
    for fn in files:
      sp = Path(root) / fn
      dp = dst / rel / fn
      if dp.exists():
        continue
      if fn in ("model.safetensors.index.json", "config.json"):
        shutil.copy2(sp, dp)
        continue
      try:
        os.link(sp, dp)
      except Exception:
        shutil.copy2(sp, dp)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--bf16_model", type=str, required=True, help="Path to BF16 model dir (safetensors)")
  ap.add_argument("--fp8_model", type=str, required=True, help="Path to FP8 model dir (safetensors)")
  ap.add_argument("--out_dir", type=str, required=True, help="Output model dir (FP8 + Huffman sidecar)")
  ap.add_argument(
    "--base_model",
    type=str,
    default=None,
    help="Directory tree to copy/link into out_dir before adding sidecar (default: fp8_model)",
  )
  ap.add_argument("--include", type=str, default=None, help="Regex to include BF16 weight keys")
  ap.add_argument("--exclude", type=str, default=None, help="Regex to exclude BF16 weight keys")
  ap.add_argument("--max_len", type=int, default=7, help="Max Huffman code length (must be 7)")
  ap.add_argument("--threads", type=int, default=1024, help="threads_per_block used for bucket metadata")
  ap.add_argument("--sidecar_name", type=str, default="huffman_fp8.safetensors", help="Sidecar shard filename")
  ap.add_argument("--store_fp8_u8", action="store_true", default=True, help="Store huff_fp8_u8 tensors")
  ap.add_argument("--no_store_fp8_u8", action="store_false", dest="store_fp8_u8")
  ap.add_argument(
    "--override_common_from_bf16",
    action="store_true",
    help="Write a small overrides shard for keys that exist in both BF16 and FP8 checkpoints",
  )
  ap.add_argument(
    "--override_allow_regex",
    type=str,
    default=r"^(?:model\.embed_tokens\.weight|lm_head\.weight)$",
    help="Regex filter for override keys (when --override_common_from_bf16 is set)",
  )
  args = ap.parse_args()

  if int(args.max_len) != 7:
    raise SystemExit("--max_len must be 7 for the FP8 Huffman decode kernel")

  bf16_dir = Path(args.bf16_model)
  fp8_dir = Path(args.fp8_model)
  out_dir = Path(args.out_dir)
  base_dir = Path(args.base_model) if args.base_model else fp8_dir

  if not bf16_dir.is_dir():
    raise SystemExit(f"--bf16_model must be a directory: {bf16_dir}")
  if not fp8_dir.is_dir():
    raise SystemExit(f"--fp8_model must be a directory: {fp8_dir}")

  include_re: Optional[re.Pattern[str]] = re.compile(args.include) if args.include else None
  exclude_re: Optional[re.Pattern[str]] = re.compile(args.exclude) if args.exclude else None

  def _selected(key: str) -> bool:
    if include_re is not None and include_re.search(key) is None:
      return False
    if exclude_re is not None and exclude_re.search(key) is not None:
      return False
    return True

  _log(
    "Start FP8 Huffman sidecar build: "
    f"bf16_model={bf16_dir} fp8_model={fp8_dir} out_dir={out_dir} "
    f"max_len={args.max_len} threads={args.threads}"
  )
  _log(f"python={sys.version.split()[0]} torch={torch.__version__} cuda_available={torch.cuda.is_available()}")

  t0 = time.time()
  _link_or_copy_tree(base_dir, out_dir)
  _log(f"Prepared out_dir (copy/link): elapsed={time.time() - t0:.2f}s")

  wm_b = _load_safetensors_weight_map(bf16_dir)
  wm_f = _load_safetensors_weight_map(fp8_dir)
  bf16_weight_keys = [k for k in wm_b.keys() if k.endswith(".weight")]
  _log(f"Loaded safetensors index: bf16_keys={len(wm_b)} fp8_keys={len(wm_f)}")
  _log(f"BF16 .weight candidates: {len(bf16_weight_keys)}")

  # Import extension.
  import MPServe_cuda  # type: ignore

  sidecar_tensors: Dict[str, torch.Tensor] = {}
  built = 0
  skipped_missing_fp8 = 0
  skipped_non_bf16_2d = 0

  for key in bf16_weight_keys:
    if not _selected(key):
      continue

    module_prefix = key[: -len(".weight")]
    fp8_weight_key = key if key in wm_f else None
    if fp8_weight_key is None:
      skipped_missing_fp8 += 1
      continue

    scales_key = None
    for cand in (f"{module_prefix}.weight_scale_inv", f"{module_prefix}.weight_scale"):
      if cand in wm_f:
        scales_key = cand
        break
    if scales_key is None:
      skipped_missing_fp8 += 1
      continue

    bf16_w = _load_tensor_from_safetensors(bf16_dir, wm_b, key).contiguous()
    if bf16_w.dtype != torch.bfloat16 or bf16_w.dim() != 2:
      skipped_non_bf16_2d += 1
      continue
    n, m = int(bf16_w.shape[0]), int(bf16_w.shape[1])

    fp8_w = _load_tensor_from_safetensors(fp8_dir, wm_f, fp8_weight_key).contiguous()
    fp8_u8 = _as_uint8_fp8_tensor(fp8_w, (n, m)).contiguous()

    scales_raw = _load_tensor_from_safetensors(fp8_dir, wm_f, scales_key).contiguous()
    scales_in = _to_supported_scale_dtype(scales_raw).contiguous()
    scales_norm = _normalize_scales_for_build(scales_in, n, m).contiguous()

    _log(f"Building artifact '{module_prefix}'")
    out_build = MPServe_cuda.huffman_build_fp8_decode_tables(
      bf16_w,
      fp8_u8,
      scales_norm,
      max_len=int(args.max_len),
      threads_per_block=int(args.threads),
    )

    if bool(args.store_fp8_u8):
      sidecar_tensors[f"{module_prefix}.huff_fp8_u8"] = fp8_u8
    else:
      sidecar_tensors[f"{module_prefix}.huff_fp8_u8"] = torch.empty((0, 0), dtype=torch.uint8)

    # safetensors does not support uint64/uint16/uint32.
    sidecar_tensors[f"{module_prefix}.huff_bitstream_u64"] = out_build["bitstream_u64"].to(torch.int64).contiguous()
    sidecar_tensors[f"{module_prefix}.huff_decode_syms_u16"] = out_build["decode_syms_u16"].to(torch.int16).contiguous()
    sidecar_tensors[f"{module_prefix}.huff_decode_lens_u8"] = out_build["decode_lens_u8"].contiguous()
    sidecar_tensors[f"{module_prefix}.huff_bucket_table_i32"] = out_build["bucket_table_i32"].contiguous()
    sidecar_tensors[f"{module_prefix}.huff_bucket_col_start_i32"] = out_build["bucket_col_start_i32"].contiguous()
    sidecar_tensors[f"{module_prefix}.huff_bucket_col_len_i32"] = out_build["bucket_col_len_i32"].contiguous()
    sidecar_tensors[f"{module_prefix}.huff_bucket_row_offsets_i32"] = out_build["bucket_row_offsets_i32"].contiguous()
    sidecar_tensors[f"{module_prefix}.huff_row_indices_i32"] = out_build["row_indices_i32"].contiguous()
    sidecar_tensors[f"{module_prefix}.huff_bucket_thread_startbit_u32"] = out_build["bucket_thread_startbit_u32"].to(torch.int32).contiguous()
    sidecar_tensors[f"{module_prefix}.huff_bucket_bit_base_i64"] = out_build["bucket_bit_base_i64"].contiguous()

    # Optional fields used by packed modes in other scripts; keep empty.
    sidecar_tensors[f"{module_prefix}.huff_2d_nb1_i32"] = torch.tensor([1], dtype=torch.int32)
    sidecar_tensors[f"{module_prefix}.huff_attn_block_row_start_i32"] = torch.empty((0,), dtype=torch.int32)
    sidecar_tensors[f"{module_prefix}.huff_attn_block_row_len_i32"] = torch.empty((0,), dtype=torch.int32)
    sidecar_tensors[f"{module_prefix}.huff_attn_block_col_start_i32"] = torch.empty((0,), dtype=torch.int32)
    sidecar_tensors[f"{module_prefix}.huff_attn_block_col_len_i32"] = torch.empty((0,), dtype=torch.int32)

    built += 1
    if built % 16 == 0:
      _log(f"Built {built} modules...")

  if not sidecar_tensors:
    raise SystemExit(
      "No modules matched. "
      f"skipped_missing_fp8={skipped_missing_fp8} skipped_non_bf16_2d={skipped_non_bf16_2d}"
    )

  from safetensors.torch import save_file

  sidecar_path = out_dir / str(args.sidecar_name)
  _log(f"Writing sidecar shard: {sidecar_path} (tensors={len(sidecar_tensors)})")
  save_file(sidecar_tensors, str(sidecar_path))

  out_index_path = out_dir / "model.safetensors.index.json"
  with open(out_index_path, "r", encoding="utf-8") as f:
    idx = json.load(f)
  weight_map = idx.get("weight_map")
  if not isinstance(weight_map, dict):
    raise ValueError(f"Invalid weight_map in {out_index_path}")

  for k in sidecar_tensors.keys():
    weight_map[k] = str(args.sidecar_name)

  if bool(args.override_common_from_bf16):
    common_keys = sorted(set(wm_b.keys()) & set(wm_f.keys()))
    allow_re = re.compile(args.override_allow_regex) if args.override_allow_regex else None
    override_keys = [k for k in common_keys if allow_re is None or allow_re.search(k) is not None]
    _log(f"Overriding common keys from BF16: common={len(common_keys)} selected={len(override_keys)}")

    overrides: Dict[str, torch.Tensor] = {}
    for k in override_keys:
      overrides[k] = _load_tensor_from_safetensors(bf16_dir, wm_b, k)
    if overrides:
      override_name = "overrides_from_bf16.safetensors"
      override_path = out_dir / override_name
      _log(f"Writing override shard: {override_path} (tensors={len(overrides)})")
      save_file(overrides, str(override_path))
      for k in override_keys:
        weight_map[k] = override_name

  idx["weight_map"] = weight_map
  with open(out_index_path, "w", encoding="utf-8") as f:
    json.dump(idx, f, indent=2, sort_keys=True)

  cfg_path = out_dir / "config.json"
  if cfg_path.exists():
    try:
      cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
      qc = cfg.get("quantization_config")
      if not isinstance(qc, dict):
        qc = {}
      qc["quant_method"] = "fp8_huffman"
      qc.setdefault("max_len", int(args.max_len))
      cfg["quantization_config"] = qc
      cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
      pass

  print(f"Wrote sidecar shard: {sidecar_path}")
  print(f"Updated index: {out_index_path}")
  print(f"Added keys: {len(sidecar_tensors)}")


if __name__ == "__main__":
  main()
