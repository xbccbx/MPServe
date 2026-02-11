#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run inference with an FP8+Huffman sidecar model via vLLM.

Prerequisites:
- You have built a sidecar model directory using scripts/compress_llm_fp8_huffman.py.
- vLLM is installed.
- The CUDA extension module MPServe_cuda is importable (build with: python setup.py build_ext --inplace).

Example:
    python scripts/infer_llm_fp8_huffman.py \
        --model <out_dir> \
        --prompt "Hello. Introduce yourself in one sentence." \
        --max_tokens 128

Notes:
- This script imports scripts/mpserve_vllm_fp8_huffman.py to register the vLLM quant method.
- The method is gated by MPSERVE_ENABLE_FP8_HUFFMAN=1.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Sidecar model directory (out_dir)")
    ap.add_argument("--prompt", type=str, required=True, help="Single prompt")
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    ap.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="vLLM activation dtype (integration supports bf16/fp16)",
    )

    args = ap.parse_args()

    # Explicitly enable the fp8_huffman method (avoid interfering with BF16 runs).
    os.environ.setdefault("MPSERVE_ENABLE_FP8_HUFFMAN", "1")

    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = Path(__file__).resolve().parent
    for p in (str(repo_root), str(scripts_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Fail fast if the extension cannot be imported.
    try:
        import MPServe_cuda  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "Failed to import MPServe_cuda. Build the extension first:\n"
            "  python setup.py build_ext --inplace\n"
            f"Original error: {e}"
        )

    # Register vLLM quantization config.
    import mpserve_vllm_fp8_huffman  # noqa: F401

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        quantization="fp8_huffman",
        dtype=args.dtype,
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        trust_remote_code=bool(args.trust_remote_code),
    )

    sampling_params = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
    )

    outputs = llm.generate([args.prompt], sampling_params)
    out = outputs[0]
    print(out.outputs[0].text)


if __name__ == "__main__":
    main()
