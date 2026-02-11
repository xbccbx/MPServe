# MPServe

This repository contains GPU kernel implementations for the paper
"Redundancy-Reduced Storage and Fast Decompression for Multi-Precision LLM Serving".

## Build

Build the PyTorch CUDA extension module (required by the scripts below):

```bash
python setup.py build_ext --inplace
```

The Python import name of the extension is `MPServe_cuda`.

## Scripts

### Compress a model (FP8 + Huffman sidecar)

Given a BF16 model directory and an FP8 model directory (both in HF safetensors
format, i.e. containing `model.safetensors.index.json`), build a new output
model directory that contains the original FP8 checkpoint plus a Huffman
sidecar shard:

```bash
python scripts/compress_llm_fp8_huffman.py \
  --bf16_model <bf16_model_dir> \
  --fp8_model  <fp8_model_dir> \
  --out_dir    <out_dir>
```

Run `python scripts/compress_llm_fp8_huffman.py -h` for advanced options.

### Inference with a sidecar model (vLLM)

```bash
python scripts/infer_llm_fp8_huffman.py \
  --model <out_dir> \
  --prompt "Hello. Introduce yourself in one sentence." \
  --max_tokens 128
```

This uses vLLM with `quantization="fp8_huffman"`. The method is gated by
`MPSERVE_ENABLE_FP8_HUFFMAN=1` (the inference script sets it by default).
