# krea2-quantizer

Quantize a Krea 2 diffusion-model `.safetensors` checkpoint for ComfyUI.

## How to Use

Tell your AI agent:

```text
Set up the repository at https://github.com/blue-pen5805/krea2-quantizer.
Use it to quantize my Krea 2 diffusion-model .safetensors for ComfyUI.
```

Or follow the manual setup steps below.

## Requirements

- Python 3.10+
- NVIDIA GPU with a CUDA-capable PyTorch environment
- Recent NVIDIA driver compatible with the PyTorch CUDA wheel you install
- Krea 2 diffusion model `.safetensors`

Install PyTorch separately before the remaining requirements. CUDA 13.0 is recommended,
because the `nvfp4` recipe requires NVFP4 support.

If CUDA 13.0 works on your machine:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch
```

If your machine needs a different CUDA wheel, install the matching PyTorch build first
from the official PyTorch instructions, then install the remaining requirements.

## Setup

After PyTorch is installed:

```bash
python -m pip install -r requirements.txt
```

Check CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

The final line must print `True` when using `--device cuda`. If it prints `False`,
fix the PyTorch/CUDA installation before running quantization.

## Recipes

- `nvfp4`
- `fp8_scaled`
- `mxfp8`

## Quantize

NVFP4:

```bash
python scripts/quantize_krea2.py \
  --input /path/to/krea2_raw_bf16.safetensors \
  --recipe nvfp4 \
  --device cuda
```

FP8 scaled:

```bash
python scripts/quantize_krea2.py \
  --input /path/to/krea2_raw_bf16.safetensors \
  --recipe fp8_scaled \
  --device cuda
```

MXFP8:

```bash
python scripts/quantize_krea2.py \
  --input /path/to/krea2_raw_bf16.safetensors \
  --recipe mxfp8 \
  --device cuda
```

## Dry Run

```bash
python scripts/quantize_krea2.py \
  --input /path/to/krea2_raw_bf16.safetensors \
  --recipe nvfp4 \
  --dry-run
```

## Options

```bash
python scripts/quantize_krea2.py --help
```

This help command works before installing the runtime dependencies. Quantization itself
still requires the packages in `requirements.txt`.

- `--recipe`: `nvfp4`, `fp8_scaled`, or `mxfp8`
- `--input`: input `.safetensors` file path, required
- `--output`: output `.safetensors` file path
- `--device`: quantization device, default `cuda`
- `--compute-dtype`: `bf16`, `fp16`, or `fp32`, default `bf16`
- `--dry-run`: validate input without writing output
- `--overwrite`: replace an existing output file. The input file can never be overwritten.

If `--output` is omitted, the output path defaults to the input file path with `_<recipe>` appended before `.safetensors`, for example `/path/to/krea2_raw_bf16_nvfp4.safetensors`.
If the output file already exists, the command exits with an error unless `--overwrite` is passed.
Using the same path for `--input` and `--output` is always an error.
If the input checkpoint does not match the selected recipe, the command exits with an error.

## Metadata Source

The metadata was obtained from https://huggingface.co/Comfy-Org/Krea-2.
