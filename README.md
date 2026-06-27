# krea2-quantizer

Quantize a Krea 2 diffusion-model `.safetensors` checkpoint for ComfyUI.

## Requirements

- Python 3.10+
- CUDA-capable PyTorch environment
- Krea 2 diffusion model `.safetensors`

## Setup

```bash
python -m pip install -r requirements.txt
```

Check CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

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

- `--recipe`: `nvfp4`, `fp8_scaled`, or `mxfp8`
- `--input`: input `.safetensors` file path, required
- `--output`: output `.safetensors` file path
- `--device`: quantization device, default `cuda`
- `--compute-dtype`: `bf16`, `fp16`, or `fp32`, default `bf16`
- `--dry-run`: validate input without writing output

If `--output` is omitted, the output path defaults to the input file path with `_<recipe>` appended before `.safetensors`, for example `/path/to/krea2_raw_bf16_nvfp4.safetensors`.
If the input checkpoint does not match the selected recipe, the command exits with an error.
