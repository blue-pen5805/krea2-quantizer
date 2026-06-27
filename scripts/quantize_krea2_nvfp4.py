#!/usr/bin/env python3
"""
Quantize Krea 2 diffusion-model safetensors to ComfyUI NVFP4 format.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

try:
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout
except Exception as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Failed to import comfy_kitchen. Install dependencies first:\n"
        "  uv pip install -r requirements.txt\n"
        f"Original error: {exc}"
    )


ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    "krea2": ROOT / "metadata" / "krea2_nvfp4_layers.json",
}

AUX_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
    ".comfy_quant",
)

DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

LAYER_PREFIXES = (
    "",
    "model.diffusion_model.",
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_recipe(name: str) -> Dict[str, Dict[str, Any]]:
    path = RECIPES[name]
    raw = read_json(path)
    layers = raw.get("layers") if isinstance(raw, dict) else None
    if not isinstance(layers, dict):
        raise ValueError(f"Recipe has no layers dict: {path}")

    normalized: Dict[str, Dict[str, Any]] = {}
    for layer, cfg_in in layers.items():
        if not isinstance(cfg_in, dict):
            raise ValueError(f"Invalid layer config for {layer!r} in {path}")
        cfg = dict(cfg_in)
        if cfg.get("format") != "nvfp4":
            raise ValueError(f"Layer {layer!r} is not nvfp4 in {path}")
        normalized[str(layer)] = cfg
    return normalized


def safetensors_metadata(path: Path) -> Dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return dict(f.metadata() or {})


def detect_layer_prefix(keys: set[str], layers: Mapping[str, Mapping[str, Any]]) -> str:
    for prefix in LAYER_PREFIXES:
        if all(f"{prefix}{layer}.weight" in keys for layer in layers):
            return prefix
    return ""


def validate_krea2_source(input_path: Path, layers: Mapping[str, Mapping[str, Any]]) -> str:
    missing = []
    not_2d = []
    not_float = []

    with safe_open(str(input_path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        prefix = detect_layer_prefix(keys, layers)
        for layer in layers:
            key = f"{prefix}{layer}.weight"
            if key not in keys:
                missing.append(key)
                continue
            tensor = f.get_tensor(key)
            if tensor.ndim != 2:
                not_2d.append((key, tuple(tensor.shape)))
            if not tensor.is_floating_point():
                not_float.append((key, str(tensor.dtype)))

    if not (missing or not_2d or not_float):
        return prefix

    print("Input is not compatible with the selected Krea 2 recipe:", file=sys.stderr)
    if missing:
        print(f"  missing .weight tensors: {len(missing)}", file=sys.stderr)
        for key in missing[:30]:
            print(f"    - {key}", file=sys.stderr)
    if not_2d:
        print(f"  non-2D .weight tensors: {len(not_2d)}", file=sys.stderr)
        for key, shape in not_2d[:30]:
            print(f"    - {key}: {shape}", file=sys.stderr)
    if not_float:
        print(f"  non-floating .weight tensors: {len(not_float)}", file=sys.stderr)
        for key, dtype in not_float[:30]:
            print(f"    - {key}: {dtype}", file=sys.stderr)
    raise SystemExit(2)


def quantize_one_weight(tensor: torch.Tensor, device: str, compute_dtype: torch.dtype):
    x = tensor.to(device=device, dtype=compute_dtype, non_blocking=True)
    qdata, params = TensorCoreNVFP4Layout.quantize(x, scale="recalculate")
    return (
        qdata.detach().cpu().contiguous(),
        params.block_scale.detach().cpu().contiguous(),
        params.scale.detach().cpu().to(torch.float32).reshape(()),
    )


def quantize_checkpoint(
    input_path: Path,
    output_path: Path,
    layers: Dict[str, Dict[str, Any]],
    layer_prefix: str,
    device: str,
    compute_dtype: torch.dtype,
    dry_run: bool,
) -> None:
    target_weight_keys = {f"{layer_prefix}{layer}.weight": f"{layer_prefix}{layer}" for layer in layers}
    output: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    quantized_layers: Dict[str, Dict[str, Any]] = {}

    with safe_open(str(input_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())

        if dry_run:
            print(f"Input tensors: {len(keys)}")
            print(f"Recipe NVFP4 layers: {len(layers)}")
            print(f"Layer key prefix: {layer_prefix or '(none)'}")
            print("Dry run only; no file written.")
            return

        for key in tqdm(keys, desc="Converting tensors"):
            if key.endswith(AUX_SUFFIXES):
                continue

            tensor = f.get_tensor(key)
            layer = target_weight_keys.get(key)
            if layer is None:
                output[key] = tensor
                continue

            qdata, weight_scale, weight_scale_2 = quantize_one_weight(tensor, device, compute_dtype)
            output[key] = qdata
            output[f"{layer}.weight_scale"] = weight_scale
            output[f"{layer}.weight_scale_2"] = weight_scale_2
            recipe_layer = layer.removeprefix(layer_prefix)
            quantized_layers[layer] = dict(layers[recipe_layer])

            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    output_metadata = safetensors_metadata(input_path)
    output_metadata.pop("_quantization_metadata", None)
    output_metadata["_quantization_metadata"] = json.dumps(
        {"format_version": "1.0", "layers": quantized_layers},
        separators=(",", ":"),
        ensure_ascii=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, str(output_path), metadata=output_metadata)

    print(f"Saved: {output_path}")
    print(f"Quantized layers: {len(quantized_layers)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a Krea 2 diffusion-model safetensors checkpoint to NVFP4."
    )
    parser.add_argument("--input", required=True, type=Path, help="Krea 2 BF16/FP16/FP32 safetensors")
    parser.add_argument("--output", required=True, type=Path, help="Output NVFP4 safetensors")
    parser.add_argument(
        "--recipe",
        choices=tuple(RECIPES.keys()),
        default="krea2",
        help="Built-in Krea 2 quantization recipe",
    )
    parser.add_argument("--device", default="cuda", help="Quantization device")
    parser.add_argument(
        "--compute-dtype",
        choices=tuple(DTYPES.keys()),
        default="bf16",
        help="dtype used while quantizing weights",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not write output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    layers = load_recipe(args.recipe)
    true_count = sum(1 for cfg in layers.values() if cfg.get("full_precision_matrix_mult", False))
    false_count = len(layers) - true_count
    print(f"Recipe: {args.recipe}")
    print(f"Loaded NVFP4 layers: {len(layers)}")
    print(f"  full_precision_matrix_mult=true : {true_count}")
    print(f"  full_precision_matrix_mult=false: {false_count}")

    layer_prefix = validate_krea2_source(args.input, layers)
    print(f"Layer key prefix: {layer_prefix or '(none)'}")
    quantize_checkpoint(
        input_path=args.input,
        output_path=args.output,
        layers=layers,
        layer_prefix=layer_prefix,
        device=args.device,
        compute_dtype=DTYPES[args.compute_dtype],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
