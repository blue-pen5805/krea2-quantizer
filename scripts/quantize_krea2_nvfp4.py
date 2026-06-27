#!/usr/bin/env python3
"""
Quantize Krea 2 diffusion-model safetensors to ComfyUI mixed precision formats.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

try:
    from comfy_kitchen.tensor import (
        QuantizedTensor,
        TensorCoreFP8Layout,
        TensorCoreMXFP8Layout,
        TensorCoreNVFP4Layout,
    )
except Exception as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Failed to import comfy_kitchen. Install dependencies first:\n"
        "  uv pip install -r requirements.txt\n"
        f"Original error: {exc}"
    )


ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    "nvfp4": ROOT / "metadata" / "krea2_nvfp4_layers.json",
    "fp8_scaled": ROOT / "metadata" / "krea2_fp8_scaled_layers.json",
    "mxfp8": ROOT / "metadata" / "krea2_mxfp8_layers.json",
    "krea2": ROOT / "metadata" / "krea2_nvfp4_layers.json",
}

SUPPORTED_FORMATS = {"float8_e4m3fn", "mxfp8", "nvfp4"}

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
        if cfg.get("format") not in SUPPORTED_FORMATS:
            raise ValueError(f"Layer {layer!r} has unsupported format: {cfg.get('format')!r}")
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


def quantized_layer_config(f, layer: str) -> Optional[Dict[str, Any]]:
    key = f"{layer}.comfy_quant"
    if key not in f.keys():
        return None
    raw = f.get_tensor(key).numpy().tobytes()
    return json.loads(raw)


def validate_krea2_source(input_path: Path, layers: Mapping[str, Mapping[str, Any]]) -> str:
    missing = []
    not_2d = []
    not_supported = []

    with safe_open(str(input_path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        prefix = detect_layer_prefix(keys, layers)
        for layer in layers:
            full_layer = f"{prefix}{layer}"
            key = f"{full_layer}.weight"
            if key not in keys:
                missing.append(key)
                continue
            tensor = f.get_tensor(key)
            if tensor.ndim != 2:
                not_2d.append((key, tuple(tensor.shape)))

            cfg = quantized_layer_config(f, full_layer)
            source_format = cfg.get("format") if cfg else None
            if tensor.is_floating_point():
                continue
            if source_format == "nvfp4":
                continue
            not_supported.append((key, str(tensor.dtype), source_format))

    if not (missing or not_2d or not_supported):
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
    if not_supported:
        print(f"  unsupported .weight tensors: {len(not_supported)}", file=sys.stderr)
        for key, dtype, source_format in not_supported[:30]:
            print(f"    - {key}: {dtype}, format={source_format}", file=sys.stderr)
    raise SystemExit(2)


def source_precision_summary(
    input_path: Path,
    layers: Mapping[str, Mapping[str, Any]],
    layer_prefix: str,
) -> tuple[Dict[str, int], Dict[str, int]]:
    dtype_counts: Dict[str, int] = {}
    format_counts: Dict[str, int] = {}

    with safe_open(str(input_path), framework="pt", device="cpu") as f:
        for layer in layers:
            full_layer = f"{layer_prefix}{layer}"
            tensor = f.get_tensor(f"{full_layer}.weight")
            dtype_name = str(tensor.dtype).replace("torch.", "")
            dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1

            cfg = quantized_layer_config(f, full_layer)
            source_format = str(cfg.get("format")) if cfg and cfg.get("format") else "plain"
            format_counts[source_format] = format_counts.get(source_format, 0) + 1

    return dtype_counts, format_counts


def format_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))


def tensor_to_compute_dtype(
    f,
    layer: str,
    tensor: torch.Tensor,
    device: str,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    cfg = quantized_layer_config(f, layer)
    source_format = cfg.get("format") if cfg else None
    looks_like_fp8 = str(tensor.dtype).startswith("torch.float8")

    if source_format == "mxfp8":
        scale_key = f"{layer}.weight_scale"
        if scale_key not in f.keys():
            raise ValueError(f"Missing MXFP8 weight_scale for source layer: {layer}")
        scale = f.get_tensor(scale_key).detach().to(device=device)
        params = TensorCoreMXFP8Layout.Params(
            scale=scale,
            orig_dtype=compute_dtype,
            orig_shape=tuple(tensor.shape),
        )
        qdata = tensor.detach().to(device=device)
        return QuantizedTensor(qdata, "TensorCoreMXFP8Layout", params).dequantize().to(dtype=compute_dtype)

    if source_format == "nvfp4":
        scale_key = f"{layer}.weight_scale_2"
        block_scale_key = f"{layer}.weight_scale"
        if scale_key not in f.keys() or block_scale_key not in f.keys():
            raise ValueError(f"Missing NVFP4 scales for source layer: {layer}")
        scale = f.get_tensor(scale_key).detach().to(device=device)
        block_scale = f.get_tensor(block_scale_key).detach().to(device=device)
        qdata = tensor.detach().to(device=device)
        params = TensorCoreNVFP4Layout.Params(
            scale=scale,
            block_scale=block_scale,
            orig_dtype=compute_dtype,
            orig_shape=TensorCoreNVFP4Layout.get_logical_shape_from_storage(tuple(tensor.shape)),
        )
        return QuantizedTensor(qdata, "TensorCoreNVFP4Layout", params).dequantize().to(dtype=compute_dtype)

    if source_format in {"float8_e4m3fn", "float8_e5m2"} or looks_like_fp8:
        scale_key = f"{layer}.weight_scale"
        if scale_key not in f.keys():
            raise ValueError(f"Missing FP8 weight_scale for source layer: {layer}")
        scale = f.get_tensor(scale_key).detach().to(device=device)
        params = TensorCoreFP8Layout.Params(
            scale=scale,
            orig_dtype=compute_dtype,
            orig_shape=tuple(tensor.shape),
        )
        qdata = tensor.detach().to(device=device)
        return QuantizedTensor(qdata, "TensorCoreFP8Layout", params).dequantize().to(dtype=compute_dtype)

    return tensor.to(device=device, dtype=compute_dtype, non_blocking=True)


def state_dict_tensors_for_format(
    tensor: torch.Tensor,
    target_format: str,
) -> Dict[str, torch.Tensor]:
    if target_format == "float8_e4m3fn":
        qdata, params = TensorCoreFP8Layout.quantize(
            tensor,
            scale="recalculate",
            dtype=torch.float8_e4m3fn,
        )
        return TensorCoreFP8Layout.state_dict_tensors(qdata, params)

    if target_format == "mxfp8":
        qdata, params = TensorCoreMXFP8Layout.quantize(tensor)
        return TensorCoreMXFP8Layout.state_dict_tensors(qdata, params)

    if target_format == "nvfp4":
        qdata, params = TensorCoreNVFP4Layout.quantize(tensor, scale="recalculate")
        return TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)

    raise ValueError(f"Unsupported target format: {target_format}")


def output_key_for_state_tensor(layer: str, state_suffix: str) -> str:
    suffixes = {
        "": ".weight",
        "_scale": ".weight_scale",
        "_scale_2": ".weight_scale_2",
    }
    if state_suffix not in suffixes:
        raise ValueError(f"Unsupported quantized state tensor suffix: {state_suffix!r}")
    return f"{layer}{suffixes[state_suffix]}"


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
            print(f"Recipe layers: {len(layers)}")
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

            recipe_layer = layer.removeprefix(layer_prefix)
            layer_config = dict(layers[recipe_layer])
            x = tensor_to_compute_dtype(
                f=f,
                layer=layer,
                tensor=tensor,
                device=device,
                compute_dtype=compute_dtype,
            )
            for state_suffix, state_tensor in state_dict_tensors_for_format(
                x,
                str(layer_config["format"]),
            ).items():
                output[output_key_for_state_tensor(layer, state_suffix)] = (
                    state_tensor.detach().cpu().contiguous()
                )

            output[f"{layer}.comfy_quant"] = torch.tensor(
                list(json.dumps(layer_config).encode("utf-8")),
                dtype=torch.uint8,
            )
            quantized_layers[recipe_layer] = layer_config

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
        description="Quantize a Krea 2 diffusion-model safetensors checkpoint."
    )
    parser.add_argument("--input", required=True, type=Path, help="Krea 2 safetensors")
    parser.add_argument("--output", required=True, type=Path, help="Output safetensors")
    parser.add_argument(
        "--recipe",
        choices=tuple(RECIPES.keys()),
        default="nvfp4",
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
    formats = sorted({str(cfg["format"]) for cfg in layers.values()})
    true_count = sum(1 for cfg in layers.values() if cfg.get("full_precision_matrix_mult", False))
    false_count = len(layers) - true_count
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print(f"Compute dtype: {args.compute_dtype}")
    print(f"Recipe: {args.recipe}")
    print(f"Loaded layers: {len(layers)}")
    print(f"  target formats: {', '.join(formats)}")
    print(f"  full_precision_matrix_mult=true : {true_count}")
    print(f"  full_precision_matrix_mult=false: {false_count}")

    layer_prefix = validate_krea2_source(args.input, layers)
    print(f"Layer key prefix: {layer_prefix or '(none)'}")
    dtype_counts, source_format_counts = source_precision_summary(args.input, layers, layer_prefix)
    print(f"Input weight dtypes: {format_counts(dtype_counts)}")
    print(f"Input quant formats: {format_counts(source_format_counts)}")
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
