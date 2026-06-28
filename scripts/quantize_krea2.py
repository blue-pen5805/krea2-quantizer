#!/usr/bin/env python3
"""
Quantize Krea 2 diffusion-model safetensors to ComfyUI mixed precision formats.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    "nvfp4": ROOT / "metadata" / "krea2_nvfp4_layers.json",
    "fp8_scaled": ROOT / "metadata" / "krea2_fp8_scaled_layers.json",
    "mxfp8": ROOT / "metadata" / "krea2_mxfp8_layers.json",
}

SUPPORTED_FORMATS = {"float8_e4m3fn", "mxfp8", "nvfp4"}

FP8_SOURCE_FORMATS = {
    "float8_e4m3fn",
    "float8_e5m2",
    "fp8",
    "fp8_e4m3fn",
    "fp8_scaled",
    "tensorcorefp8layout",
}
MXFP8_SOURCE_FORMATS = {"mxfp8", "tensorcoremxfp8layout"}
NVFP4_SOURCE_FORMATS = {"nvfp4", "tensorcorenvfp4layout"}
INT8_SOURCE_FORMATS = {"int8", "int8_tensorwise", "tensorwise_int8", "tensorwiseint8layout"}
AWQ_SOURCE_FORMATS = {"awq", "awq_w4a16", "w4a16", "int4_awq", "tensorcoreawqw4a16layout"}
SVDQUANT_SOURCE_FORMATS = {
    "svdquant",
    "svdquant_w4a4",
    "w4a4",
    "int4_svdquant",
    "tensorcoresvdquantw4a4layout",
}
SUPPORTED_SOURCE_FORMATS = (
    FP8_SOURCE_FORMATS
    | MXFP8_SOURCE_FORMATS
    | NVFP4_SOURCE_FORMATS
    | INT8_SOURCE_FORMATS
    | AWQ_SOURCE_FORMATS
    | SVDQUANT_SOURCE_FORMATS
)

MISSING_FP8_SCALE_WARNING = (
    "WARNING: FP8 source weights have no weight_scale tensors; "
    "using raw FP8 values directly. Original scaled values cannot be recovered."
)
MISSING_FP8_SCALE_WARNING_KEY = "missing_fp8_scale"

AUX_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_2",
    ".weight_zeros",
    ".weight_proj_down",
    ".weight_proj_up",
    ".weight_smooth_factor",
    ".input_scale",
    ".comfy_quant",
)

DTYPE_NAMES = ("bf16", "fp16", "fp32")

LAYER_PREFIXES = (
    "",
    "model.diffusion_model.",
)


def load_runtime_dependencies() -> Dict[str, Any]:
    global torch, safe_open, save_file, tqdm
    global QuantizedTensor, TensorCoreFP8Layout, TensorCoreMXFP8Layout, TensorCoreNVFP4Layout
    global TensorCoreAWQW4A16Layout, TensorCoreSVDQuantW4A4Layout, TensorWiseINT8Layout

    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        from tqdm import tqdm
        from comfy_kitchen.tensor import (
            QuantizedTensor,
            TensorCoreAWQW4A16Layout,
            TensorCoreFP8Layout,
            TensorCoreMXFP8Layout,
            TensorCoreNVFP4Layout,
            TensorCoreSVDQuantW4A4Layout,
            TensorWiseINT8Layout,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Failed to import runtime dependencies. Install dependencies first:\n"
            "  install a CUDA-capable PyTorch build first "
            "(CUDA 13.0 is recommended for nvfp4; see README.md)\n"
            "  python -m pip install -r requirements.txt\n"
            f"Original error: {exc}"
        )

    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }


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


def normalized_source_format(source_format: Optional[str]) -> Optional[str]:
    if source_format is None:
        return None
    return str(source_format).lower()


def infer_source_format(f, layer: str, tensor) -> Optional[str]:
    cfg = quantized_layer_config(f, layer)
    source_format = normalized_source_format(cfg.get("format") if cfg else None)
    if source_format:
        return source_format

    keys = f.keys()
    if f"{layer}.weight_proj_down" in keys or f"{layer}.weight_smooth_factor" in keys:
        return "svdquant_w4a4"
    if f"{layer}.weight_zeros" in keys:
        return "awq_w4a16"
    if str(tensor.dtype).startswith("torch.float8"):
        return "float8"
    if str(tensor.dtype) == "torch.int8" and f"{layer}.weight_scale" in keys:
        return "int8_tensorwise"
    return None


def validate_krea2_source(input_path: Path, layers: Mapping[str, Mapping[str, Any]]) -> str:
    missing = []
    not_2d = []
    not_supported = []
    missing_aux = []

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

            source_format = infer_source_format(f, full_layer, tensor)
            for aux_key in required_aux_keys_for_source_format(
                full_layer,
                source_format,
                tensor,
                keys,
            ):
                if aux_key not in keys:
                    missing_aux.append(aux_key)
            if tensor.is_floating_point():
                continue
            if source_format in SUPPORTED_SOURCE_FORMATS:
                continue
            not_supported.append((key, str(tensor.dtype), source_format))

    if not (missing or not_2d or not_supported or missing_aux):
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
    if missing_aux:
        print(f"  missing quantization auxiliary tensors: {len(missing_aux)}", file=sys.stderr)
        for key in missing_aux[:30]:
            print(f"    - {key}", file=sys.stderr)
    raise SystemExit(2)


def source_precision_summary(
    input_path: Path,
    layers: Mapping[str, Mapping[str, Any]],
    layer_prefix: str,
) -> tuple[Dict[str, int], Dict[str, int], bool]:
    dtype_counts: Dict[str, int] = {}
    format_counts: Dict[str, int] = {}
    has_unscaled_fp8 = False

    with safe_open(str(input_path), framework="pt", device="cpu") as f:
        for layer in layers:
            full_layer = f"{layer_prefix}{layer}"
            tensor = f.get_tensor(f"{full_layer}.weight")
            dtype_name = str(tensor.dtype).replace("torch.", "")
            dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1

            source_format = infer_source_format(f, full_layer, tensor) or "plain"
            format_counts[source_format] = format_counts.get(source_format, 0) + 1
            if (
                source_format in FP8_SOURCE_FORMATS
                or str(tensor.dtype).startswith("torch.float8")
            ) and f"{full_layer}.weight_scale" not in f.keys():
                has_unscaled_fp8 = True

    return dtype_counts, format_counts, has_unscaled_fp8


def format_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))


def warn_missing_fp8_scale_once(warned_missing_fp8_scale: set[str]) -> None:
    if MISSING_FP8_SCALE_WARNING_KEY in warned_missing_fp8_scale:
        return
    sys.stdout.flush()
    print(MISSING_FP8_SCALE_WARNING, file=sys.stderr)
    warned_missing_fp8_scale.add(MISSING_FP8_SCALE_WARNING_KEY)


def require_tensor(f, key: str):
    if key not in f.keys():
        raise ValueError(f"Missing source tensor: {key}")
    return f.get_tensor(key)


def required_aux_keys_for_source_format(
    layer: str,
    source_format: Optional[str],
    tensor,
    keys: set[str],
) -> tuple[str, ...]:
    looks_like_fp8 = str(tensor.dtype).startswith("torch.float8")

    if source_format in MXFP8_SOURCE_FORMATS:
        return (f"{layer}.weight_scale",)
    if source_format in NVFP4_SOURCE_FORMATS:
        return (f"{layer}.weight_scale", f"{layer}.weight_scale_2")
    if source_format in INT8_SOURCE_FORMATS:
        return (f"{layer}.weight_scale",)
    if source_format in AWQ_SOURCE_FORMATS:
        return (f"{layer}.weight_scale", f"{layer}.weight_zeros")
    if source_format in SVDQUANT_SOURCE_FORMATS:
        return (
            f"{layer}.weight_scale",
            f"{layer}.weight_proj_down",
            f"{layer}.weight_proj_up",
            f"{layer}.weight_smooth_factor",
        )
    if source_format in FP8_SOURCE_FORMATS or looks_like_fp8:
        scale_key = f"{layer}.weight_scale"
        return (scale_key,) if scale_key in keys else ()
    return ()


def infer_awq_group_size(qdata: torch.Tensor, scale: torch.Tensor, cfg: Mapping[str, Any]) -> int:
    if "group_size" in cfg:
        return int(cfg["group_size"])
    if "groupsize" in cfg:
        return int(cfg["groupsize"])
    if scale.ndim >= 1 and int(scale.shape[0]) > 0:
        return int(qdata.shape[1] * 2 // scale.shape[0])
    return 64


def infer_svdquant_orig_shape(qdata: torch.Tensor, smooth_factor: torch.Tensor) -> tuple[int, int]:
    out_features = TensorCoreSVDQuantW4A4Layout.get_out_features_from_storage(qdata)
    in_features = int(smooth_factor.shape[0])
    return (out_features, in_features)


def tensor_to_compute_dtype(
    f,
    layer: str,
    tensor: torch.Tensor,
    device: str,
    compute_dtype: torch.dtype,
    warned_missing_fp8_scale: set[str],
) -> torch.Tensor:
    cfg = quantized_layer_config(f, layer)
    if cfg is None:
        cfg = {}
    source_format = infer_source_format(f, layer, tensor)
    looks_like_fp8 = str(tensor.dtype).startswith("torch.float8")

    if source_format in MXFP8_SOURCE_FORMATS:
        scale_key = f"{layer}.weight_scale"
        scale = require_tensor(f, scale_key).detach().to(device=device)
        params = TensorCoreMXFP8Layout.Params(
            scale=scale,
            orig_dtype=compute_dtype,
            orig_shape=tuple(tensor.shape),
        )
        qdata = tensor.detach().to(device=device)
        return QuantizedTensor(qdata, "TensorCoreMXFP8Layout", params).dequantize().to(dtype=compute_dtype)

    if source_format in NVFP4_SOURCE_FORMATS:
        scale_key = f"{layer}.weight_scale_2"
        block_scale_key = f"{layer}.weight_scale"
        scale = require_tensor(f, scale_key).detach().to(device=device)
        block_scale = require_tensor(f, block_scale_key).detach().to(device=device)
        qdata = tensor.detach().to(device=device)
        params = TensorCoreNVFP4Layout.Params(
            scale=scale,
            block_scale=block_scale,
            orig_dtype=compute_dtype,
            orig_shape=TensorCoreNVFP4Layout.get_logical_shape_from_storage(tuple(tensor.shape)),
        )
        return QuantizedTensor(qdata, "TensorCoreNVFP4Layout", params).dequantize().to(dtype=compute_dtype)

    if source_format in INT8_SOURCE_FORMATS:
        scale_key = f"{layer}.weight_scale"
        scale = require_tensor(f, scale_key).detach().to(device=device)
        params = TensorWiseINT8Layout.Params(
            scale=scale,
            orig_dtype=compute_dtype,
            orig_shape=tuple(tensor.shape),
            is_weight=True,
            convrot=bool(cfg.get("convrot", False)),
            convrot_groupsize=int(cfg.get("convrot_groupsize", 256)),
        )
        qdata = tensor.detach().to(device=device)
        return QuantizedTensor(qdata, "TensorWiseINT8Layout", params).dequantize().to(dtype=compute_dtype)

    if source_format in AWQ_SOURCE_FORMATS:
        scale = require_tensor(f, f"{layer}.weight_scale").detach().to(device=device)
        zeros = require_tensor(f, f"{layer}.weight_zeros").detach().to(device=device)
        qdata = tensor.detach().to(device=device)
        params = TensorCoreAWQW4A16Layout.Params(
            scale=scale,
            zeros=zeros,
            orig_dtype=compute_dtype,
            orig_shape=(int(qdata.shape[0]), int(qdata.shape[1]) * 2),
            group_size=infer_awq_group_size(qdata, scale, cfg),
        )
        return QuantizedTensor(qdata, "TensorCoreAWQW4A16Layout", params).dequantize().to(dtype=compute_dtype)

    if source_format in SVDQUANT_SOURCE_FORMATS:
        scale = require_tensor(f, f"{layer}.weight_scale").detach().to(device=device)
        proj_down = require_tensor(f, f"{layer}.weight_proj_down").detach().to(device=device)
        proj_up = require_tensor(f, f"{layer}.weight_proj_up").detach().to(device=device)
        smooth_factor = require_tensor(f, f"{layer}.weight_smooth_factor").detach().to(device=device)
        qdata = tensor.detach().to(device=device)
        params = TensorCoreSVDQuantW4A4Layout.Params(
            scale=scale,
            proj_down=proj_down,
            proj_up=proj_up,
            smooth_factor=smooth_factor,
            orig_dtype=compute_dtype,
            orig_shape=infer_svdquant_orig_shape(qdata, smooth_factor),
            act_unsigned=bool(cfg.get("act_unsigned", False)),
        )
        return QuantizedTensor(qdata, "TensorCoreSVDQuantW4A4Layout", params).dequantize().to(dtype=compute_dtype)

    if source_format in FP8_SOURCE_FORMATS or looks_like_fp8:
        scale_key = f"{layer}.weight_scale"
        if scale_key not in f.keys():
            warn_missing_fp8_scale_once(warned_missing_fp8_scale)
            return tensor.to(device=device, dtype=compute_dtype, non_blocking=True)
        scale = require_tensor(f, scale_key).detach().to(device=device)
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


def merged_quantization_metadata(
    input_metadata: Mapping[str, str],
    quantized_layers: Mapping[str, Mapping[str, Any]],
) -> str:
    metadata_layers: Dict[str, Any] = {}
    existing_raw = input_metadata.get("_quantization_metadata")
    if existing_raw:
        try:
            existing = json.loads(existing_raw)
        except json.JSONDecodeError:
            existing = None
        if isinstance(existing, dict) and isinstance(existing.get("layers"), dict):
            metadata_layers.update(existing["layers"])

    metadata_layers.update(quantized_layers)
    return json.dumps(
        {"format_version": "1.0", "layers": metadata_layers},
        separators=(",", ":"),
        ensure_ascii=False,
    )


def save_file_atomically(
    tensors: Mapping[str, torch.Tensor],
    output_path: Path,
    metadata: Mapping[str, str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        save_file(dict(tensors), str(tmp_path), metadata=dict(metadata))
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def quantize_checkpoint(
    input_path: Path,
    output_path: Path,
    layers: Dict[str, Dict[str, Any]],
    layer_prefix: str,
    device: str,
    compute_dtype: torch.dtype,
    dry_run: bool,
    warned_missing_fp8_scale: Optional[set[str]] = None,
) -> None:
    target_weight_keys = {f"{layer_prefix}{layer}.weight": f"{layer_prefix}{layer}" for layer in layers}
    target_aux_keys = {
        f"{layer_prefix}{layer}{suffix}"
        for layer in layers
        for suffix in AUX_SUFFIXES
    }
    output: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    quantized_layers: Dict[str, Dict[str, Any]] = {}
    if warned_missing_fp8_scale is None:
        warned_missing_fp8_scale = set()

    with safe_open(str(input_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())

        if dry_run:
            print(f"Input tensors: {len(keys)}")
            print(f"Recipe layers: {len(layers)}")
            print("Dry run only; no file written.")
            return

        for key in tqdm(keys, desc="Converting tensors"):
            if key in target_aux_keys:
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
                warned_missing_fp8_scale=warned_missing_fp8_scale,
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
    output_metadata["_quantization_metadata"] = merged_quantization_metadata(
        output_metadata,
        quantized_layers,
    )

    save_file_atomically(output, output_path, output_metadata)

    print(f"Saved: {output_path}")
    print(f"Quantized layers: {len(quantized_layers)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a Krea 2 diffusion-model safetensors checkpoint."
    )
    parser.add_argument("--input", required=True, type=Path, help="Krea 2 safetensors")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output safetensors. Defaults to <input_stem>_<recipe>.safetensors next to input.",
    )
    parser.add_argument(
        "--recipe",
        choices=tuple(RECIPES.keys()),
        default="nvfp4",
        help="Built-in Krea 2 quantization recipe",
    )
    parser.add_argument("--device", default="cuda", help="Quantization device")
    parser.add_argument(
        "--compute-dtype",
        choices=DTYPE_NAMES,
        default="bf16",
        help="dtype used while quantizing weights",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not write output")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file. The input file can never be overwritten.",
    )
    return parser.parse_args()


def default_output_path(input_path: Path, recipe: str) -> Path:
    return input_path.with_name(f"{input_path.stem}_{recipe}.safetensors")


def validate_output_path(
    input_path: Path,
    output_path: Path,
    overwrite: bool,
    dry_run: bool,
) -> None:
    input_resolved = input_path.resolve(strict=False)
    output_resolved = output_path.resolve(strict=False)
    if output_resolved == input_resolved:
        raise SystemExit("Output path must be different from input path.")
    if not dry_run and output_path.exists() and not overwrite:
        raise SystemExit(
            f"Output file already exists: {output_path}\n"
            "Pass --overwrite to replace it."
        )


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(args.input, args.recipe)
    validate_output_path(args.input, output_path, args.overwrite, args.dry_run)
    dtypes = load_runtime_dependencies()

    layers = load_recipe(args.recipe)
    formats = sorted({str(cfg["format"]) for cfg in layers.values()})
    true_count = sum(1 for cfg in layers.values() if cfg.get("full_precision_matrix_mult", False))
    false_count = len(layers) - true_count
    print(f"Input: {args.input}")
    print(f"Output: {output_path}")
    print(f"Device: {args.device}")
    print(f"Compute dtype: {args.compute_dtype}")
    print(f"Recipe: {args.recipe}")
    print(f"Loaded layers: {len(layers)}")
    print(f"  target formats: {', '.join(formats)}")
    print(f"  full_precision_matrix_mult=true : {true_count}")
    print(f"  full_precision_matrix_mult=false: {false_count}")

    layer_prefix = validate_krea2_source(args.input, layers)
    print(f"Layer key prefix: {layer_prefix or '(none)'}")
    dtype_counts, source_format_counts, has_unscaled_fp8 = source_precision_summary(
        args.input, layers, layer_prefix
    )
    print(f"Input weight dtypes: {format_counts(dtype_counts)}")
    print(f"Input quant formats: {format_counts(source_format_counts)}")
    warned_missing_fp8_scale: set[str] = set()
    if has_unscaled_fp8 and not args.dry_run:
        warn_missing_fp8_scale_once(warned_missing_fp8_scale)
    quantize_checkpoint(
        input_path=args.input,
        output_path=output_path,
        layers=layers,
        layer_prefix=layer_prefix,
        device=args.device,
        compute_dtype=dtypes[args.compute_dtype],
        dry_run=args.dry_run,
        warned_missing_fp8_scale=warned_missing_fp8_scale,
    )


if __name__ == "__main__":
    main()
