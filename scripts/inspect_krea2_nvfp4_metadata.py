#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    "nvfp4": ROOT / "metadata" / "krea2_nvfp4_layers.json",
    "fp8_scaled": ROOT / "metadata" / "krea2_fp8_scaled_layers.json",
    "mxfp8": ROOT / "metadata" / "krea2_mxfp8_layers.json",
    "krea2": ROOT / "metadata" / "krea2_nvfp4_layers.json",
}

LAYER_PREFIXES = (
    "",
    "model.diffusion_model.",
)


def load_layers(recipe: str):
    with RECIPES[recipe].open("r", encoding="utf-8") as f:
        return json.load(f)["layers"]


def detect_layer_prefix(keys: set[str], layers) -> str:
    for prefix in LAYER_PREFIXES:
        if all(f"{prefix}{layer}.weight" in keys for layer in layers):
            return prefix
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a built-in Krea 2 quantization recipe.")
    parser.add_argument(
        "--recipe",
        choices=tuple(RECIPES.keys()),
        default="nvfp4",
        help="Built-in Krea 2 recipe",
    )
    parser.add_argument("--source", type=Path, help="Optional Krea 2 safetensors to validate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = load_layers(args.recipe)

    true_count = sum(1 for cfg in layers.values() if cfg.get("full_precision_matrix_mult", False))
    false_count = len(layers) - true_count
    formats = sorted({str(cfg["format"]) for cfg in layers.values()})
    print(f"recipe: {args.recipe}")
    print(f"layers: {len(layers)}")
    print(f"formats: {', '.join(formats)}")
    print(f"full_precision_matrix_mult=true : {true_count}")
    print(f"full_precision_matrix_mult=false: {false_count}")

    if not args.source:
        return

    missing = []
    not_2d = []
    with safe_open(str(args.source), framework="pt", device="cpu") as f:
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

    print(f"layer key prefix: {prefix or '(none)'}")
    print(f"missing .weight keys: {len(missing)}")
    print(f"non-2D .weight keys: {len(not_2d)}")
    for key in missing[:20]:
        print(f"missing: {key}")
    for key, shape in not_2d[:20]:
        print(f"not_2d: {key} {shape}")


if __name__ == "__main__":
    main()
