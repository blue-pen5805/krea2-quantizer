# krea2-nvfp4-quantizer

Krea 2 diffusion-model `.safetensors` を ComfyUI 用 NVFP4 checkpoint に変換します。

## セットアップ

```powershell
uv venv
uv pip install -r requirements.txt
```

CUDA の確認:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## 検査

```powershell
.\.venv\Scripts\python.exe scripts\inspect_krea2_nvfp4_metadata.py `
  --source "L:\models\krea2\diffusion_models\krea2_raw_bf16.safetensors"
```

成功時の目安:

```text
missing .weight keys: 0
non-2D .weight keys: 0
```

## 変換

```powershell
.\.venv\Scripts\python.exe scripts\quantize_krea2_nvfp4.py `
  --input "L:\models\krea2\diffusion_models\krea2_raw_bf16.safetensors" `
  --output "L:\models\krea2\diffusion_models\krea2_raw_nvfp4.safetensors" `
  --device cuda
```

## Dry Run

```powershell
.\.venv\Scripts\python.exe scripts\quantize_krea2_nvfp4.py `
  --input "L:\models\krea2\diffusion_models\krea2_raw_bf16.safetensors" `
  --output "L:\models\krea2\diffusion_models\dummy.safetensors" `
  --dry-run
```

## 出力確認

```powershell
.\.venv\Scripts\python.exe scripts\inspect_krea2_nvfp4_metadata.py `
  --source "L:\models\krea2\diffusion_models\krea2_raw_nvfp4.safetensors"
```
