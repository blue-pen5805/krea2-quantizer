# krea2-quantizer

[ENGLISH](README.md) | [日本語](README_JP.md)

ComfyUI 向けに Krea 2 拡散モデルの `.safetensors` チェックポイントを量子化します。

## 使い方

AI エージェントに次のように伝えてください。

```text
https://github.com/blue-pen5805/krea2-quantizer のリポジトリをセットアップしてください。
環境セットアップが完了した時点で停止してください。
その後、量子化を実行する前に、私の Krea 2 diffusion_model .safetensors ファイルの場所を確認してください。
```

または、以下の手動セットアップ手順に従ってください。

## 要件

- Python 3.10 以上
- CUDA 対応 PyTorch 環境を利用できる NVIDIA GPU
- インストールする PyTorch CUDA wheel と互換性のある新しめの NVIDIA ドライバー
- Krea 2 diffusion_model の `.safetensors`

残りの依存関係をインストールする前に、PyTorch を別途インストールしてください。
`nvfp4` レシピには NVFP4 サポートが必要なため、CUDA 13.0 を推奨します。

CUDA 13.0 が使える環境の場合:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch
```

別の CUDA wheel が必要な環境の場合は、公式 PyTorch 手順に従って対応する
PyTorch ビルドを先にインストールし、その後で残りの依存関係をインストールしてください。

## セットアップ

PyTorch をインストールした後:

```bash
python -m pip install -r requirements.txt
```

CUDA を確認します。

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

`--device cuda` を使う場合、最後の行は `True` と表示される必要があります。
`False` と表示される場合は、量子化を実行する前に PyTorch/CUDA のインストールを修正してください。

## レシピ

- `nvfp4`
- `fp8_scaled`
- `mxfp8`

## 量子化

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

## ドライラン

```bash
python scripts/quantize_krea2.py \
  --input /path/to/krea2_raw_bf16.safetensors \
  --recipe nvfp4 \
  --dry-run
```

## オプション

```bash
python scripts/quantize_krea2.py --help
```

このヘルプコマンドは、実行時依存関係をインストールする前でも動作します。
量子化そのものには `requirements.txt` 内のパッケージが必要です。

- `--recipe`: `nvfp4`、`fp8_scaled`、または `mxfp8`
- `--input`: 入力 `.safetensors` ファイルパス。必須です。
- `--output`: 出力 `.safetensors` ファイルパス
- `--device`: 量子化に使うデバイス。デフォルトは `cuda`
- `--compute-dtype`: `bf16`、`fp16`、または `fp32`。デフォルトは `bf16`
- `--dry-run`: 出力を書き込まずに入力を検証します。
- `--overwrite`: 既存の出力ファイルを置き換えます。入力ファイルは決して上書きできません。

`--output` を省略した場合、出力パスは入力ファイルパスの `.safetensors` の前に
`_<recipe>` を付けたものになります。例: `/path/to/krea2_raw_bf16_nvfp4.safetensors`。
出力ファイルがすでに存在する場合、`--overwrite` を指定しない限りコマンドはエラーで終了します。
`--input` と `--output` に同じパスを使うことは常にエラーです。
入力チェックポイントが選択したレシピに一致しない場合、コマンドはエラーで終了します。

## メタデータの出典

メタデータは https://huggingface.co/Comfy-Org/Krea-2 から取得しました。
