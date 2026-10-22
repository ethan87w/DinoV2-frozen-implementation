# Frozen DINOv2

Turns robot images into embeddings and finds similar images from other episodes.

## Setup (Windows / VS Code)

1. Install Python 3.12. Download this repository and open its folder in VS Code.
2. Put your shared dataset here:

```text
data/metadata.csv
data/images/
```

Place all JPEG images inside `data/images/` and your `metadata.csv` directly inside `data/`. Keep CSV headers unchanged; its `image_path` values should look like `images/filename.jpg`. The folders contain placeholders only; add your own shared dataset locally.

3. Run these commands in the VS Code PowerShell terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install pillow matplotlib
```

## Run

Generate and save embeddings (first run downloads DINOv2):

```powershell
.\.venv\Scripts\python.exe test_dino.py
```

Compare an image with the saved embeddings:

```powershell
.\.venv\Scripts\python.exe retrieve.py --query 25
```

Change `25` to an image index (0–199 for 200 images). Results are displayed and saved in `outputs/`. You only need to regenerate embeddings if the images, model, or preprocessing change.

Uses frozen [DINOv2 ViT-S/14](https://github.com/facebookresearch/dinov2). No training. Keep the original CSV headers and image paths. Do not upload datasets, model weights, or generated outputs to GitHub.

The original `data/bridge_sample/` layout also works automatically. Panels and terminal results show each image's scene from the saved metadata. If you update scene values in the CSV, regenerate embeddings to refresh that metadata.

You can also run `compare_embed.py --query 25` instead of `retrieve.py --query 25` (using the same Python environment).
