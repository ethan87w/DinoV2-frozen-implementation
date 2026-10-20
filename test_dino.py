"""Generate frozen DINOv2 embeddings for the shared Bridge sample."""
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    with (args.data / 'metadata.csv').open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        required = {'image_id', 'image_path', 'episode_id', 'frame_index'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('CSV must contain: ' + ', '.join(sorted(required)))
        rows = list(reader)
    if args.limit:
        rows = rows[:args.limit]
    if not rows or len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Metadata must be nonempty with unique image IDs')
    for row in rows:
        if not (args.data / row['image_path']).is_file():
            raise FileNotFoundError(args.data / row['image_path'])
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    print('Loading pretrained dinov2_vits14 on CPU...', flush=True)
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                           pretrained=True, trust_repo=True)
    model.eval()
    model.requires_grad_(False)
    vectors = []
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            with Image.open(args.data / row['image_path']) as image:
                tensor = preprocess(image.convert('RGB')).unsqueeze(0)
            vector = model.forward_features(tensor)['x_norm_clstoken']
            if vector.shape != (1, 384) or not torch.isfinite(vector).all().item():
                raise ValueError('Invalid embedding for ' + row['image_id'])
            vectors.append(vector.squeeze(0))
            print(f"Encoded {index}/{len(rows)}: {row['image_id']}", flush=True)
    embeddings = torch.stack(vectors)
    output = ROOT / 'outputs'
    output.mkdir(exist_ok=True)
    path = output / ('embeddings_all.pt' if args.limit is None else f'embeddings_{args.limit}.pt')
    torch.save({
        'embeddings': embeddings, 'metadata': rows,
        'encoder': 'dinov2_vits14', 'checkpoint': 'dinov2_vits14_pretrain.pth',
        'output': 'x_norm_clstoken', 'torch_version': str(torch.__version__),
        'preprocessing': {'resize': 256, 'crop': 224, 'interpolation': 'bicubic',
                          'color': 'RGB', 'mean': [0.485, 0.456, 0.406],
                          'std': [0.229, 0.224, 0.225]},
    }, path)
    print('Saved embedding shape:', embeddings.shape)
    print('Saved to:', path)


if __name__ == '__main__':
    main()
