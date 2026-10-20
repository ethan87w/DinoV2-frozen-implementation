"""Rank saved embeddings and display cross-episode image matches."""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', type=int, default=0)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'bridge_sample')
    parser.add_argument('--embeddings', type=Path, default=ROOT / 'outputs' / 'embeddings_all.pt')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--print-embedding', action='store_true')
    args = parser.parse_args()
    saved = torch.load(args.embeddings, map_location='cpu', weights_only=True)
    embeddings, metadata = saved['embeddings'], saved['metadata']
    if embeddings.shape != (len(metadata), 384) or not torch.isfinite(embeddings).all().item():
        raise ValueError('Embeddings must be finite and match metadata rows')
    if len({r['image_id'] for r in metadata}) != len(metadata):
        raise ValueError('Duplicate image IDs')
    print(f'Loaded images: {len(metadata)}; valid query indices: 0-{len(metadata)-1}')
    if not 0 <= args.query < len(metadata):
        parser.error('--query is outside the saved image range')
    if args.top_k < 1:
        parser.error('--top-k must be positive')
    query = metadata[args.query]
    normalized = F.normalize(embeddings, p=2, dim=1)
    scores = normalized @ normalized[args.query]
    candidates = [i for i, r in enumerate(metadata) if r['episode_id'] != query['episode_id']]
    if not candidates:
        raise ValueError('No images from other episodes; encode a larger sample')
    ranked = sorted(candidates, key=lambda i: scores[i].item(), reverse=True)[:args.top_k]
    print('Query:', query['image_id'], '| episode:', query['episode_id'])
    print('Eligible images:', len(candidates))
    if args.print_embedding:
        print('Query embedding:', embeddings[args.query].tolist())
    for rank, i in enumerate(ranked, 1):
        assert metadata[i]['episode_id'] != query['episode_id']
        print(f"{rank}. {metadata[i]['image_id']} | episode: {metadata[i]['episode_id']} | cosine: {scores[i].item():.4f}")
    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    indices = [args.query] + ranked
    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 5), squeeze=False)
    for position, i in enumerate(indices):
        row = metadata[i]
        with Image.open(args.data / row['image_path']) as image:
            axes[0, position].imshow(image.convert('RGB'))
        title = 'QUERY' if position == 0 else f'Match {position} | cosine: {scores[i].item():.3f}'
        a, b = str(query.get('scene') or 'unknown').strip(), str(row.get('scene') or 'unknown').strip()
        unknown = {'', 'unknown', 'none', 'nan'}
        relation = 'unknown' if a.lower() in unknown or b.lower() in unknown else ('same' if a == b else 'different')
        label = 'Reference image' if position == 0 else f'Scene relationship: {relation}'
        axes[0, position].set_title(f"{title}\nEpisode: {row['episode_id']}\nFrame: {row['frame_index']}\n{label}", fontsize=9)
        axes[0, position].axis('off')
    fig.suptitle('Frozen DINOv2 retrieval - query episode excluded')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output = ROOT / 'outputs'
    output.mkdir(exist_ok=True)
    path = output / f'retrieval_query_{args.query}.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    print('Saved comparison panel:', path)
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
