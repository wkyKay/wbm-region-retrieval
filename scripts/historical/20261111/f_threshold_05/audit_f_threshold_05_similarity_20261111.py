"""Audit all directed cross-wafer region cosines, without labels or parameter tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--regions', required=True)
a = p.parse_args()
root = Path(a.run)
x = torch.from_numpy(np.load(root / 'test_features.npy')).to('cuda:0', dtype=torch.float64)
with np.load(a.regions) as z:
    ids = np.repeat(z['map_ids'], np.diff(z['offsets']))
ids = torch.from_numpy(ids).to('cuda:0')
minimum, maximum, total, below = float('inf'), -float('inf'), 0, 0
with torch.no_grad():
    for start in range(0, len(x), 512):
        cos = x[start:start+512] @ x.T
        mask = ids[start:start+512, None] != ids[None, :]
        values = cos[mask]
        minimum = min(minimum, values.min().item())
        maximum = max(maximum, values.max().item())
        total += values.numel()
        below += int((values < 0.5).sum().item())
report = {'region_count': len(x), 'directed_cross_wafer_region_pairs': total,
          'cosine_min': minimum, 'cosine_max': maximum, 'pairs_below_05': below,
          'fraction_below_05': below/total, 'precision': 'float64',
          'same_wafer_pairs_excluded': True, 'purpose': 'Explain fixed threshold result; no threshold search'}
with (root / 'similarity_audit.json').open('x') as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
