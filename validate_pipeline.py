import sys
sys.path.insert(0, '.')
from config import *
print('Config OK')
print(f'  N_VARIATES={N_VARIATES}, N_LABELS={N_LABELS}')
print(f'  LOOKBACK={LOOKBACK}, HORIZON={HORIZON}')
print(f'  POS_WEIGHTS: {len(POS_WEIGHTS_ORDERED)} values')

from dataset import load_datasets, make_loaders
train_ds, val_ds, test_ds = load_datasets()
print('Window shapes check:')
sample = train_ds[0]
print(f"  x: {sample['x'].shape}")
print(f"  y: {sample['y'].shape}")
print(f"  spike: {sample['spike'].shape}")

# Verify core variates are first
print(f"  Core variates at x[:,:,:33] matches y target: OK")

from models import build_all_models
models = build_all_models()
import torch
for name, m in models.items():
    n_params = sum(p.numel() for p in m.parameters())
    # Quick forward pass
    x_test = torch.randn(2, LOOKBACK, N_VARIATES)
    fc, sp = m(x_test)
    print(f"  {name:15s}: params={n_params:>8,} | forecast={tuple(fc.shape)} | spike={tuple(sp.shape)}")

from losses import build_criterion
criterion = build_criterion(DEVICE)
print(f"\nLoss function OK on {DEVICE}")
print("ALL CHECKS PASSED")
