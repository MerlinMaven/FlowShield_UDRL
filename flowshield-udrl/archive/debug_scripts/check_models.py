import torch

# Check all model architectures
models = [
    '../results/lunarlander/models/quantile_shield.pt',
    '../results/lunarlander/models/flow_shield.pt',
    '../results/lunarlander/models/diffusion_shield.pt'
]

for m in models:
    name = m.split('/')[-1]
    print(f'\n=== {name} ===')
    cp = torch.load(m, weights_only=False)
    for k in list(cp.keys())[:10]:
        if hasattr(cp[k], 'shape'):
            print(f'  {k}: {cp[k].shape}')
        else:
            print(f'  {k}: {type(cp[k]).__name__}')
    if len(cp) > 10:
        print(f'  ... and {len(cp)-10} more keys')
