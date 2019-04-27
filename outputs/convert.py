import torch
import os

for f in os.listdir():
    if f[-3:] == '.pt':
        a = torch.load(f, map_location='cpu')
        a = a.squeeze().permute(1,0)
        print(a.size())
        torch.save(a, 'new_{}'.format(f))
