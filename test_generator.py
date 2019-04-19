import torch
import torch.nn as nn

from transformer.mem_transformer import MemTransformer

in_data = torch.rand([4, 2, 8])
in_style = torch.rand([1, 2, 16])
out_target = torch.rand([4, 2, 8])


model = MemTransformer(2, 4, 8, 4, 8, 16, 0.5, 0.5, tgt_len=4, ext_len=4, mem_len=4)
pred = model.forward(in_data, in_style)
criterion = nn.MSELoss()
loss = criterion(pred, out_target)
print(loss)
loss.backward()
print(sum(map(lambda x: x.grad.detach().sum().abs(), list(model.parameters()))))
from torchsummary import summary
summary(model, [(4,2,8),(1,2,16)])
print("Done")
