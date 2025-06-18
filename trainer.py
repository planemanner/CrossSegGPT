import torch

a = torch.randn(2, 32, 32, 128)
l = torch.nn.Linear(128, 256)
print(l(a).shape)