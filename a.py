import torch
import utils

a = torch.randn((3,1))
print(a)
utils.tensor_discrete(a)
print(a)