import torch

a = torch.randn((3,1))
print(a)
a.cuda(0)
print(a)
# utils.tensor_discrete(a)
# print(a)

flag = torch.cuda.is_available()
print(flag)
