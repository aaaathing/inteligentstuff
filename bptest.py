import torch
from torch import tensor
import torch.nn as nn
actor = nn.Sequential(
		nn.Linear(4, 4),
		nn.Linear(4, 2),
)

a=actor(tensor([0,0,0.1,0],dtype=torch.float))
(a-tensor([0,0],dtype=torch.float)).abs().mean().backward()
print(a)
for i in actor.parameters():
	print(i.grad)
