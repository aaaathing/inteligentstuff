import sys
sys.path.append("..")
import torch
from torch import tensor
import parts
import matplotlib.pyplot as plt

layer1 = parts.Layer(10)
layer2 = parts.Layer(10)
for i in range(100):
	if (i%10)==0:
		layer1.constantOutput = (layer2.output>layer2.output.mean()).type(torch.float)
	layer2.inputTensor(torch.ones(10,dtype=torch.float))
	layer2.input(layer1)
	layer2.update()
	layer1.update()
	plt.imshow([layer1.output,layer2.output], vmin=0,vmax=1)
	plt.pause(0.1)
plt.pause(10)
