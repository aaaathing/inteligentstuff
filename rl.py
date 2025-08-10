import torch
from torch import tensor
import matplotlib.pyplot as plt

def flattenShape(shape):
	res = 1
	for x in shape:
		res *= x
	return res

layerid = 0
class nlayer:
	def __init__(self, shape):
		self.shape = shape
		global layerid
		self.name = "layer"+str(layerid)
		layerid+=1
		self.w = {}
		self.trace = {}
		self.nextV = torch.zeros(flattenShape(shape))
	def input(self, otherLayer):
		if not otherLayer.name in self.w:
			self.w[otherLayer.name] = torch.rand((flattenShape(otherLayer.shape),flattenShape(self.shape)))
		self.nextV += otherLayer.v @ self.w[otherLayer.name]
	def inputTensor(self, t):
		self.nextV += t.flatten()
	def update(self):
		self.v = self.nextV
		self.nextV = torch.zeros(flattenShape(self.shape))
		return self.v

position = [0,0]

layer1 = nlayer((10,10))
layer2 = nlayer((4,))

for i in range(1):
	input = torch.zeros((10,10))
	input[round(position[0]),round(position[1])] = 1.
	layer1.inputTensor(input)
	layer1.update()
	layer2.input(layer1)
	out = layer2.update()

	position[0] += float(out[0]-out[1])
	position[1] += float(out[2]-out[3])

	reward = position[0]>10 and position[0]<20 and position[1]>-10 and position[1]<10

	print(out)
	plt.imshow(input, interpolation='nearest', vmin=0,vmax=1)
	plt.pause(10)