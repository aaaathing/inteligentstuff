import torch
from torch import tensor
from time import sleep


class Path:
	def __init__(self, sender, reciever):
		self.sender = sender
		self.reciever = reciever
		# each neuron connects to many neurons
		self.weight = (torch.rand((reciever.size,sender.size)) * 0.1 + 1).to_sparse_coo()


class Layer:
	def __init__(self, size: int):
		self.size = size
		self.paths = []
		self.gExcite = torch.zeros(size)
		self.gInhibit = torch.zeros(size)
		self.potential = torch.zeros(size)
		self.spike = None
		self.sentSpike = torch.zeros(size)
	def gatherInputs(self):
		self.gExcite.zero_()
		for path in self.paths:
			assert path.reciever == self
			self.gExcite += (path.sender.sentSpike * path.weight).to_dense().sum(1)
	def update(self):
		# Vm is self.potential
		GbarE = 1; GbarI = 1; GbarL = 0.2; GbarK = 1; ErevE = 1; ErevL = 0.3; ErevI = 0.1; ErevK = 0.1
		Inet = GbarE * self.gExcite * (ErevE - self.potential) + GbarI * self.gInhibit * (ErevI - self.potential) + GbarL * (ErevL - self.potential) #+ GbarK * Gk * (ErevK - self.potential)
		VmC = 2.81; ExpSlope = 0.02
		Thr = 0.5
		self.potential += (Inet + 0.2 * ExpSlope * torch.exp((self.potential-Thr) / ExpSlope)) / VmC
		ExpThr = 0.9
		self.spike = self.potential > ExpThr
		self.potential[self.spike] = 0
	def sendOutput(self):
		self.sentSpike = self.spike

layers = []
def addLayer(l):
	layers.append(l)
	return l

import matplotlib.pyplot as plt
import matplotlib


l1=addLayer(Layer(5))
l2=addLayer(Layer(6))
l3=addLayer(Layer(6))
l4=addLayer(Layer(5))
l2.paths.append(Path(l1,l2))
l3.paths.append(Path(l2,l3))
l4.paths.append(Path(l3,l4))
for i in range(1,50):
	l1.sentSpike = tensor([1,0,0,0,0])
	for l in layers: l.gatherInputs()
	for l in layers: l.update()
	for l in layers: l.sendOutput()
	print(l4.potential)
	plt.imshow([l3.potential], interpolation='nearest', vmin=0, vmax=1)
	plt.pause(0.1)
#print(l2.paths[0].weight.to_dense())
