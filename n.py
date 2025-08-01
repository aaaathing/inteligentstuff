import torch
from torch import tensor


class Path:
	def __init__(self, sender, reciever):
		self.sender = sender
		self.reciever = reciever
		# each neuron connects to many neurons
		self.weight = (torch.rand((reciever.size,sender.size)) * 0.1 + 0.4).to_sparse_coo()


class Layer:
	def __init__(self, size: int):
		self.size = size
		self.paths = []
		self.gExcite = torch.zeros(size)
		self.gInhibit = torch.zeros(size)
		self.potential = torch.zeros(size)
		self.spike = None
		self.sentSpike = torch.zeros(size)
	def gather(self):
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
	def send(self):
		self.sentSpike = self.spike


l1=Layer(5)
l2=Layer(6)
l2.paths.append(Path(l1,l2))
l1.sentSpike = tensor([1,0,0,0,0])
l2.gather();l2.update()
print(l2.gExcite)
print(l2.paths[0].weight.to_dense())
