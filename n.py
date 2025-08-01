import torch
from torch import tensor
from time import sleep


GbarE = 1; GbarI = 1; GbarL = 0.2; GbarK = 1; ErevE = 1; ErevL = 0.3; ErevI = 0.1; ErevK = 0.1
VmC = 2.81; ExpSlope = 0.02
Thr = 0.5
GeDt = 1/5; GiDt = 1/7
FFAvgDt = 1/50
FB_weight = 1; FSTau = 1/6; GiOverallScaling = 1.1; FS0 = 0.1
SSiTau = 50; SSfTau = 20; SSExtraFactor = 30

class Path:
	def __init__(self, sender, reciever):
		self.sender = sender
		self.reciever = reciever
		self.gExciteRaw = torch.zeros(reciever.size)
		# each neuron connects to many neurons
		self.weight = None # shape (reciever.size, sender.size)
	def gatherInputs(self):
		gExciteRaw = (self.sender.sentSpike * self.weight).to_dense().sum(1)
		self.gExciteRaw = self.gExciteRaw*(1-GeDt) + gExciteRaw
		self.reciever.gExciteRaw += self.gExciteRaw

paths = []
def connectAllToAll(sender, reciever):
	p = Path(sender,reciever)
	p.weight = (torch.rand((reciever.size,sender.size)) * 0.1 + 0.4).to_sparse_coo()
	reciever.paths.append(p)
	paths.append(p)
	return p


class Pools:
	def __init__(self):
		self.FFsRaw = 0
		self.FBsRaw = 0
		self.FFAverage = 0
		self.FSi = 0
		self.SSi = 0
		self.SSf = 0
	def inhibit(self, layer):
		self.FFsRaw = layer.gExciteRaw.sum(0)
		self.FBsRaw = layer.spike.sum(0)

		FFs = self.FFsRaw/layer.size
		FBs = self.FBsRaw/layer.size

		self.FFAverage += FFAvgDt * (FFs - self.FFAverage)

		# Fast spiking (FS) PV from FFs and FBs
		self.FSi = FFs + FB_weight * FBs - self.FSi / FSTau
		FSGi = GiOverallScaling * max(self.FSi - FS0, 0)

		# Slow spiking (SS) SST from FBs only, with facilitation factor SSf
		self.SSi += (self.SSf * FBs - self.SSi) / SSiTau
		self.SSf += FBs * (1 - self.SSf) - self.SSf / SSfTau
		SSGi = GiOverallScaling * SSExtraFactor * self.SSi

		layer.gInhibit = FSGi + SSGi
		

class Layer:
	def __init__(self, size: int):
		self.size = size
		self.paths = []
		self.gExciteRaw = torch.zeros(size)
		self.gInhibitSyn = torch.zeros(size)
		self.gExcite = torch.zeros(size) # these reset every time
		self.gInhibit = torch.zeros(size)
		self.potential = torch.zeros(size)
		self.spike = None
		self.sentSpike = torch.zeros(size)
	"""def decay(self):
		decayOfActivation = 0.2
		self.gExciteSyn *= 1-decayOfActivation
		self.gInhibitSyn *= 1-decayOfActivation
		self.potential *= 1-decayOfActivation"""
	def gatherInputs(self):
		# gather spikes init
		#gExciteRaw = torch.zeros(self.size)
		self.gExciteRaw.zero_() # todo: set to avg
		self.gInhibitSyn.zero_() # todo: set to avg

		for path in self.paths:
			assert path.reciever == self
			path.gatherInputs()
		#self.gExciteSyn += gExciteRaw
		self.gExcite = self.gExciteRaw # todo: add external input to gExcite
		# todo: add noise to gExcite
	def update(self):
		# Vm is self.potential
		Inet = GbarE * self.gExcite * (ErevE - self.potential) + GbarI * self.gInhibit * (ErevI - self.potential) + GbarL * (ErevL - self.potential) #+ GbarK * Gk * (ErevK - self.potential)
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
#l4=addLayer(Layer(5))
connectAllToAll(l1,l2)
connectAllToAll(l2,l3)
#l4.paths.append(Path(l3,l4))
for i in range(1,50):
	l1.sentSpike = tensor([1,0,0,0,0])
	for l in layers: l.gatherInputs()
	for l in layers: l.update()
	for l in layers: l.sendOutput()
	print(l3.spike)
	plt.imshow([l3.potential], interpolation='nearest', vmin=0, vmax=1)
	plt.pause(0.1)
#print(l2.paths[0].weight.to_dense())
