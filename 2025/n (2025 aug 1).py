import torch
from torch import tensor
from time import sleep


GbarE = 1; GbarI = 1; GbarL = 0.2; GbarK = 1; ErevE = 1; ErevL = 0.3; ErevI = 0.1; ErevK = 0.1
VmC = 2.81; ExpSlope = 0.02
Thr = 0.5
GeDt = 1/5; GiDt = 1/7

FFAvgDt = 1/50
FB_weight = 1; FSTau = 6; GiOverallScaling = 1.1; FS0 = 0.1
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
		self.gInhibit = 0
		
	"""
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
		print("fs",self.FSi, FSGi)

		# Slow spiking (SS) SST from FBs only, with facilitation factor SSf
		self.SSi += (self.SSf * FBs - self.SSi) / SSiTau
		self.SSf += FBs * (1 - self.SSf) - self.SSf / SSfTau
		SSGi = GiOverallScaling * SSExtraFactor * self.SSi
		print("ss",self.SSi,self.SSf,SSGi)

		self.gInhibit = FSGi + SSGi
	"""
		

class Layer:
	def __init__(self, size: int):
		self.size = size
		self.paths = []
		self.potential = torch.full((size,), -70.0) # in millivolts
		self.refractoryTimer = torch.zeros(size)
		self.spike = torch.zeros(size)
		self.sentSpike = torch.zeros(size)
		self.pools = Pools()

	def gatherInputs(self):
		pass
	def layerInhibit(self):
		pass
	def update(self, inputs): # todo: inputs IS FOR TEMPORARYLY TESTING
		# Integrate-and-fire model
		self.refractoryTimer -= 1.0
		self.refractoryTimer.clamp_min_(0)
		sodiumChannelInactivation = 1-(self.refractoryTimer/3.5).clamp_max(1) # during refractory, at first, it is 0, but when refractoryTimer gets to 3.5, it gradually goes to 1
		#sodiumChannelActivation = 10.0 * torch.exp((self.potential+70.0)/50.0)
		self.potential += inputs * sodiumChannelInactivation - (self.potential + 70)*0.01
		self.spike = self.potential > -65.0
		self.potential[self.spike] = -70.0
		self.refractoryTimer[self.spike] = 5.0
	def sendOutput(self):
		self.sentSpike = self.spike

layers = []
def addLayer(l):
	layers.append(l)
	return l

import matplotlib.pyplot as plt
import matplotlib
plts = [plt.subplot(2,2,1),plt.subplot(2,2,2),plt.subplot(2,2,3),plt.subplot(2,2,4)]
prog1=[];prog2=[]


l1=addLayer(Layer(5))
#l2=addLayer(Layer(6))
#l3=addLayer(Layer(6))
#l4=addLayer(Layer(5))
#connectAllToAll(l1,l2)
#connectAllToAll(l2,l3)
#l4.paths.append(Path(l3,l4))
for i in range(1,30):
	l1.sentSpike = tensor([1,0,0,0,0])
	for l in layers: l.gatherInputs()
	for l in layers: l.layerInhibit()
	for l in layers: l.update(float((i%5)==0))
	for l in layers: l.sendOutput()
	#print(l1.potential)
	plts[0].imshow([l1.potential], interpolation='nearest', vmin=-70, vmax=-65)
	plts[1].imshow([l1.refractoryTimer], interpolation='nearest', vmin=0, vmax=1)
	prog1.append(tensor(l1.potential)); plts[2].imshow(prog1, interpolation='nearest', vmin=-70, vmax=-65)
	prog2.append(tensor(l1.refractoryTimer)); plts[3].imshow(prog2, interpolation='nearest', vmin=0, vmax=1)
	plt.pause(1)
#print(l2.paths[0].weight.to_dense())

plt.show()
