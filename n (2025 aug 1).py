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
		self.gExciteRaw = torch.zeros(size)
		self.gInhibitRaw = torch.zeros(size)
		self.gExcite = torch.zeros(size) # these reset every time
		self.gInhibit = torch.zeros(size)
		self.potential = torch.full((size), -70)
		self.sodium = torch.zeros(size)
		self.potassium = torch.zeros(size)
		self.sodiumChannelInactivation = torch.zeros(size)
		self.spike = torch.zeros(size)
		self.sentSpike = torch.zeros(size)
		self.pools = Pools()

	"""def decay(self):
		decayOfActivation = 0.2
		self.gExciteSyn *= 1-decayOfActivation
		self.gInhibitRaw *= 1-decayOfActivation
		self.potential *= 1-decayOfActivation
	"""
	def gatherInputs(self):
		# gather spikes init
		#gExciteRaw = torch.zeros(self.size)
		self.gExciteRaw.zero_() # todo: set to avg
		self.gInhibitRaw.zero_() # todo: set to avg

		for path in self.paths:
			assert path.reciever == self
			path.gatherInputs()
		#self.gExciteSyn += gExciteRaw
		self.gExcite = self.gExciteRaw # todo: add external input to gExcite
		# todo: add noise to gExcite
		self.gInhibit = self.gInhibitRaw
	def layerInhibit(self):
		pass
		#self.pools.inhibit(self)
		#self.gInhibit += self.pools.gInhibit
	def update(self):
		# Hodgkin-Huxley model
		# Constants
		Cm = 1.0  # Membrane capacitance in uF/cm^2
		gNa = 120.0  # Sodium channel conductance in mS/cm^2
		gK = 36.0  # Potassium channel conductance in mS/cm^2
		gL = 0.3  # Leak conductance in mS/cm^2
		ENa = 50.0  # Sodium reversal potential in mV
		EK = -77.0  # Potassium reversal potential in mV
		EL = -54.387  # Leak reversal potential in mV
		I_ext = 10.0  # External current in uA/cm^2

		I_Na = gNa*(self.sodiumChannelActivation**3)*self.sodiumChannelInactivation*(self.potential-ENa)
		I_K = gK*(self.potassiumChannelActivation**4)*(self.potential-EK)
		I_L = gL*(self.potential-EL)
		self.potential += (I_ext - I_Na - I_K - I_L) / Cm

		self.sodiumChannelActivation += (0.1*(self.potential+40)/(1-torch.exp(-(self.potential+40)/10)))*(1-self.sodiumChannelActivation) - (4.0*torch.exp(-(self.potential+65)/18))*self.sodiumChannelActivation

		self.sodiumChannelInactivation += (0.07*torch.exp(-(self.potential+65)/20))*(1-self.sodiumChannelInactivation) - (1.0/(1+torch.exp(-(self.potential+35)/10)))*self.sodiumChannelInactivation

		self.potassiumChannelActivation += (0.01*(self.potential+55)/(1-torch.exp(-(self.potential+55)/10)))*(1-self.potassiumChannelActivation) - (0.125*torch.exp(-(self.potential+65)/80))*self.potassiumChannelActivation

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
plts = [plt.subplot(2,2,1),plt.subplot(2,2,2),plt.subplot(2,2,3),plt.subplot(2,2,4)]
prog1=[];prog2=[]


l1=addLayer(Layer(5))
l2=addLayer(Layer(6))
#l3=addLayer(Layer(6))
#l4=addLayer(Layer(5))
connectAllToAll(l1,l2)
#connectAllToAll(l2,l3)
#l4.paths.append(Path(l3,l4))
for i in range(1,30):
	l1.sentSpike = tensor([1,0,0,0,0])
	for l in layers: l.gatherInputs()
	for l in layers: l.layerInhibit()
	for l in layers: l.update()
	for l in layers: l.sendOutput()
	#print(l2.potential,l2.gInhibit[0])
	plts[0].imshow([l2.gExcite], interpolation='nearest', vmin=0, vmax=1)
	plts[1].imshow([l2.gInhibit], interpolation='nearest', vmin=0, vmax=1)
	prog1.append(tensor(l2.gExcite)); plts[2].imshow(prog1, interpolation='nearest', vmin=0, vmax=2)
	prog2.append(tensor(l2.gInhibit)); plts[3].imshow(prog2, interpolation='nearest', vmin=0, vmax=2)
	plt.pause(1)
#print(l2.paths[0].weight.to_dense())

plt.show()
