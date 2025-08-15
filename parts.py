import torch
from torch import tensor

class Layer:
	def __init__(self, shape: int):
		self.shape = shape
		self.v = torch.zeros(shape)
	def preupdate(self):
		self.prevV = self.v
		self.v = torch.zeros(self.shape)

class OutcomePredictingLayer:
	def __init__(self, shape: int):
		self.shape = shape
		self.w = {}
		self.senderTrace = {}
		self.v = torch.zeros(shape)
	def preupdate(self):
		self.prevV = self.v
		self.v = torch.zeros(self.shape)
	def input(self, sender):
		if not sender in self.w:
			self.w[sender] = torch.rand((sender.shape,self.shape))*0.2+0.4
		self.v += sender.v @ self.w[sender]
	def inputTensor(self, t):
		self.v += t.flatten()
	def updateInhibit(self):
		inhibition = (self.v.mean()+self.v.max())/2.0
		self.v -= inhibition
	def update(self, lr=0.5, achLearningSignal=0.0, daSignal=0.0, outcomeLayer=None):
		self.v += outcomeLayer.v*achLearningSignal
		self.updateInhibit()
		self.v = self.v.clamp_min(0.0)
		for sender in self.w:
			if not sender in self.senderTrace:
				self.senderTrace[sender] = torch.zeros((sender.shape))
			self.w[sender] += daSignal * self.senderTrace[sender][:,None] * (self.v-self.prevV)[None,:] * lr
			self.senderTrace[sender] *= max(1.0-achLearningSignal, 0.0)
			self.senderTrace[sender] += sender.v * achLearningSignal
		return self.v

l1=Layer(4)
lp=OutcomePredictingLayer(4)
for i in range(10):
	print("")
	lp.preupdate(); l1.preupdate()
	l1.v = tensor([1,0,0,0],dtype=torch.float); lp.input(l1)
	lp.update(achLearningSignal=1.0,outcomeLayer=l1)
	print(lp.v)
	print("trace",lp.senderTrace)

	lp.preupdate(); l1.preupdate()
	l1.v = tensor([0,1,1,0],dtype=torch.float); lp.input(l1)
	lp.update(outcomeLayer=l1)
	print(lp.v)
	print("trace",lp.senderTrace)

	lp.preupdate(); l1.preupdate()
	l1.v = tensor([0,0,0,1],dtype=torch.float); lp.input(l1)
	lp.update(achLearningSignal=1.0,daSignal=1.0,outcomeLayer=l1)
	print(lp.v)
	print("trace",lp.senderTrace)

	lp.preupdate(); l1.preupdate()
	l1.v = tensor([0,0,0,0],dtype=torch.float); lp.input(l1)
	lp.update(outcomeLayer=l1)
	print(lp.v)
	print("trace",lp.senderTrace)
