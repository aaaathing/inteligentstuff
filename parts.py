import torch
from torch import tensor

class Layer:
	def __init__(self, shape: int):
		self.shape = shape
		self.v = torch.zeros(shape)
		self.prevV = torch.zeros(shape)
	def update(self):
		self.prevV = self.v
		self.v = torch.zeros(self.shape)
	def inputTensor(self, t):
		self.v += t.flatten()
	def updateInhibit(self):
		inhibition = (self.v.mean()+self.v.max())/2.0
		self.v -= inhibition

class OutcomePredictingLayer(Layer):
	def __init__(self, shape: int):
		self.shape = shape
		self.v = torch.zeros(shape)
		self.prevV = torch.zeros(shape)
		self.w = {}
		self.senderTrace = {}
		#self.vtrace = torch.zeros(shape)
	def input(self, sender):
		if not sender in self.w:
			self.w[sender] = torch.rand((sender.shape,self.shape))*0.2+0.4
		self.v += sender.prevV @ self.w[sender]
	def update(self, lr=0.5, achLearningSignal=0.0, hasReward=0.0):
		#self.v += outcomeLayer.prevV*achLearningSignal
		self.updateInhibit()
		self.v = self.v.clamp_min(0.0)
		#achLearningSignal = achLearningSignal + max(1.0-achLearningSignal,0.0)*self.v.max().item()
		print("ach",achLearningSignal)
		for sender in self.w:
			if not sender in self.senderTrace:
				self.senderTrace[sender] = torch.zeros((sender.shape))
			if hasReward:
				self.senderTrace[sender] += (sender.prevV - self.senderTrace[sender]) * 0.5
				self.w[sender] += self.senderTrace[sender][:,None] * (self.v-self.prevV)[None,:] * lr
				self.senderTrace[sender].zero_()
			elif achLearningSignal>0.1:
				self.senderTrace[sender] += (sender.prevV - self.senderTrace[sender]) * achLearningSignal
		#self.vtrace *= max(1.0-achLearningSignal, 0.0)
		#self.vtrace += self.v * achLearningSignal
		self.prevV = self.v
		self.v = torch.zeros(self.shape)
		return self.prevV

"""
l1=Layer(4)
lp=OutcomePredictingLayer(4)
for i in range(12):
	print("")
	l1.v = tensor([1,0,0,0],dtype=torch.float); l1.update()
	lp.input(l1)
	lp.update(achLearningSignal=1.0 if i<9 else 0.0)
	print(lp.prevV)
	print("trace",lp.senderTrace)

	l1.v = tensor([0,0,0,0],dtype=torch.float); l1.update()
	lp.input(l1)
	lp.update()
	print(lp.prevV)
	print("trace",lp.senderTrace)

	l1.v = tensor([0,1,0,0],dtype=torch.float); l1.update()
	lp.input(l1)
	lp.update()
	print(lp.prevV)
	print("trace",lp.senderTrace)

	l1.v = tensor([0,0,0,0],dtype=torch.float); l1.update()
	lp.input(l1)
	lp.update()
	print(lp.prevV)
	print("trace",lp.senderTrace)

	l1.v = tensor([0,0,0,1],dtype=torch.float); l1.update()
	lp.input(l1)
	lp.update(achLearningSignal=1.0 if i<9 else 0.0,hasReward=1.0)
	print(lp.prevV)
	print("trace",lp.senderTrace)

	l1.v = tensor([0,0,0,0],dtype=torch.float); l1.update()
	lp.input(l1)
	lp.update()
	print(lp.prevV)
	print("trace",lp.senderTrace)

	lp.senderTrace[l1].zero_()
"""


class DecideLayer(Layer):
	def __init__(self, shape: int):
		self.shape = shape
		self.v = torch.zeros(shape)
		self.prevV = torch.zeros(shape)
		self.w = {}
		self.senderTrace = {}
		self.trace = torch.zeros(shape)
	def input(self, sender, inhibit=False):
		if not sender in self.w:
			self.w[sender] = torch.rand((sender.shape,self.shape))*0.2+0.4
		if inhibit:
			self.v -= sender.prevV @ self.w[sender]
		else:
			self.v += sender.prevV @ self.w[sender]
	def update(self, lr=0.5, hasReward=0.0):
		self.updateInhibit()
		self.v = self.v.clamp_min(0.0)
		updateTrace = self.v.max()
		self.trace += (self.v - self.trace) * updateTrace
		for sender in self.w:
			self.senderTrace[sender] += (sender.prevV - self.senderTrace[sender]) * updateTrace
			if hasReward:
				self.w[sender] += self.senderTrace[sender][:,None] * self.trace[None,:] * lr
				self.senderTrace[sender].zero_()
		self.prevV = self.v
		self.v = torch.zeros(self.shape)
		return self.prevV
