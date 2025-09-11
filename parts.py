import torch
from torch import tensor

def initWeights(shape):
	#match initWeights:
	#	case "zero":
	#		return torch.rand()*0.1
	#	case "random":
			return torch.rand(shape)*0.2+0.4
		
class Layer:
	def __init__(self, shape: int):
		self.shape = shape
		self.inputExcitatory = torch.zeros(shape)
		self.inputInhibition = torch.zeros(shape)
		self.v = torch.zeros(shape)
		self.prevV = torch.zeros(shape)
		self.w = {}
		self.feedbackInhibition = torch.zeros(shape)
		self.constantV = None

	def input(self, sender, inhibit=False, bidirectional=True):
		if not sender in self.w:
			self.w[sender] = initWeights((sender.shape,self.shape))
		if inhibit:
			self.inputExcitatory -= sender.v @ self.w[sender]
		else:
			self.inputExcitatory += sender.v @ self.w[sender]
			if bidirectional:
				sender.inputExcitatory += (self.v @ self.w[sender].t())

	def inputTensor(self, t):
		self.inputExcitatory += t.flatten()
	def updateInhibit(self):
		# self.inhibition.lerp_((self.inputAmount.mean()+self.inputAmount.max())/2.0, 0.5)
		feedforwardInhibition = max((self.inputExcitatory.mean()+self.inputExcitatory.max())/2.0 - 1.0, 0.0) # if input is more than 1, start inhibiting it
		self.feedbackInhibition.lerp_(self.prevV.mean(), 0.5)
		self.inputInhibition += feedforwardInhibition + self.feedbackInhibition

	def updateV(self):
		self.prevV = self.v
		self.updateInhibit()
		if self.constantV is not None:
			self.v.copy_(self.constantV)
		else:
			netInput = self.inputExcitatory - self.inputInhibition
			self.v = netInput # += (netInput - self.v) * 0.5
			self.v = torch.tanh(self.v.clamp_min(0.0))
		self.inputExcitatory = torch.zeros(self.shape)
		self.inputInhibition = torch.zeros(self.shape)
	def update(self):
		self.updateV()

class OutcomePredictingLayer(Layer):
	def __init__(self, shape: int):
		super().__init__(shape)
		self.senderTrace = {}
	def update(self, lr=0.5, achLearningSignal=0.0, hasReward=0.0):
		self.updateV()
		print("ach",achLearningSignal)
		for sender in self.w:
			if not sender in self.senderTrace:
				self.senderTrace[sender] = torch.zeros((sender.shape))
			if hasReward:
				self.senderTrace[sender] += (sender.v - self.senderTrace[sender]) * 0.5
				self.w[sender] += self.senderTrace[sender][:,None] * (self.v-self.prevV)[None,:] * lr
				self.senderTrace[sender].zero_()
			elif achLearningSignal>0.1:
				self.senderTrace[sender] += (sender.v - self.senderTrace[sender]) * achLearningSignal

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
		super().__init__(shape)
		self.senderTrace = {}
		self.trace = torch.zeros(shape)
	def update(self, lr=0.5, hasReward=0.0):
		self.updateV()
		updateTrace = self.v.max()
		self.trace += (self.v - self.trace) * updateTrace
		for sender in self.w:
			if not sender in self.senderTrace:
				self.senderTrace[sender] = torch.zeros((sender.shape))
			self.senderTrace[sender] += (sender.v - self.senderTrace[sender]) * updateTrace
			if hasReward:
				self.w[sender] += self.senderTrace[sender][:,None] * self.trace[None,:] * lr
				self.senderTrace[sender].zero_()

class GatingWindowLayer:
	def __init__(self):
		self.store = {}
		self.released = {}
	def update(self, inputs):
		self.totalReleased = 0
		for i in inputs:
			if not i in self.store:
				self.store[i] = torch.ones(i.shape)
				self.released[i] = torch.zeros(i.shape)
			self.store[i] += 0.05
			self.released[i][i.v > 0.5] = torch.max(self.store[i][i.v > 0.5],self.released[i][i.v > 0.5])
			self.store[i][i.v > 0.5] = 0
			self.totalReleased += self.released[i].sum()
	def reset(self):
		for i in self.released:
			self.released[i].zero_()
