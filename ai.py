import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import absvit


# variables that can be changed
class vars:
	lr = 0.5
	hasReward = 0.0
	alphaCycleProgress = {"minusPhaseEnd":False, "end":False, "plusPhase":False}
	traceDecay = 0.1


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
		self.output = torch.zeros(shape)
		self.prevOutput = torch.zeros(shape)
		self.w = {}
		self.feedbackInhibition = torch.zeros(shape)
		self.constantOutput = None

	def input(self, sender, inhibit=False, bidirectional=True):
		if not sender in self.w:
			self.w[sender] = initWeights((sender.shape,self.shape))
		if inhibit:
			self.inputExcitatory -= sender.output @ self.w[sender]
		else:
			self.inputExcitatory += sender.output @ self.w[sender]
			if bidirectional:
				sender.inputExcitatory += (self.output @ self.w[sender].t())

	def inputTensor(self, t):
		self.inputExcitatory += t.flatten()
	def updateInhibit(self):
		feedforwardInhibition = max((self.inputExcitatory.mean()+self.inputExcitatory.max())/2.0 - 1.0, 0.0) # if input is more than 1, start inhibiting it
		self.feedbackInhibition.lerp_(self.prevOutput.mean(), 0.5)
		self.inputInhibition += feedforwardInhibition + self.feedbackInhibition

	def updateV(self):
		self.prevOutput = self.output
		self.updateInhibit()
		if self.constantOutput is not None:
			self.output.copy_(self.constantOutput)
		else:
			netInput = self.inputExcitatory - self.inputInhibition
			self.v += (netInput - self.v) * 0.5
			self.output = torch.tanh(2.0*self.v.clamp_min(0.0))
		self.inputExcitatory = torch.zeros(self.shape)
		self.inputInhibition = torch.zeros(self.shape)
	def update(self):
		self.updateV()

def phaseLearn(self):
	if vars.alphaCycleProgress["minusPhaseEnd"]:
		self.outputMinusPhase = self.output.clone()
	if vars.alphaCycleProgress["end"]:
		for sender in self.w:
			self.w[sender] += (self.output-self.outputMinusPhase)[None,:] * sender.output[:,None] * 0.1
			self.w[sender].clamp_(0.0,1.0)
def traceRewardLearn(self):
	if vars.alphaCycleProgress["end"]:
		if not hasattr(self, "senderTrace"):
			self.trace = {}
		for sender in self.w:
			if not sender in self.trace:
				self.trace[sender] = torch.zeros((sender.shape,self.shape))
			self.trace[sender] += (sender.output[:,None] * self.output[None,:] - self.trace[sender]) * vars.traceDecay
			if vars.hasReward:
				self.w[sender] += self.trace[sender] * vars.hasReward * vars.lr

class DeeppredLayers:
	" see 2025/09-09-temporal-predict "
	def __init__(self, shape):
		self.s = Layer(shape) # superficial
		self.p = {} # predicting layers / pulvinar
		self.c = Layer(shape) # previous superficial layer activation / CT
		#self.cInput = Layer(shape)
		self.lowerLayers = set()
	def input(self, sender, *args, **kwargs):
		if not sender in self.lowerLayers:
			self.lowerLayers.add(sender)
		self.s.input(sender, *args, **kwargs)
		
	def update(self):
		self.c.update()
		#self.c.input(self.cInput, bidirectional=False)
		for lowerLayer in self.lowerLayers:
			if not lowerLayer in self.p:
				self.p[lowerLayer] = Layer(lowerLayer.shape)

			if vars.alphaCycleProgress["plusPhase"]:
				self.p[lowerLayer].constantOutput = lowerLayer.output
			else:
				self.p[lowerLayer].constantOutput = None
				self.p[lowerLayer].input(self.c, bidirectional=False)

			self.p[lowerLayer].update()
			phaseLearn(self.p[lowerLayer])
			self.s.input(self.p[lowerLayer], bidirectional=False)

		self.s.update()
		phaseLearn(self.s)
		
		if vars.alphaCycleProgress["end"]:
			self.c.constantOutput = self.s.output.clone()

class DecideLayer(Layer):
	def update(self):
		self.updateV()
		phaseLearn(self)

class MotorLayers:
	def __init__(self, shape):
		self.prevLayer = Layer(shape)
		self.layer = Layer(shape)
	def update(self, gate):
		self.layer.input(self.prevLayer, bidirectional=False)
		if vars.alphaCycleProgress["end"]:
			self.prevLayer.constantOutput = self.layer.output.clone()
		if vars.alphaCycleProgress["plusPhase"]:
			self.layer.input(gate)
		self.layer.update()
		phaseLearn(self.layer)
		traceRewardLearn(self.layer)

class PFLayers(DeeppredLayers):
	def __init__(self, shape):
		super().__init__(shape)
		self.mem = torch.zeros(shape)
	def update(self, gate):
		self.s.inputTensor(self.mem)
		super().update()
		if gate.output[0] > 0.5:
			self.mem.copy_(self.s.output)


whereInputLayer = Layer(28*28)
whatInputLayer = Layer(192)

yesLayer = DecideLayer(20)
noLayer = DecideLayer(20)
actionLayer_m1 = DecideLayer(20)
actionLayer_wm1 = DecideLayer(1)

wm1 = PFLayers(20)

m1 = MotorLayers(15)

def updateLayers(whatwhere):
	learningSignal = 0.0
	learningSignal += abs(env.reward)

	#if env.video[0,0,0].item()>0.5: # this is temporary, will be removed later
	#	learningSignal = 1.0

	whatInputLayer.inputTensor(whatwhere[0])
	whereInputLayer.inputTensor(whatwhere[1])
	whatInputLayer.update()
	whereInputLayer.update()

	yesLayer.input(whatInputLayer)
	yesLayer.input(whereInputLayer)
	noLayer.input(whatInputLayer)
	noLayer.input(whereInputLayer)
	actionLayer_m1.input(yesLayer)
	actionLayer_m1.input(noLayer, inhibit=True)
	actionLayer_wm1.input(yesLayer)
	actionLayer_wm1.input(noLayer, inhibit=True)

	yesLayer.update()
	noLayer.update()
	actionLayer_m1.update()
	actionLayer_wm1.update()

	wm1.input(whatInputLayer)
	wm1.input(whereInputLayer)
	wm1.update(actionLayer_wm1)

	m1.layer.input(whatInputLayer)
	m1.layer.input(whereInputLayer)
	m1.update(actionLayer_m1)
	
	axs[1,0].clear()
	axs[1,0].text(0.1,0.1, f"learningSignal: {learningSignal} \nreward: {env.reward}")
	plotAt(0,0, env.video.reshape(224,224,4), "video")
	plotAt(0,1, whatInputLayer.output.view(12,16), "whatInputLayer")
	plotAt(0,2, whereInputLayer.output.view(28,28), "whereInputLayer")
	#axs[2,0].imshow([positiveOutcomePredictor.output], vmin=0,vmax=1); axs[2,0].set_title("positiveOutcomePredictor")
	plotAt(2,1, [yesLayer.output], "yesLayer")
	plotAt(2,2, [actionLayer_m1.output], "actionLayer_m1")
	plotAt(3,0, [m1.layer.output], "m1")
	plt.pause(0.1)

fig, axs = plt.subplots(4, 3)
def plotAt(x,y, v, title):
	axs[x,y].imshow(v, vmin=0,vmax=1)
	axs[x,y].set_title(title)
	


async def ailoop():
	for i in range(100):
		await env.step(m1.layer.output)
		vars.hasReward = env.reward

		video = (tensor(env.video, dtype=torch.float) / 255.0).view(224,224,4)[:,:,0:3].permute(2,0,1)
		whatwhere = absvit.run(video, whatInputLayer.output, whereInputLayer.output)

		for j in range(10):
			vars.alphaCycleProgress = {"minusPhaseEnd": j==9, "end":False, "plusPhase":False}
			updateLayers(whatwhere)

		for j in range(10):
			vars.alphaCycleProgress = {"minusPhaseEnd": False, "end":j==9, "plusPhase":False}
			updateLayers(whatwhere)

		plt.pause(5)



from interact_server import env
import asyncio
asyncio.run(env.run(ailoop))
