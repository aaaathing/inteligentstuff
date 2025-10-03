import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import absvit

# © thingmaker (http://github.com/aaaathing)
# if you copy, explain that the code is from here

# variables that can be changed
class vars:
	lr = 0.5
	hasReward = 0.0
	alphaCycleProgress = {"minusPhaseEnd":False, "end":False, "plusPhase":False}
	traceDecay = 0.1


def initWeights(self, sender, howInitWeights="random"):
	match howInitWeights:
		case "weak":
			return torch.full((sender.shape,self.shape), 0.1)
		case "random" | None:
			return torch.rand((sender.shape,self.shape))*0.2+0.4
		case "zero":
			return torch.zeros((sender.shape,self.shape))

class Layer:
	def __init__(self, shape: int):
		self.shape = shape
		self.inputExcitatory = torch.zeros(shape)
		self.inputInhibition = torch.zeros(shape)
		#self.inputCount = 0.0
		self.v = torch.zeros(shape)
		self.output = torch.zeros(shape)
		self.prevOutput = torch.zeros(shape)
		self.w = {}
		self.senderInhibit = {} # which senders inhibit
		self.feedbackInhibition = torch.zeros(shape)
		self.constantOutput = None

	def rescale(self, thisInput, rescaleTo):
		m = thisInput.mean()
		if m>rescaleTo: thisInput /= m/rescaleTo
		return thisInput

	def input(self, sender, inhibit=False, bidirectional=True, rescaleTo=1.0, howInitWeights=None):
		if not sender in self.w:
			self.w[sender] = initWeights(self, sender, howInitWeights=howInitWeights)
		self.senderInhibit[sender] = inhibit

		if inhibit:
			self.inputExcitatory -= self.rescale(sender.output @ self.w[sender], rescaleTo=rescaleTo)
		else:
			self.inputExcitatory += self.rescale(sender.output @ self.w[sender], rescaleTo=rescaleTo)
			#self.inputCount += rescaleTo
			if bidirectional:
				sender.inputExcitatory += self.rescale(self.output @ self.w[sender].t(), rescaleTo=rescaleTo*0.2)

	def inputTensor(self, t):
		self.inputExcitatory += t.flatten()
	def updateInhibit(self):
		feedforwardInhibition = max((self.inputExcitatory.mean()+self.inputExcitatory.max())/2.0 - 1.0, 0.0) # if input is more than 1, start inhibiting it
		self.feedbackInhibition.lerp_(self.prevOutput.mean(), 0.5)
		self.inputInhibition += feedforwardInhibition + self.feedbackInhibition

	def updateV(self):
		self.prevOutput = self.output
		#if self.inputCount != 0.0: self.inputExcitatory /= self.inputCount
		self.updateInhibit()
		if self.constantOutput is not None:
			self.output.copy_(self.constantOutput)
		else:
			netInput = self.inputExcitatory - self.inputInhibition
			self.v += (netInput - self.v) * 0.5
			self.output = torch.tanh(2.0*self.v.clamp_min(0.0))
		self.prevInputExcitatory = self.inputExcitatory
		self.prevInputInhibition = self.inputInhibition
		self.inputExcitatory = torch.zeros(self.shape)
		self.inputInhibition = torch.zeros(self.shape)
		self.inputCount = 0.0
	def update(self):
		self.updateV()

def phaseLearn(self):
	if vars.alphaCycleProgress["minusPhaseEnd"]:
		self.outputMinusPhase = self.output.clone()
	if vars.alphaCycleProgress["end"]:
		for sender in self.w:
			senderOutput = sender.output
			if self.senderInhibit[sender]: senderOutput = -senderOutput
			self.w[sender] += (self.output-self.outputMinusPhase)[None,:] * senderOutput[:,None] * vars.lr
			self.w[sender].clamp_(0.0,1.0)
def traceRewardLearn(self):
	if vars.alphaCycleProgress["end"]:
		if not hasattr(self, "senderTrace"):
			self.trace = {}
		for sender in self.w:
			if not sender in self.trace:
				self.trace[sender] = torch.zeros((sender.shape,self.shape))
			senderOutput = sender.output
			if self.senderInhibit[sender]: senderOutput = -senderOutput
			self.trace[sender] += (senderOutput[:,None] * self.output[None,:] - self.trace[sender]) * vars.traceDecay
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

class RewardPredLayers(): #todo: finish
	def __init__(self, shape):
		self.acq = Layer(shape)
		self.acq.senderTrace = {}
		#self.ext = Layer(shape)
	def update(self, daSign=1.0):
		self.acq.update()
		#self.ext.update()
		if vars.hasReward:
			self.acq.inputExcitatory += max(vars.hasReward*daSign,0.0)
			#self.ext.inputExcitatory += -max(vars.hasReward*daSign,0.0)
		if vars.alphaCycleProgress["end"]:
			self.acq.prevOutput = self.acq.output.clone()
			#self.ext.prevOutput = self.ext.output.clone()
		for sender in self.acq.w:
			if not sender in self.acq.senderTrace:
				self.acq.senderTrace[sender] = torch.zeros((sender.shape,))
			senderOutput = sender.output
			if self.acq.senderInhibit[sender]: senderOutput = -senderOutput
			self.acq.senderTrace[sender] += (senderOutput - self.acq.senderTrace[sender]) * vars.traceDecay
			if vars.hasReward:
				self.acq.w[sender] += abs(vars.hasReward) * self.acq.senderTrace[sender][:,None] * (self.acq.output * (self.acq.prevOutput-self.acq.output))[None,:] * vars.lr
				self.acq.senderTrace[sender] *= min(1.0 - abs(vars.hasReward), 0.0)
			self.acq.w[sender].clamp_(0.0,1.0)

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
			self.prevLayer.constantOutput = self.layer.output>0.5
		if vars.alphaCycleProgress["plusPhase"]:
			self.layer.input(gate, bidirectional=False)
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

rewardPredPositive = RewardPredLayers(20)
rewardPredNegative = RewardPredLayers(20)

yesLayer = DecideLayer(20)
noLayer = DecideLayer(20)
actionLayer_m1 = DecideLayer(20)
actionLayer_wm1 = DecideLayer(1)

wm1 = PFLayers(20)

m1 = MotorLayers(15)

def updateLayers(whatwhere):
	learningSignal = 0.0
	learningSignal += abs(env.reward)

	whatInputLayer.inputTensor(whatwhere[0])
	whereInputLayer.inputTensor(whatwhere[1])
	whatInputLayer.update()
	whereInputLayer.update()

	rewardPredPositive.acq.inputExcitatory[0:5] += 1.0 # bias to expect reward

	rewardPredPositive.acq.input(whatInputLayer, howInitWeights="zero")
	rewardPredPositive.acq.input(whereInputLayer, howInitWeights="zero")
	rewardPredPositive.update()
	rewardPredNegative.acq.input(whatInputLayer, howInitWeights="zero")
	rewardPredNegative.acq.input(whereInputLayer, howInitWeights="zero")
	rewardPredNegative.update()

	yesLayer.input(rewardPredPositive.acq)
	noLayer.input(rewardPredNegative.acq)
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

	m1.layer.input(whatInputLayer, rescaleTo=0.1) # these inputs should not trigger movements
	m1.layer.input(whereInputLayer, rescaleTo=0.1)
	m1.update(actionLayer_m1)


fig, axs = plt.subplots(4, 3)
def plotAt(x,y, v, title):
	if len(axs[x,y].images):
		axs[x,y].images[0].set_data(v)
	else:
		axs[x,y].imshow(v, vmin=0,vmax=1)
	axs[x,y].set_title(title)

def plotThem():
	axs[1,0].clear()
	axs[1,0].text(0.1,0.1, f"reward: {env.reward}")
	plotAt(0,0, env.video.reshape(224,224,4), "video")
	plotAt(0,1, whatInputLayer.output.view(12,16), "whatInputLayer")
	plotAt(0,2, whereInputLayer.output.view(28,28), "whereInputLayer")
	plotAt(1,1, [rewardPredPositive.acq.output], "rewardPredPositive")
	plotAt(1,2, [rewardPredNegative.acq.output], "rewardPredNegative")
	#axs[2,0].imshow([positiveOutcomePredictor.output], vmin=0,vmax=1); axs[2,0].set_title("positiveOutcomePredictor")
	plotAt(2,0, [yesLayer.output], "yesLayer")
	plotAt(2,1, [noLayer.output], "noLayer")
	plotAt(2,2, [actionLayer_m1.output], "actionLayer_m1")
	plotAt(3,0, [m1.layer.output], "m1")
	plotAt(3,1, [m1.layer.prevInputExcitatory], "m1")
	plotAt(3,2, [m1.layer.prevInputInhibition], "m1")
	plt.pause(1)
	


async def ailoop():
	for i in range(10000):
		await env.step(m1.layer.output)
		vars.hasReward = env.reward

		video = (tensor(env.video, dtype=torch.float) / 255.0).view(224,224,4)
		whatwhere = absvit.run(video[:,:,0:3].permute(2,0,1), whatInputLayer.output, whereInputLayer.output)
		#whatwhere[1] *= video[:,:,3].view(784,1) # ignore transparent parts

		for j in range(10):
			vars.alphaCycleProgress = {"minusPhaseEnd": j==9, "end":False, "plusPhase":False}
			updateLayers(whatwhere)

		plotThem()

		for j in range(10):
			vars.alphaCycleProgress = {"minusPhaseEnd": False, "end":j==9, "plusPhase":True}
			updateLayers(whatwhere)

		plotThem()



from interact_server import env
import asyncio
asyncio.run(env.run(ailoop))
