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
	boostActivitySpeed = 0.7


def initWeightsKaiming(senderShape, receiverShape):
	return torch.nn.init.kaiming_normal_(torch.empty((senderShape, receiverShape)), nonlinearity='relu').T
def initWeightsS(senderShape, receiverShape):
	chance = min(1.0 / senderShape * 10.0, 1.5)
	w = torch.rand((senderShape, receiverShape))
	w = (w - (1.0-chance)).clamp_min(0.0) / chance
	w /= w.sum(dim=0, keepdim=True) + 0.0001
	return w
def initWeightsZero(senderShape, receiverShape):
	return torch.zeros((senderShape, receiverShape))

def rescale(thisInput, rescaleTo):
	m = thisInput.mean()
	if m>rescaleTo: thisInput = thisInput/(m/rescaleTo)
	return thisInput

class Layer:
	def __init__(self, shape: int):
		self.shape = shape
		self.inputExcitatory = torch.zeros(shape)
		self.inputInhibition = torch.zeros(shape)
		self.v = torch.zeros(shape)
		self.output = torch.zeros(shape)
		self.prevOutput = torch.zeros(shape)
		self.w = {}
		self.inputs = {}
		self.senderInhibit = {} # which senders inhibit
		self.feedbackInhibition = torch.zeros(shape)
		self.constantOutput = None

	def input(self, sender, inhibit=False, bidirectional=True, initWeights=initWeightsS):
		if not sender in self.w:
			self.w[sender] = initWeights(sender.shape, self.shape)
		self.senderInhibit[sender] = inhibit
		self.inputs[sender] = sender.output

		if inhibit:
			self.inputExcitatory -= sender.output @ self.w[sender]
		else:
			self.inputExcitatory += sender.output @ self.w[sender]
			if bidirectional:
				sender.inputExcitatory += self.output @ self.w[sender].T * 0.2

	def inputv(self, v, key, initWeights=initWeightsS, rescaleTo=None):
		if not key in self.w:
			self.w[key] = initWeights(v.shape[0], self.shape)
		self.senderInhibit[key] = False
		self.inputs[key] = v
		v = v @ self.w[key]
		if rescaleTo is not None:
			v = rescale(v, rescaleTo)
		self.inputExcitatory += v

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
		self.prevInputExcitatory = self.inputExcitatory
		self.prevInputInhibition = self.inputInhibition
		self.inputExcitatory = torch.zeros(self.shape)
		self.inputInhibition = torch.zeros(self.shape)
		self.inputCount = 0.0
	def update(self):
		self.updateV()
		boostActivity(self)

def phaseLearn(self):
	if vars.alphaCycleProgress["minusPhaseEnd"]:
		self.outputMinusPhase = self.output.clone()
	if vars.alphaCycleProgress["end"]:
		for sender in self.inputs:
			senderOutput = self.inputs[sender]
			if self.senderInhibit[sender]: senderOutput = -senderOutput
			self.w[sender] += (self.output-self.outputMinusPhase)[None,:] * senderOutput[:,None] * vars.lr
			self.w[sender].clamp_(0.0,1.0)

def boostActivity(self):
	target = 0.5
	layerTarget = 0.1
	if not hasattr(self, "avgActivity"):
		self.avgActivity = torch.full((self.shape,), target)
		self.avgLayerActivity = target
	self.avgActivity += (self.output - self.avgActivity) * vars.boostActivitySpeed
	self.avgLayerActivity = (self.output.mean() - self.avgLayerActivity) * vars.boostActivitySpeed
	if vars.alphaCycleProgress["end"]:
		#std = self.output.std()
		#mean = self.output.mean()
		for sender in self.inputs:
			self.w[sender] += ((target - self.avgActivity) + (layerTarget - self.avgLayerActivity))[None,:] * self.w[sender] * vars.boostActivitySpeed
			#self.w[sender] += ((self.output-mean)*(1.0-std))[None,:] * self.w[sender] * vars.boostActivitySpeed

def traceRewardLearn(self):
	if vars.alphaCycleProgress["end"]:
		if not hasattr(self, "senderTrace"):
			self.trace = {}
		for sender in self.inputs:
			if not sender in self.trace:
				self.trace[sender] = torch.zeros(self.w[sender].shape)
			senderOutput = self.inputs[sender]
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
		for sender in self.acq.inputs:
			if not sender in self.acq.senderTrace:
				self.acq.senderTrace[sender] = torch.zeros(self.acq.inputs[sender].shape)
			senderOutput = self.acq.inputs[sender]
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
		boostActivity(self)

class MotorLayers:
	def __init__(self, shape):
		self.prevLayer = Layer(shape)
		self.layer = Layer(shape)
	def update(self, gate):
		self.layer.input(self.prevLayer, bidirectional=False)
		if vars.alphaCycleProgress["end"]:
			self.prevLayer.constantOutput = self.layer.output>0.5
		self.layer.inputv(gate, key="gate")
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
		if gate[0] > 0.5:
			self.mem.copy_(self.s.output)


whereInputLayer = Layer(28*28)
whatInputLayer = Layer(192)

rewardPredPositive = RewardPredLayers(20)
rewardPredNegative = RewardPredLayers(20)

yesLayer = DecideLayer(20)
noLayer = DecideLayer(20)
actionLayer = DecideLayer(20+1)

wm1 = PFLayers(20)

m1 = MotorLayers(15)

def updateLayers(whatwhere):
	learningSignal = 0.0
	learningSignal += abs(env.reward)

	whatInputLayer.inputTensor(whatwhere[0])
	whereInputLayer.inputTensor(whatwhere[1])
	whatInputLayer.update()
	whereInputLayer.update()
	
	rewardPredPositive.acq.inputExcitatory[0:5] += 0.5 # bias to expect reward

	rewardPredPositive.acq.input(whatInputLayer, initWeights=initWeightsZero)
	rewardPredPositive.acq.input(whereInputLayer, initWeights=initWeightsZero)
	rewardPredPositive.update()
	rewardPredNegative.acq.input(whatInputLayer, initWeights=initWeightsZero)
	rewardPredNegative.acq.input(whereInputLayer, initWeights=initWeightsZero)
	rewardPredNegative.update()

	yesLayer.input(rewardPredPositive.acq)
	noLayer.input(rewardPredNegative.acq)
	actionLayer.input(yesLayer)
	actionLayer.input(noLayer, inhibit=True)

	yesLayer.update()
	noLayer.update()
	actionLayer.update()

	wm1.input(whatInputLayer)
	wm1.input(whereInputLayer)
	wm1.update(actionLayer.output[20:21])

	m1.layer.inputv(whatInputLayer.output, whatInputLayer, rescaleTo=0.1) # these inputs should not trigger movements
	m1.layer.inputv(whereInputLayer.output, whereInputLayer, rescaleTo=0.1)
	m1.update(actionLayer.output[0:20])


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
	plotAt(2,2, [actionLayer.output], "actionLayer")
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
