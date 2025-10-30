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
	boostActivitySpeed = 0.1
	age = 0


def flattenShape(shape):
	if isinstance(shape, tuple):
		prod = 1
		for s in shape:
			prod *= s
		return prod
	else:
		return shape
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
		if not isinstance(shape, tuple): self.shape = (shape,)
		self.size = flattenShape(shape)
		self.inputExcitatory = torch.zeros(self.size)
		self.inputInhibition = torch.zeros(self.size)
		self.v = torch.zeros(self.size)
		self.output = torch.zeros(self.size)
		self.prevOutput = torch.zeros(self.size)
		self.w = {}
		self.inputs = {}
		self.senderInhibit = {} # which senders inhibit
		self.receivers = set()
		self.feedbackInhibition = torch.zeros(shape)
		self.constantOutput = None

	def input(self, sender, inhibit=False, bidirectional=True, initWeights=initWeightsS):
		if not sender in self.w:
			self.w[sender] = initWeights(sender.size, self.size)
		self.senderInhibit[sender] = inhibit
		self.inputs[sender] = sender.output
		sender.receivers.add(self)

		if inhibit:
			self.inputExcitatory -= sender.output @ self.w[sender]
		else:
			self.inputExcitatory += sender.output @ self.w[sender]
			if bidirectional:
				sender.inputExcitatory += self.output @ self.w[sender].T * 0.2

	def inputv(self, v, key, initWeights=initWeightsS, rescaleTo=None):
		if not key in self.w:
			self.w[key] = initWeights(flattenShape(v.shape), self.size)
		self.senderInhibit[key] = False
		self.inputs[key] = v
		v = v @ self.w[key]
		if rescaleTo is not None:
			v = rescale(v, rescaleTo)
		self.inputExcitatory += v

	def inputTensor(self, t):
		self.inputExcitatory += t.flatten()
	def updateInhibit(self):
		feedforwardInhibition = ((self.inputExcitatory.mean()+self.inputExcitatory.max())/2.0 - 1.0).clamp_min(0.0) # if input is more than 1, start inhibiting it
		feedbackInhibition = self.prevOutput.mean()
		if len(self.shape) > 1:
			feedforwardInhibition = torch.maximum(feedforwardInhibition, ((self.inputExcitatory.view(self.shape).mean(dim=1, keepdim=True) + self.inputExcitatory.view(self.shape).amax(dim=1, keepdim=True))/2.0 - 1.0).clamp_min(0.0))
			feedbackInhibition = torch.maximum(feedbackInhibition, self.prevOutput.view(self.shape).mean(dim=1, keepdim=True))
		self.feedbackInhibition.lerp_(feedbackInhibition, 0.5)
		self.inputInhibition += (feedforwardInhibition + self.feedbackInhibition).expand(self.shape).flatten()

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
		self.inputExcitatory = torch.zeros(self.size)
		self.inputInhibition = torch.zeros(self.size)
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
	if not hasattr(self, "avgOutputActivity"):
		self.avgOutputActivity = torch.full((self.size,), target)
		#self.avgInputActivity = torch.full((self.size,), target)
		self.avgLayerActivity = layerTarget
	self.avgOutputActivity += (self.output - self.avgOutputActivity) * vars.boostActivitySpeed
	#self.avgInputActivity += (self.v - self.avgInputActivity) * vars.boostActivitySpeed
	self.avgLayerActivity = (self.output.mean() - self.avgLayerActivity) * vars.boostActivitySpeed
	if vars.alphaCycleProgress["end"]:
		#std = self.output.std()
		#mean = self.output.mean()
		for sender in self.inputs:
			self.w[sender] += ((target - self.avgOutputActivity)/2.0 + (layerTarget - self.avgLayerActivity))[None,:] * self.w[sender] * vars.boostActivitySpeed
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

class SpikingLayer(Layer):
	def __init__(self, shape: int):
		super().__init__(shape)
		self.refractoryTimer = torch.zeros(self.size)
		self.v.fill_(-70.0)
	def updateV(self):
		self.prevOutput = self.output
		self.updateInhibit()
		netInput = self.inputExcitatory*(0.0-self.v) + self.inputInhibition*(-90-self.v) + 0.2*(-70.0-self.v) + torch.sigmoid((self.v+35.0)/2.0)*(145.0-self.v)
		refractory = self.refractoryTimer > 0.0; notRefractory = self.refractoryTimer <= 0.0
		self.v[notRefractory] += netInput[notRefractory]
		self.v[refractory] += (-70.0 - self.v[refractory]) * 0.6
		self.output = torch.logical_and(self.v > 0.0, notRefractory).float()
		self.refractoryTimer -= 1.0
		self.refractoryTimer[self.v > 0.0] = 3.0
		self.inputExcitatory = torch.zeros(self.size)
		self.inputInhibition = torch.zeros(self.size)

"""
s=SpikingLayer(10)
w=(torch.rand((10,10))>0.8).float()
for i in range(100):
	if i<20: s.inputExcitatory[0:5] += 1.0
	s.v += ((s.v[:,None]-s.v[None,:])*(w+w.T)/2).sum(dim=0) * 0.5
	s.update()
	plt.cla()
	plt.imshow([s.v], vmin=-70,vmax=50)
	plt.pause(1)
"""


class DeeppredLayer(Layer):
	" see 2025/09-09-temporal-predict for original "
	def __init__(self, shape):
		super().__init__(shape)
		self.predLayer = Layer(shape)
		self.predLayerInputs = {}
		
	def update(self):
		super().update()
		phaseLearn(self)

		if vars.alphaCycleProgress["plusPhase"]:
			self.predLayer.constantOutput = self.output
		else:
			self.predLayer.constantOutput = None
			for input in self.predLayerInputs:
				self.predLayer.inputv(self.predLayerInputs[input], input)

		self.predLayer.update()
		phaseLearn(self.predLayer)
		for receiver in self.receivers:
			if isinstance(receiver, DeeppredLayer):
				receiver.input(self.predLayer, bidirectional=False)
				if vars.alphaCycleProgress["end"]:
					self.predLayerInputs[receiver] = receiver.output.clone()

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

class PFLayer(DeeppredLayer):
	def __init__(self, shape):
		super().__init__(shape)
		self.memLayer = torch.zeros(self.size)
	def update(self, gate):
		self.inputTensor(self.memLayer)
		super().update()
		if gate[0] > 0.5:
			self.memLayer.copy_(self.output)


# Awesome idea!
decideShape = (20,36)
class DecideLayer(Layer):
	def __init__(self, shape: int):
		super().__init__(shape)
		self.trace = {}
	def addTrace(self, sender, trace):
		if not sender in self.trace:
			self.trace[sender] = torch.zeros(self.w[sender].shape)
		self.trace[sender] += (trace - self.trace[sender]) * vars.traceDecay
dMtxGo = DecideLayer(decideShape)
dMtxNogo = DecideLayer(decideShape)
#actionLayer = DecideLayer(20+1)
dPatchD1 = DecideLayer(decideShape)
dPatchD2 = DecideLayer(decideShape)
dDecision = None
def updateDecide():
	dInputs = [whatInputLayer,whereInputLayer]
	for i in dInputs:
		dMtxGo.input(i, bidirectional=False)
		dMtxNogo.input(i, bidirectional=False)
		dPatchD1.input(i, bidirectional=False)
		dPatchD2.input(i, bidirectional=False)
	dMtxGo.update()
	dMtxNogo.update()
	dPatchD1.update()
	dPatchD2.update()
	global dDecision
	dDecision = dMtxGo.output.view(decideShape).mean(dim=1, keepdim=True) - dMtxNogo.output.view(decideShape).mean(dim=1, keepdim=True)
	decision = dDecision.expand(decideShape).flatten()
	dDecision = dDecision.flatten()
	d1 = dPatchD1.output.view(decideShape).mean(dim=1, keepdim=True).expand(decideShape).flatten()
	d2 = dPatchD2.output.view(decideShape).mean(dim=1, keepdim=True).expand(decideShape).flatten()
	for sender in dInputs:
		goOffTrace = torch.where(decision<0.1, (d2 - d1) * sender.output[:,None] * dMtxGo.output[None,:] * 0.1, 0.0)
		nogoOffTrace = torch.where(decision<0.1, (d2 - d1) * sender.output[:,None] * dMtxNogo.output[None,:] * 0.1, 0.0)
		dMtxGo.addTrace(sender, decision * ((1.0-d1)+d2) * sender.output[:,None] * dMtxGo.output[None,:] + goOffTrace)
		dMtxNogo.addTrace(sender, decision * ((1.0-d1)+d2) * sender.output[:,None] * dMtxNogo.output[None,:] + nogoOffTrace)
		dPatchD1.addTrace(sender, decision * sender.output[:,None] * dPatchD1.output[None,:])
		dPatchD2.addTrace(sender, decision * sender.output[:,None] * dPatchD2.output[None,:])
		if vars.hasReward:
			dMtxGo.w[sender] += dMtxGo.trace[sender] * vars.hasReward * vars.lr
			dMtxNogo.w[sender] -= dMtxNogo.trace[sender] * vars.hasReward * vars.lr
			dPatchD1.w[sender] += dPatchD1.trace[sender] * vars.hasReward * vars.lr
			dPatchD2.w[sender] -= dPatchD2.trace[sender] * vars.hasReward * vars.lr
		dPatchD1.w[sender].clamp_(0.0,1.0)
		dPatchD2.w[sender].clamp_(0.0,1.0)

	# <ChatGPT>: For the motor thalamus (VA/VL nuclei) specifically: It’s tonically active, but its activity is continuously inhibited by the basal ganglia output nuclei (GPi/SNr).

	#vMtxGo.input(rewardPredPositive.acq)
	#vMtxNogo.input(rewardPredNegative.acq)
	#vPatchD1.input(rewardPredPositive.acq)
	#vPatchD2.input(rewardPredNegative.acq)
	# vs patch recieves from PTp, vs mtx recieves from normal


whereInputLayer = DeeppredLayer(28*28)
whatInputLayer = DeeppredLayer(192)

rewardPredPositive = RewardPredLayers(20)
rewardPredNegative = RewardPredLayers(20)

wm1 = PFLayer(20)

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

	updateDecide()

	#wm1.input(whatInputLayer)
	#wm1.input(whereInputLayer)
	#wm1.update(actionLayer.output[20:21])

	m1.layer.inputv(whatInputLayer.output, whatInputLayer, rescaleTo=0.1) # these inputs should not trigger movements
	m1.layer.inputv(whereInputLayer.output, whereInputLayer, rescaleTo=0.1)
	m1.update(dDecision)


fig, axs = plt.subplots(4, 4)
def plotAt(x,y, v, title):
	if len(axs[x,y].images): axs[x,y].images[0].set_data(v)
	else: axs[x,y].imshow(v, vmin=0,vmax=1)
	axs[x,y].set_title(title)
def plotThem():
	axs[0,3].clear()
	axs[0,3].text(0.1,0.1, f"reward: {vars.hasReward}\n age: {vars.age}")
	plotAt(0,0, env.video.reshape(224,224,4), "video")
	plotAt(0,1, whatInputLayer.output.view(12,16), "whatInputLayer")
	plotAt(0,2, whereInputLayer.output.view(28,28), "whereInputLayer")
	plotAt(1,1, [rewardPredPositive.acq.output], "rewardPredPositive")
	plotAt(1,2, [rewardPredNegative.acq.output], "rewardPredNegative")
	#axs[2,0].imshow([positiveOutcomePredictor.output], vmin=0,vmax=1); axs[2,0].set_title("positiveOutcomePredictor")
	plotAt(2,0, dMtxGo.output.view(dMtxGo.shape), "dMtxGo")
	plotAt(2,1, dMtxNogo.output.view(dMtxNogo.shape), "dMtxNogo")
	plotAt(2,2, dPatchD1.output.view(dPatchD1.shape), "dPatchD1")
	plotAt(2,3, dPatchD2.output.view(dPatchD2.shape), "dPatchD2")
	#plotAt(2,2, [actionLayer.output], "actionLayer")
	plotAt(3,0, [m1.layer.output], "m1")
	plotAt(3,1, [m1.layer.prevInputExcitatory], "m1")
	plotAt(3,2, [m1.layer.prevInputInhibition], "m1")
	plt.pause(1)
	


async def ailoop():
	for i in range(10000):
		await env.step(m1.layer.output)
		vars.hasReward = env.reward

		if vars.age < 20:
			vars.lr = 0.1
			vars.boostActivitySpeed = 0.1
		elif vars.age < 50:
			vars.lr = 0.1
			vars.boostActivitySpeed = 0.01
		else:
			vars.lr = 0.01
			vars.boostActivitySpeed = 0.001

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
		vars.age += 1

		plotThem()



from interact_server import env
import asyncio
asyncio.run(env.run(ailoop))
