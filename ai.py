import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import parts
import matplotlib.pyplot as plt

import absvit

whereInputLayer = parts.Layer(28*28)
whatInputLayer = parts.Layer(192)

positiveOutcomePredictor = parts.OutcomePredictingLayer(4)
negativeOutcomePredictor = parts.OutcomePredictingLayer(4)

gatingWindowLayer = parts.GatingWindowLayer()

yesLayer = parts.DecideLayer(4)
noLayer = parts.DecideLayer(4)
actionLayer = parts.DecideLayer(4)

fig, axs = plt.subplots(3, 3)

def updateLayers(alphaCycleProgress, whatwhere):
	parts.alphaCycleProgress = alphaCycleProgress
	learningSignal = 0.0
	learningSignal += env.reward

	#if env.video[0,0,0].item()>0.5: # this is temporary, will be removed later
	#	learningSignal = 1.0
	
	parts.lr = 0.5

	whatInputLayer.inputTensor(whatwhere[0])
	whereInputLayer.inputTensor(whatwhere[1])
	whatInputLayer.update()
	whereInputLayer.update()
	
	positiveOutcomePredictor.input(whatInputLayer)
	positiveOutcomePredictor.input(whereInputLayer)
	negativeOutcomePredictor.input(whatInputLayer)
	negativeOutcomePredictor.input(whereInputLayer)

	positiveOutcomePredictor.update(hasReward=env.reward, achLearningSignal=learningSignal)
	negativeOutcomePredictor.update(hasReward=env.reward, achLearningSignal=learningSignal)

	#print(positiveOutcomePredictor.prevV,positiveOutcomePredictor.senderTrace)

	gatingWindowLayer.update([positiveOutcomePredictor,negativeOutcomePredictor])
	gatingInhibit = max(1.0-gatingWindowLayer.totalReleased, 0.0)*10.0

	yesLayer.input(positiveOutcomePredictor)
	yesLayer.inputInhibition += gatingInhibit
	noLayer.input(negativeOutcomePredictor)
	noLayer.inputInhibition += gatingInhibit
	actionLayer.input(yesLayer)
	actionLayer.inputInhibition += gatingInhibit
	#actionLayer.input(noLayer, inhibit=True)

	yesLayer.update(hasReward=env.reward)
	noLayer.update(hasReward=env.reward)
	actionLayer.update(hasReward=env.reward)

	if alphaCycleProgress["end"]:
		if actionLayer.output.max() > 0.5:
			gatingWindowLayer.reset()
	
	axs[1,0].clear()
	axs[1,0].text(0.1,0.1, f"learningSignal: {learningSignal} \nreward: {env.reward} \ntotalReleased: {gatingWindowLayer.totalReleased}")
	axs[0,0].imshow(env.video.reshape(224,224,4)); axs[0,0].set_title("video")
	axs[0,1].imshow(whatInputLayer.output.view(12,16), vmin=0,vmax=1); axs[0,1].set_title("whatInputLayer")
	axs[0,2].imshow(whereInputLayer.output.view(28,28), vmin=0,vmax=1); axs[0,2].set_title("whereInputLayer")
	axs[2,0].imshow([positiveOutcomePredictor.output], vmin=0,vmax=1); axs[2,0].set_title("positiveOutcomePredictor")
	axs[2,1].imshow([yesLayer.output], vmin=0,vmax=1); axs[2,1].set_title("yesLayer")
	axs[2,2].imshow([actionLayer.output], vmin=0,vmax=1); axs[2,2].set_title("actionLayer")
	plt.pause(0.1)


async def ailoop():
	for i in range(100):
		await env.step(actionLayer.output)

		video = (tensor(env.video, dtype=torch.float) / 255.0).view(224,224,4)[:,:,0:3].permute(2,0,1)
		whatwhere = absvit.run(video, whatInputLayer.output, whereInputLayer.output)

		for j in range(10):
			updateLayers({"end":j==9}, whatwhere)

		plt.pause(5)



from interact_server import env
import asyncio
asyncio.run(env.run(ailoop))
