import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import parts
import matplotlib.pyplot as plt

import smallgame
game = smallgame.clickergame()

stateLayer = parts.Layer(18)

positiveOutcomePredictor = parts.OutcomePredictingLayer(4)
negativeOutcomePredictor = parts.OutcomePredictingLayer(4)

gatingWindowLayer = parts.GatingWindowLayer()

yesLayer = parts.DecideLayer(4)
noLayer = parts.DecideLayer(4)
actionLayer = parts.DecideLayer(4)

fig, axs = plt.subplots(2, 3)

for i in range(100):
		learningSignal = 0.0
		learningSignal += game.reward

		if game.video[0,0,0].item()>0.5: # this is temporary, will be removed later
			learningSignal = 1.0
		
		lr = 0.5 if i<30 else 0

		stateLayer.inputTensor(tensor(game.video.flatten(), dtype=float))
		stateLayer.update()
		
		positiveOutcomePredictor.input(stateLayer)
		negativeOutcomePredictor.input(stateLayer)

		positiveOutcomePredictor.update(hasReward=game.reward, achLearningSignal=learningSignal)
		negativeOutcomePredictor.update(hasReward=game.reward, achLearningSignal=learningSignal)

		print(positiveOutcomePredictor.prevV,positiveOutcomePredictor.senderTrace)

		gatingWindowLayer.update([positiveOutcomePredictor,negativeOutcomePredictor])
		gatingInhibit = max(1.0-gatingWindowLayer.totalReleased, 0.0)*10.0

		yesLayer.input(positiveOutcomePredictor)
		yesLayer.inhibition += gatingInhibit
		noLayer.input(negativeOutcomePredictor)
		noLayer.inhibition += gatingInhibit
		actionLayer.input(yesLayer)
		actionLayer.inhibition += gatingInhibit
		#actionLayer.input(noLayer, inhibit=True)

		yesLayer.update(hasReward=game.reward)
		noLayer.update(hasReward=game.reward)
		actionLayer.update(hasReward=game.reward)

		if actionLayer.v.max() > 0.5:
			gatingWindowLayer.reset()

		game.step(actionLayer.v[0].item())

		axs[0,0].clear()
		axs[0,0].text(0.1,0.1, f"learningSignal: {learningSignal} \nreward: {game.reward} \ntotalReleased: {gatingWindowLayer.totalReleased}")
		axs[1,0].imshow([positiveOutcomePredictor.prevV], vmin=0,vmax=1); axs[1,0].set_title("positiveOutcomePredictor")
		axs[1,1].imshow([yesLayer.prevV], vmin=0,vmax=1); axs[1,1].set_title("yesLayer")
		axs[1,2].imshow([actionLayer.prevV], vmin=0,vmax=1); axs[1,2].set_title("actionLayer")
		plt.pause(1)
