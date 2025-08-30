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
yesLayer = parts.DecideLayer(4)
noLayer = parts.DecideLayer(4)
actionLayer = parts.DecideLayer(4)


fig, axs = plt.subplots(1, 2)

for i in range(10):
		learningSignal = 0.0
		learningSignal += game.reward

		if game.video[0,0,0].item()>0.5: # this is temporary, will be removed later
			learningSignal = 1.0

		stateLayer.inputTensor(game.video.flatten())
		stateLayer.update()
		
		positiveOutcomePredictor.input(stateLayer)
		negativeOutcomePredictor.input(stateLayer)

		positiveOutcomePredictor.update(hasReward=game.reward, achLearningSignal=learningSignal)
		negativeOutcomePredictor.update(hasReward=game.reward, achLearningSignal=learningSignal)

		yesLayer.input(positiveOutcomePredictor)
		noLayer.input(negativeOutcomePredictor)
		actionLayer.input(yesLayer)
		actionLayer.input(noLayer, inhibit=True)

		yesLayer.update(hasReward=game.reward)
		noLayer.update(hasReward=game.reward)
		actionLayer.update(hasReward=game.reward)

		game.step(actionLayer.v[0].item())

		axs[0].imshow(actionLayer.v, interpolation='nearest', vmin=0,vmax=1)
		axs[1].clear()
		axs[1].text(0.4,0.4, f"learningSignal: {learningSignal} \nreward: {game.reward}")
		plt.pause(1)
