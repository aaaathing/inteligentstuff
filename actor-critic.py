import torch
from torch import tensor
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
actor = nn.Sequential(
		nn.Linear(4, 128),
		nn.ReLU(),
		nn.Linear(128, 128),
		nn.ReLU(),
		nn.Linear(128, 2),
		nn.Softmax()
)
critic = nn.Sequential(
		nn.Linear(4, 128),
		nn.ReLU(),
		nn.Linear(128, 128),
		nn.ReLU(),
		nn.Linear(128, 1)
)

fig, axs = plt.subplots(1, 2)

for i in range(10):
		state = tensor([1.0, 0.0, 0.0, 0.0])
		action_probs = actor(state)
		action = torch.multinomial(action_probs, 1)
		
		# Simulate a reward
		with torch.no_grad():
			reward = tensor([1.0]) if action.item() == 0 else tensor([-1.0])
		
		# Critic evaluation
		value = critic(state)
		
		# Loss calculation
		advantage = reward - value
		actor_loss = -torch.log(action_probs[action]) * advantage
		critic_loss = advantage.pow(2)
		
		# Backpropagation
		actor.zero_grad()
		critic.zero_grad()
		(actor_loss + critic_loss).backward()
		
		# Update weights
		for param in actor.parameters():
				param.data -= 0.01 * param.grad.data
		for param in critic.parameters():
				param.data -= 0.01 * param.grad.data

		print(action_probs)
		axs[0].imshow(action_probs.detach()[:,None], interpolation='nearest', vmin=0,vmax=1)
		axs[1].clear()
		axs[1].text(0.4,0.4,value.item())
		plt.pause(1)
