#https://medium.com/@paulswenson2/an-introduction-to-building-custom-reinforcement-learning-environment-using-openai-gym-d8a5e7cf07ea
import random
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

class game:
	def __init__(self):
		self.player_position = np.random.uniform(0.5,5.49, 2)
		self.win_position = np.random.uniform(0.5,5.49, 2)
		self.lose_position = np.random.uniform(0.5,5.49, 2)

	def step(self,x,y):
		self.player_position += [x,y]
		self.player_position = np.clip(self.player_position, 0.5,5.49)
		if np.all(self.player_position+0.5 > self.win_position-0.5) and np.all(self.player_position-0.5 < self.win_position+0.5):
			self.__init__()
			self.reward = 1
		if np.all(self.player_position+0.5 > self.lose_position-0.5) and np.all(self.player_position-0.5 < self.lose_position+0.5):
			self.__init__()
			self.reward = -1
		self.reward = -0.01

		self.video = np.zeros((6,6,3))
		self.video[*np.round(self.player_position).astype(int)] = [0,0,1]
		self.video[*np.round(self.win_position).astype(int)] = [0,1,0]
		self.video[*np.round(self.lose_position).astype(int)] = [1,0,0]

		ax.imshow(self.video)

def example():
	g=game()
	for i in range(100):
		g.step(random.uniform(-1,1),random.uniform(-1,1))
		plt.pause(0.1)