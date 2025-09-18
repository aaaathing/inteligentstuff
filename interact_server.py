import websockets
import asyncio
import json
import numpy as np

class Env:
	connection = None
	async def run(self, onconnect):
		""" Run server. Only accepts one connection """
		async def wshandler(websocket):
			if not self.connection:
				self.connection = websocket
				await onconnect()
		self.server = await websockets.serve(wshandler, "localhost", 8000)
		await self.server.wait_closed()


	video = None
	reward = 0
	async def step(self, action=None):
		if not self.connection:
			return

		await self.connection.send(json.dumps({"action": action.tolist() if action is not None else None}))

		#nextMessage = None
		notdone = True
		while notdone:
			message = await self.connection.recv()
			if isinstance(message, str):
				if message == "video":
					nextMessage = message
				elif message == "done":
					notdone = False
				else:
					message = json.loads(message)
					self.reward = message["reward"]
			elif isinstance(message, bytes):
				if nextMessage == "video":
					self.video = np.frombuffer(message, dtype=np.uint8)

env = Env()

# example
if False:
	import matplotlib.pyplot as plt
	async def loop():
		for i in range(100):
			await env.step([12,34])
			# do computation here
			if env.video is not None:
				plt.imshow(env.video.reshape((128,128,4)))
			plt.pause(0.2)
	asyncio.run(env.run(loop))
