# Mar 11, 2026
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from math import sin, cos

def cube(size=16,a=torch.rand(1)*6.28, offset=(0.0,0.0)):
    V=torch.tensor([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                    [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1.]])
    E=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
       (0,4),(1,5),(2,6),(3,7)]
    # Rotation matrices
    Rx=torch.tensor([[1,0,0],[0,torch.cos(a),-torch.sin(a)],[0,torch.sin(a),torch.cos(a)]])
    Ry=torch.tensor([[torch.cos(a),0,torch.sin(a)],[0,1,0],[-torch.sin(a),0,torch.cos(a)]])
    # Rotate and project
    P=(V@(Ry@Rx).T)
    P=torch.stack([P[:,0]/(P[:,2]+4),P[:,1]/(P[:,2]+4)],1)
    # Apply 2D offset
    ox, oy = offset
    P += torch.tensor([ox, oy])
    # Map to pixel coordinates
    P_pixel = ((P + 0.5) * (size-1)).long().clamp(0,size-1)
    img=torch.zeros(3,size,size)
    # Draw edges with Bresenham-style interpolation
    for i,j in E:
        x0,y0 = P_pixel[i]
        x1,y1 = P_pixel[j]
        dx,dy = abs(x1-x0), abs(y1-y0)
        sx = 1 if x0<x1 else -1
        sy = 1 if y0<y1 else -1
        err = dx-dy
        x,y = x0,y0
        while True:
            img[:,y,x] = 1
            if x==x1 and y==y1: break
            e2 = 2*err
            if e2 > -dy: err -= dy; x += sx
            if e2 < dx: err += dx; y += sy
    return img


class test(nn.Module):
	def __init__(self):
		super().__init__()
		self.layers = nn.ParameterList()
		self.p_layers = nn.ParameterList()

		for i in range(3):
			self.layers.append(nn.Conv2d(3,3, 5, padding=2))
			self.p_layers.append(nn.Conv2d(3,3, 5, padding=2))

	def update(self, x):
		toOptimize = tensor(0.0)
		for i in range(3):
			prevX = x
			x = self.layers[i](x)
			x = F.tanh(x)
			self.layers[i].v = x.detach()
			p = self.p_layers[i](x)
			self.p_layers[i].v = p.detach()
			toOptimize = toOptimize - F.mse_loss(p, prevX.detach())
		toOptimize.backward()
		print("loss: ",toOptimize.item())
		with torch.no_grad():
			for p in self.parameters():
				p += p.grad*0.01

		return x


n=test()

fig, axs = plt.subplots(4, 4)
def plotAt(x,y, v, title):
		if len(v.shape)==3: v=v.permute(1,2,0)#.flatten(1)
		elif len(v.shape)==1: v=[v]
    
		axs[x,y].clear()
		axs[x,y].imshow(v*0.5+0.5)
		axs[x,y].set_title(title)


for i in range(400):
	input = cube(64, tensor(float(i)*0.2), (sin(i*0.456)*0.1,cos(i*0.67)*0.1))
	x = n.update(input)
	plotAt(0,0,input,"input")
	plotAt(1,0,n.layers[0].v,"l0")
	plotAt(1,1,n.p_layers[0].v,"pl0")
	plotAt(2,0,n.layers[1].v,"l1")
	plotAt(2,1,n.p_layers[1].v,"pl1")
	plotAt(3,0,x.detach(),"x")
	plt.pause(1)
