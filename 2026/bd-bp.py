# Mar 9, 2026
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def initWeights(shape):
	w = torch.empty(shape)
	nn.init.kaiming_uniform_(w.reshape(shape[0],-1).T)
	return w

layers = []
params = []
toOptimize = None

# --- Connection types ---

class Connection:
	"""Base class. Subclass to add new shapes (conv, pooling, etc.)."""
	def forward(self, x): raise NotImplementedError
	def allParams(self): raise NotImplementedError
	def homeostaticScale(self, _scale): raise NotImplementedError

class DenseConnection(Connection):
	def __init__(self, param):
		self.param = param
	def forward(self, x):
		return x.flatten() @ self.param.clone()
	def allParams(self):
		return [self.param]
	def homeostaticScale(self, scale):
		# scale: (post_size,) — broadcast over pre_size dim
		self.param *= scale.unsqueeze(0)
	@classmethod
	def make(cls, fromLayer, toLayer):
		return cls(nn.Parameter(initWeights((fromLayer.size, toLayer.size))))
	@classmethod
	def makeTranspose(cls, existingParam, _fromLayer, _toLayer):
		return cls(nn.Parameter(existingParam.transpose(0,1).clone()))

class ConvConnection(Connection):
	"""2D convolution. Handles same size, downsampling, and upsampling via stride."""
	def __init__(self, param, in_shape, out_shape, stride, upsample, kernel_size):
		self.param = param
		self.in_shape = in_shape    # (C, H, W)
		self.out_shape = out_shape  # (C, H, W)
		self.stride = stride
		self.upsample = upsample    # True → conv_transpose2d, param is (C_in, C_out, kH, kW)
		self.kernel_size = kernel_size
	def forward(self, x):
		x = x.reshape(1, *self.in_shape)
		pad = self.kernel_size // 2
		if self.upsample:
			out = F.conv_transpose2d(x, self.param.clone(), stride=self.stride,
			                         padding=pad, output_padding=self.stride-1)
		else:
			out = F.conv2d(x, self.param.clone(), stride=self.stride, padding=pad)
		return out.reshape(-1)
	def allParams(self):
		return [self.param]
	def homeostaticScale(self, scale):
		C_out = self.out_shape[0]
		channel_scale = scale.reshape(C_out, -1).mean(dim=1)
		# upsample param: (C_in, C_out, kH, kW) → output channels at dim 1
		# downsample/same param: (C_out, C_in, kH, kW) → output channels at dim 0
		if self.upsample:
			self.param *= channel_scale.reshape(1, -1, 1, 1)
		else:
			self.param *= channel_scale.reshape(-1, 1, 1, 1)
	@classmethod
	def make(cls, fromLayer, toLayer, kernel_size=3):
		C_in, H_in = fromLayer.shape[0], fromLayer.shape[1]
		C_out, H_out = toLayer.shape[0], toLayer.shape[1]
		if H_out > H_in:
			stride, upsample = H_out // H_in, True
			w = torch.empty(C_in, C_out, kernel_size, kernel_size)  # transposed conv shape
		else:
			stride, upsample = H_in // H_out, False
			w = torch.empty(C_out, C_in, kernel_size, kernel_size)
		nn.init.kaiming_uniform_(w)
		return cls(nn.Parameter(w), fromLayer.shape, toLayer.shape, stride, upsample, kernel_size)
	@classmethod
	def makeTranspose(cls, _existingParam, fromLayer, toLayer, kernel_size=3):
		return cls.make(fromLayer, toLayer, kernel_size)

# --- Layer ---

class Layer:
	def __init__(self, shape):
		layers.append(self)
		self.shape = shape  # (C, H, W) for conv layers; None for flat
		self.size = shape
		if isinstance(shape,tuple):
			self.size = 1
			for i in range(len(shape)):
				self.size *= shape[i]
		self.v = torch.zeros(self.size)
		self.connections = {}
		self.backConnections = {}
		self.avgAct = torch.zeros(self.size)

	def updateV(self,input):
		inputAdj = input.detach().abs()
		feedforwardInhibition = ((inputAdj.mean()*0.7+inputAdj.amax()*0.3) - 0.05).clamp_min(0.0)
		input = input / feedforwardInhibition.clamp(1.0)
		self.v = torch.tanh(input)
		self.v = self.v.detach() + input-input.detach()
		self.avgAct += (self.v.detach()-self.avgAct)*0.1

	def update(self):
		x = torch.zeros(self.size)
		for other, conn in self.connections.items():
			x = x + conn.forward(other.v)
		bx = torch.zeros(self.size)
		for other, conn in self.backConnections.items():
			bx = bx + conn.forward(other.v) * 0.2
		self.bx=bx
		self.updateV(x+bx)

	def homeostaticAdjust(self):
		"""Scale down incoming weights of neurons that are too active."""
		target = 0.15
		scale = 1.0 - (self.avgAct.abs() - target).clamp(min=0) * 0.05
		with torch.no_grad():
			for conn in self.connections.values():
				conn.homeostaticScale(scale)
			for conn in self.backConnections.values():
				conn.homeostaticScale(scale)


def smaller(x):
	return x.detach()*0.9 + x*0.1

class PLayer(Layer):
	def __init__(self,shape,lowerLayer):
		super().__init__(shape)
		self.lowerLayer=lowerLayer
		self.prevInputs = {}
	def postUpdate(self):
		for other in self.connections:
			self.prevInputs[other] = other.v
		for other in self.backConnections:
			self.prevInputs[other] = other.v
	def update(self):
		x = torch.zeros(self.size)
		for other, conn in self.connections.items():
			if hasattr(other,"prevV"): x = x + conn.forward(smaller(other.prevV))
		bx = torch.zeros(self.size)
		for other, conn in self.backConnections.items():
			if hasattr(other,"prevV"): bx = bx + conn.forward(smaller(other.prevV)) * 0.2
		self.bx=bx
		self.updateV(x+bx)

		global toOptimize
		if toOptimize is not None:
			toOptimize = toOptimize - F.mse_loss(self.v,self.lowerLayer.v.detach())
		self.v = self.v.detach()


# --- Connection helpers ---

def connect(otherLayer, layer, conn_type=DenseConnection, **kwargs):
	conn = conn_type.make(otherLayer, layer, **kwargs)
	layer.connections[otherLayer] = conn
	params.extend(conn.allParams())

def connectBidir(otherLayer, layer, conn_type=DenseConnection, **kwargs):
	conn = conn_type.make(otherLayer, layer, **kwargs)
	backConn = conn_type.makeTranspose(conn.param, layer, otherLayer, **kwargs)
	layer.connections[otherLayer] = conn
	otherLayer.backConnections[layer] = backConn
	params.extend(conn.allParams())
	params.extend(backConn.allParams())


# --- Network ---

l1=Layer((3,16,16))
l1.update = lambda: None
l1p=PLayer((3,16,16),l1)
l2=Layer((3,32,32))
l2p=PLayer((3,32,32),l2)
l3=Layer((3,32,32))
connectBidir(l1,l2, ConvConnection)
connect(l2,l1p, ConvConnection)
connect(l1p,l2, ConvConnection)
connectBidir(l2,l3, ConvConnection)
connect(l3,l2p, ConvConnection)
connect(l2p,l3, ConvConnection)


fig, axs = plt.subplots(4, 4)
plt.tight_layout()

def showAt(ax, v, title):
	ax.clear()
	ax.set_title(title, fontsize=7)
	ax.axis('off')
	if v is None: return
	v = v.detach().float()
	if v.dim() == 3:  # (C,H,W) → (H,W,C) RGB
		img = v.permute(1,2,0)
		lo, hi = img.min(), img.max()
		img = ((img - lo) / (hi - lo + 1e-8)).clamp(0,1)
		ax.imshow(img.numpy())
	else:  # 1D or 2D weights/grads
		img = v.reshape(1,-1) if v.dim()==1 else v.reshape(v.shape[0],-1)
		hi = img.abs().max().clamp(min=1e-8)
		ax.imshow(img.numpy(), vmin=-hi, vmax=hi, cmap='RdBu', aspect='auto')

def layerV(l):
	v = l.v.detach()
	if isinstance(l.shape, tuple): v = v.reshape(l.shape)
	return v

def plotThem():
	showAt(axs[0,0], layerV(l1),  "l1 (input)")
	showAt(axs[0,1], layerV(l1p), "l1p (pred)")
	showAt(axs[1,0], layerV(l2),  "l2")
	showAt(axs[1,1], layerV(l2p), "l2p (pred)")
	showAt(axs[2,0], layerV(l3),  "l3")
	axs[0,3].clear(); axs[0,3].axis('off')
	loss_val = toOptimize.item() if toOptimize is not None and toOptimize.grad_fn is None else (toOptimize.detach().item() if toOptimize is not None else float('nan'))
	axs[0,3].text(0.1, 0.5, f"loss: {loss_val}", transform=axs[0,3].transAxes)
	plt.pause(1)


torch.manual_seed(123)
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

# Adam state
adam_lr = 0.001
adam_b1, adam_b2, adam_eps = 0.9, 0.999, 1e-8
adam_m = [torch.zeros_like(p) for p in params]
adam_v = [torch.zeros_like(p) for p in params]
adam_t = 0

for i in range(1000):
	l1.v = cube(16,tensor(i*0.1)).reshape(-1)
	toOptimize = None
	for l in reversed(layers): l.update()
	toOptimize = tensor(0.0)
	for l in layers: l.update()
	if toOptimize.grad_fn is not None: toOptimize.backward()
	for l in layers:
		#if hasattr(l,"postUpdate"): l.postUpdate()
		l.prevV = l.v
	for l in layers: l.v = l.v.detach()

	#if i % 10 == 9:
	#	for l in layers:
	#		l.homeostaticAdjust()

	plotThem()
	adam_t += 1
	with torch.no_grad():
		for p, m, v in zip(params, adam_m, adam_v):
			if p.grad is not None:
				m.mul_(adam_b1).add_(p.grad, alpha=1-adam_b1)
				v.mul_(adam_b2).addcmul_(p.grad, p.grad, value=1-adam_b2)
				m_hat = m / (1 - adam_b1**adam_t)
				v_hat = v / (1 - adam_b2**adam_t)
				p += adam_lr * m_hat / (v_hat.sqrt() + adam_eps)
				p.grad.zero_()
