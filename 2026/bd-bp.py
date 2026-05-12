# Mar 9, 2026
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import av, io
from datasets import load_dataset

def initWeights(shape):
	w = torch.empty(shape)
	nn.init.kaiming_uniform_(w.reshape(shape[0],-1).T, nonlinearity="relu")
	return w

layers = []
sLayers = []
pLayers = []
params = []
toOptimize = None

# --- Connection types ---

class Connection:
	"""Base class. Subclass to add new shapes (conv, pooling, etc.)."""
	def forward(self, x): raise NotImplementedError
	def allParams(self): raise NotImplementedError
	def homeostaticScale(self, _scale): raise NotImplementedError

class DenseConnection(nn.Linear, Connection):
	def __init__(self, in_f, out_f, positive_only=False):
		super().__init__(in_f, out_f, bias=not positive_only)
		self.positive_only = positive_only
	@property
	def param(self): return self.weight
	def forward(self, x):
		b = None if self.positive_only else self.bias
		return F.linear(x.flatten(), self.weight.clone(), b)
	def allParams(self):
		return list(self.parameters())
	def clampWeights(self):
		if self.positive_only: self.weight.clamp_(min=0)
	def homeostaticScale(self, scale):
		self.weight *= scale.unsqueeze(1)
		if not self.positive_only: self.bias *= scale
	def synapticStrip(self, dead, strip_frac=0.1):
		# for dead neurons, zero out their most negative incoming weights
		threshold = self.weight.detach().quantile(strip_frac, dim=1, keepdim=True)
		mask = (self.weight.detach() < threshold) & (dead.unsqueeze(1) > 1.0)
		self.weight.data[mask] = 0.0
	def shiftPositive(self, factor):
		self.weight *= 1 + self.weight.detach().sign() * factor.unsqueeze(1)
		if not self.positive_only and self.bias is not None:
			self.bias += factor * 0.01
	@classmethod
	def make(cls, fromLayer, toLayer, positive_only=False):
		return cls(fromLayer.size, toLayer.size, positive_only)
	@classmethod
	def makeTranspose(cls, existingParam, fromLayer, toLayer, positive_only=False):
		conn = cls(fromLayer.size, toLayer.size, positive_only)
		with torch.no_grad(): conn.weight.copy_(existingParam.T)
		return conn

class ConvConnection(Connection):
	"""2D convolution. Handles same size, downsampling, and upsampling via stride."""
	def __init__(self, param, bias, in_shape, out_shape, stride, upsample, kernel_size, positive_only=False):
		self.param = param
		self.bias = bias
		self.in_shape = in_shape
		self.out_shape = out_shape
		self.stride = stride
		self.upsample = upsample
		self.kernel_size = kernel_size
		self.positive_only = positive_only
	def forward(self, x):
		x = x.reshape(1, *self.in_shape)
		pad = self.kernel_size // 2
		w = self.param.clone()
		b = None if self.positive_only else self.bias
		if self.upsample:
			_, _, H_out, W_out = (1,) + self.out_shape
			x = F.interpolate(x, size=(H_out, W_out), mode='bilinear', align_corners=False)
			out = F.conv2d(x, w, bias=b, padding=pad)
		else:
			out = F.conv2d(x, w, bias=b, stride=self.stride, padding=pad)
		return out.reshape(-1)
	def allParams(self):
		return [self.param] if self.positive_only else [self.param, self.bias]
	def clampWeights(self):
		if self.positive_only: self.param.clamp_(min=0)
	def homeostaticScale(self, scale):
		C_out = self.out_shape[0]
		self.param *= scale.reshape(C_out, -1).mean(1).reshape(-1, 1, 1, 1)
		if not self.positive_only:
			self.bias *= scale.reshape(C_out, -1).mean(1)
	def synapticStrip(self, dead, strip_frac=0.1):
		C_out = self.out_shape[0]
		ch_dead = dead.reshape(C_out, -1).mean(1)  # (C_out,)
		w = self.param.detach()
		threshold = w.reshape(C_out, -1).quantile(strip_frac, dim=1).reshape(-1, 1, 1, 1)
		mask = (w < threshold) & (ch_dead.reshape(-1, 1, 1, 1) > 1.0)
		self.param.data[mask] = 0.0
	def shiftPositive(self, factor):
		C_out = self.out_shape[0]
		ch_factor = factor.reshape(C_out, -1).mean(1)
		self.param *= 1 + self.param.detach().sign() * ch_factor.reshape(-1, 1, 1, 1)
		if not self.positive_only:
			self.bias += ch_factor * 0.01
	@classmethod
	def make(cls, fromLayer, toLayer, kernel_size=9, positive_only=False):
		C_in, H_in = fromLayer.shape[0], fromLayer.shape[1]
		C_out, H_out = toLayer.shape[0], toLayer.shape[1]
		if H_out > H_in:
			stride, upsample = H_out // H_in, True
			w = torch.empty(C_out, C_in, kernel_size, kernel_size)
		else:
			stride, upsample = H_in // H_out, False
			w = torch.empty(C_out, C_in, kernel_size, kernel_size)
		#w.fill_(1.0/(C_out*C_in*kernel_size*kernel_size))
		#w += (torch.rand(w.shape)-0.5)/(C_out*C_in*kernel_size*kernel_size)
		nn.init.kaiming_uniform_(w, nonlinearity="relu")
		#center = kernel_size // 2
		#coords = torch.arange(kernel_size).float() - center
		#g = torch.exp(-coords**2 / (2 * (kernel_size / 4.0)**2))
		#mask = (g.unsqueeze(0) * g.unsqueeze(1))  # (kH, kW), peak=1 at center
		#mask /= mask.max()
		#w *= mask  # broadcast over channel dims
		bias = nn.Parameter(torch.zeros(C_out))
		#bias = nn.Parameter(torch.ones(C_out) * (6.0 / (C_in * kernel_size * kernel_size)) ** 0.5)
		return cls(nn.Parameter(w), bias, fromLayer.shape, toLayer.shape, stride, upsample, kernel_size, positive_only=positive_only)
	@classmethod
	def makeTranspose(cls, _existingParam, fromLayer, toLayer, kernel_size=9, positive_only=False):
		return cls.make(fromLayer, toLayer, kernel_size, positive_only=positive_only)

class LocalConnection(Connection):
	"""Locally connected: param (C_out, C_in*kH*kW, P_out) for both up/downsample."""
	def __init__(self, param, bias, in_shape, out_shape, stride, upsample, kernel_size, p_in_idx=None, positive_only=False):
		self.param = param
		self.bias = bias
		self.in_shape = in_shape
		self.out_shape = out_shape
		self.stride = stride
		self.upsample = upsample
		self.kernel_size = kernel_size
		self.p_in_idx = p_in_idx
		self.positive_only = positive_only
	def forward(self, x):
		pad = self.kernel_size // 2
		x = x.reshape(1, *self.in_shape)
		w = self.param.clone()
		if self.upsample:
			patches_in = F.unfold(x, self.kernel_size, padding=pad)
			patches = patches_in[0, :, self.p_in_idx]
		else:
			patches = F.unfold(x, self.kernel_size, stride=self.stride, padding=pad)[0]
		out = (w * patches.unsqueeze(0)).sum(1)
		if not self.positive_only: out = out + self.bias
		return out.reshape(-1)
	def allParams(self):
		return [self.param] if self.positive_only else [self.param, self.bias]
	def clampWeights(self):
		if self.positive_only: self.param.clamp_(min=0)
	def homeostaticScale(self, scale):
		self.param *= scale.reshape(self.out_shape[0], 1, -1)
		if not self.positive_only:
			self.bias *= scale.reshape(self.out_shape[0], -1)
	def synapticStrip(self, dead, strip_frac=0.1):
		C_out = self.out_shape[0]
		w = self.param.detach()  # (C_out, D, P_out)
		threshold = w.quantile(strip_frac, dim=1, keepdim=True)  # (C_out, 1, P_out) per-neuron
		dead_mask = dead.reshape(C_out, 1, -1) > 1.0
		self.param.data[(w < threshold) & dead_mask] = 0.0
		self.dead_mask=dead_mask
	def shiftPositive(self, factor):
		C_out = self.out_shape[0]
		self.param *= 1 + self.param.detach().sign() * factor.reshape(C_out, 1, -1)
		if not self.positive_only:
			self.bias += factor.reshape(C_out, -1) * 0.01
	@classmethod
	def make(cls, fromLayer, toLayer, kernel_size=3, positive_only=False):
		C_in, H_in, W_in = fromLayer.shape
		C_out, H_out, W_out = toLayer.shape
		D = C_in * kernel_size * kernel_size
		P_out = H_out * W_out
		w = nn.init.kaiming_uniform_(torch.empty(C_out*P_out, D), nonlinearity='relu').reshape(C_out, P_out, D).permute(0,2,1)
		if H_out > H_in:
			stride = H_out // H_in
			oh = torch.arange(H_out) // stride
			ow = torch.arange(W_out) // stride
			p_in_idx = (oh.unsqueeze(1) * W_in + ow.unsqueeze(0)).reshape(-1)
		else:
			stride = max(1, H_in // H_out)
			p_in_idx = None
		bias = nn.Parameter(torch.zeros(C_out, P_out))
		#bias = nn.Parameter(torch.ones(C_out, P_out) * (6.0 / (D)) ** 0.5)
		return cls(nn.Parameter(w), bias, fromLayer.shape, toLayer.shape, stride, H_out > H_in, kernel_size, p_in_idx, positive_only=positive_only)
	@classmethod
	def makeTranspose(cls, _existingParam, fromLayer, toLayer, kernel_size=3, positive_only=False):
		return cls.make(fromLayer, toLayer, kernel_size, positive_only=positive_only)

"""a=torch.stack((torch.rand(10,1),torch.zeros(10,1),torch.zeros(10,1)))
print(a)
class xx:pass
f=xx()
t=xx()
f.shape=(3,10,1)
t.shape=(3,20,1)
b=LocalConnection.make(f,t).forward(a)
print(b)
exit()"""

# --- Layer ---

class Layer:
	def __init__(self, shape, noappend=False):
		layers.append(self)
		if not noappend: sLayers.append(self)
		self.shape = shape  # (C, H, W) for conv layers; None for flat
		self.size = shape
		if isinstance(shape,tuple):
			self.size = 1
			for i in range(len(shape)):
				self.size *= shape[i]
		self.v = torch.zeros(self.size)
		self.connections = {}
		self.backConnections = {}
		self.pConnections = {}
		self.avgAct = torch.full((self.size,), 0.15)  # init at target so startup doesn't divide near-zero
		self.avgForwardInput = torch.full((self.size,), 1.0)
		self.avgBackInput = torch.full((self.size,), 0.75)
		self.avgPInput = torch.full((self.size,), 0.1)
		self.avgForwardInputMean = torch.zeros(self.size)
		self.avgBackInputMean = torch.zeros(self.size)
		self.avgPInputMean = torch.zeros(self.size)

	def connectFromP(layer, otherLayer, conn_type=DenseConnection, **kwargs):
		conn = conn_type.make(otherLayer, layer, **kwargs)
		layer.pConnections[otherLayer] = conn
		params.extend(conn.allParams())
		return conn
	
	positive_only=False
	def updateV(self,forwardInput,backInput,pInput):
		#backInput = backInput * forwardInput.detach().amax().clamp(0.01)/backInput.detach().amax().clamp(0.01) * 0.5
		input = forwardInput+backInput*0.5+pInput*0.1
		#input = input.clamp_max(2.0)
		if self.positive_only:
			if isinstance(self.shape, tuple) and len(self.shape) == 3:
				input = input.view(self.shape)
				inputAdj = input.detach()
				feedforwardInhibition = (inputAdj.mean(0,keepdim=True)*0.7 + inputAdj.amax(0,keepdim=True)*0.3 - 0.05).clamp_min(0.0)
				input = (input - feedforwardInhibition).view(self.size)
			else:
				inputAdj = input.detach()
				feedforwardInhibition = ((inputAdj.mean()*0.7+inputAdj.amax()*0.3) - 0.05).clamp_min(0.0)
				input = input - feedforwardInhibition
			#input = input / input.amax().clamp(0.01)
			input = (input.clamp(0)*8.0).tanh()
		else:
			"""if isinstance(self.shape, tuple) and len(self.shape) == 3:
				input = input.view(self.shape)
				inputAdj = input.detach()
				feedforwardInhibition = (inputAdj.mean(0,keepdim=True)*0.7 + inputAdj.amax(0,keepdim=True)*0.3 - 0.05).clamp_min(0.0)
				input = (input - feedforwardInhibition).view(self.size) * 8.0"""
			input = F.relu(input)
			input = input / input.detach().amax().clamp(1.0)
		"""if isinstance(self.shape, tuple) and len(self.shape) == 3:
			C, H, W = self.shape
			vs = v.reshape(C, H*W)
			# divide by historical channel activity before competition:
			# channels that have been dormant get a relative boost, breaking the
			# always-on/never-on lock where instantaneous inhibition undoes homeostasis
			hist = self.avgAct.reshape(C, H*W).mean(1, keepdim=True).clamp(min=0.01)
			vs = vs / hist
			# per-channel spatial norm
			cadj = vs.detach().abs()
			ch_inhib = (cadj.mean(1,keepdim=True)*0.7 + cadj.amax(1,keepdim=True)*0.3 - 0.05).clamp_min(0.0)
			vs = vs / ch_inhib.clamp(1.0)
			# per-position channel competition
			padj = vs.detach().abs()
			pos_inhib = (padj.mean(0,keepdim=True)*0.7 + padj.amax(0,keepdim=True)*0.3 - 0.05).clamp_min(0.0)
			v = (vs / pos_inhib.clamp(1.0)).reshape(-1)
		else:"""
		#outAdj = v.detach().abs()
		#outInhibition = ((outAdj.mean()*0.7+outAdj.amax()*0.3) - 0.05).clamp_min(0.0)
		#v = v / outInhibition.clamp(1.0)
		self.v = input
		global toOptimize
		if toOptimize is not None:
			if hasattr(self,"prevV"):
				rho = 0.15
				p = self.avgAct.clamp(1e-6, 1 - 1e-6)
				kl_grad = (-rho / p + (1 - rho) / (1 - p)).detach()
				toOptimize = toOptimize - (kl_grad * self.prevV).mean() * 0.1
				v_c = self.prevV - self.avgAct.detach()
				toOptimize = toOptimize - ((v_c**2 - 1.0)**2).mean() * 0.01
			if hasattr(self, "prevForwardInput"):
				fi_c = self.prevForwardInput - self.avgForwardInputMean.detach()
				bi_c = self.prevBackInput - self.avgBackInputMean.detach()
				pi_c = self.prevPInput - self.avgPInputMean.detach()
				var_loss = (
					((fi_c**2 - 1.0)**2).mean() +
					((bi_c**2 - 1.0)**2).mean() +
					((pi_c**2 - 1.0)**2).mean()
				) * 0.01
				toOptimize = toOptimize - var_loss
		self.avgAct += (self.v.detach() - self.avgAct) * 0.01
		self.avgForwardInput += (forwardInput.detach().abs() - self.avgForwardInput) * 0.01
		self.avgBackInput += (backInput.detach().abs() - self.avgBackInput) * 0.01
		self.avgPInput += (pInput.detach().abs() - self.avgPInput) * 0.01
		self.avgForwardInputMean += (forwardInput.detach() - self.avgForwardInputMean) * 0.01
		self.avgBackInputMean += (backInput.detach() - self.avgBackInputMean) * 0.01
		self.avgPInputMean += (pInput.detach() - self.avgPInputMean) * 0.01
		if toOptimize is not None:
			self.prevForwardInput = forwardInput
			self.prevBackInput = backInput
			self.prevPInput = pInput

	def update(self):
		x = torch.zeros(self.size)
		for other, conn in self.connections.items():
			x = x + conn.forward(other.v)
		bx = torch.zeros(self.size)
		for other, conn in self.backConnections.items():
			bx = bx + conn.forward(other.v) #* 0.2
		px = torch.zeros(self.size)
		for other, conn in self.pConnections.items():
			px = px + conn.forward(other.v)
		self.updateV(x,bx,px)

	def homeostaticAdjust(self):
		target = self.avgAct.mean() # 0.15
		s = self.avgAct
		ratio = target / s.clamp(min=1e-4)
		#scale = torch.lerp(torch.ones_like(ratio), ratio, 0.01).clamp(0.5, 2.0)
		scale = torch.pow(ratio.clamp(0.1, 10.0), 1/350) #* 0.2 + 0.8
		self.forwardAdjust = torch.pow((1.0 / self.avgForwardInput.clamp(min=1e-4)).clamp(0.5, 2.0), 1/35)
		self.backwardAdjust = torch.pow((0.5 / self.avgBackInput.clamp(min=1e-4)).clamp(0.5, 2.0), 1/35)
		self.pAdjust = torch.pow((0.1 / self.avgPInput.clamp(min=1e-4)).clamp(0.5, 2.0), 1/35)
		# for negative neurons, add a small positive amount to all their weights
		#neg_bias = (target - s)#*0.1 #torch.where(s<0.01, 0.5, (target - s)*0.1) #(-s).clamp(min=0) * 0.01
		#self.adjust = neg_bias*0.035
		#self.adjustp = torch.where(s<0.05, self.adjust, 0.0)
		dead = (0.01 - (self.avgAct).clamp(0,0.01)) / 0.005
		self.forwardAdjust[dead>1.0] = 1.0
		self.backwardAdjust[dead>1.0] = 1.0
		with torch.no_grad():
			for conn in self.connections.values():
				conn.synapticStrip(dead, 0.1)
			for conn in self.backConnections.values():
				conn.synapticStrip(dead, 0.1)
			for conn in self.pConnections.values():
				conn.synapticStrip(dead, 0.1)
	def homeostaticAdjust2(self):
		return
		if not hasattr(self,"forwardAdjust"):return
		with torch.no_grad():
			for conn in self.connections.values():
				conn.homeostaticScale(self.forwardAdjust)
			for conn in self.backConnections.values():
				conn.homeostaticScale(self.backwardAdjust)
			for conn in self.pConnections.values():
				conn.homeostaticScale(self.pAdjust)


def smaller(x):
	return x#x.detach()*0.9 + x*0.1

class PLayer(Layer):
	def __init__(self,shape,lowerLayer):
		super().__init__(shape, noappend=True)
		pLayers.append(self)
		self.lowerLayer=lowerLayer
		self.ctxConnections = {}
	def update(self):
		x = torch.zeros(self.size)
		for other, conn in self.ctxConnections.items():
			if hasattr(other,"prevV"): x = x + conn.forward(smaller(other.prevV))
		for other, conn in self.connections.items():
			x = x + conn.forward(other.v)
		bx = torch.zeros(self.size)
		for other, conn in self.backConnections.items():
			bx = bx + conn.forward(other.v) #* 0.2
		#self.updateV(x+bx)
		self.v = F.leaky_relu(x+bx, 0.1)

		global toOptimize
		if toOptimize is not None:
			toOptimize = toOptimize - F.mse_loss(self.v,self.lowerLayer.v.detach())
		self.v = self.v.detach()
	def homeostaticAdjust(self): pass

"""class VLayer(Layer):
	def __init__(self,shape):
		super().__init__(shape)
	def update(self):
		x = torch.zeros(self.size)
		for other, conn in self.connections.items():
			x = x + conn.forward(other.v)
		bx = torch.zeros(self.size)
		for other, conn in self.backConnections.items():
			bx = bx + conn.forward(other.v) #* 0.2
		a = x+bx
		if isinstance(self.shape, tuple) and len(self.shape) == 3:
			C, H, W = self.shape
			av = a.reshape(C, H, W)
			# average the 8 surrounding neighbors for each channel independently
			av_pad = F.pad(av.unsqueeze(0), (2,2,2,2)).squeeze(0)  # (C, H+2, W+2)
			patches = av_pad.unfold(1, 5, 1).unfold(2, 5, 1)  # (C, H, W, 3, 3)
			mask = torch.ones(5, 5); mask[1, 1] = 0.0
			surround = (patches * mask).sum((-2, -1)) / 25.0  # (C, H, W)
			surround = surround.reshape(-1)
		else:
			surround = a
		self.updateV(x+bx+surround)
		"""


# --- Connection helpers ---

def connect(otherLayer, layer, conn_type=DenseConnection, **kwargs):
	conn = conn_type.make(otherLayer, layer, **kwargs)
	layer.connections[otherLayer] = conn
	params.extend(conn.allParams())
	return conn

def connectBidir(otherLayer, layer, conn_type=DenseConnection, **kwargs):
	conn = conn_type.make(otherLayer, layer, **kwargs)
	backConn = conn_type.makeTranspose(conn.param, layer, otherLayer, **kwargs)
	layer.connections[otherLayer] = conn
	otherLayer.backConnections[layer] = backConn
	params.extend(conn.allParams())
	params.extend(backConn.allParams())
	return conn, backConn

def connectCtx(otherLayer, layer, conn_type=DenseConnection, **kwargs):
	conn = conn_type.make(otherLayer, layer, **kwargs)
	layer.ctxConnections[otherLayer] = conn
	params.extend(conn.allParams())


# --- Network ---

v1=Layer((3,64,64))
v1.update = lambda: None
v1.homeostaticAdjust = lambda: None
#v1adj=Layer((3,64,64))
#v1adj.update = lambda: None
#v1adj.homeostaticAdjust = lambda: None
v1p=PLayer((3,64,64),v1)
v1p.homeostaticAdjust = lambda: None
v2=Layer((6,32,32))
v2p=PLayer((6,32,32),v2)
v3=Layer((11,16,16))
v3p=PLayer((11,16,16),v3)
v4=Layer((16,8,8))
v4p=PLayer((16,8,8),v4)
"""connect(v1,v2, ConvConnection)#temp
connect(v2,v3, ConvConnection)#temp
connect(v3,v2p, ConvConnection)#temp
connect(v2p,v1p, ConvConnection)#temp
layers=[v1,v2,v3,v2p,v1p]
#connect(v2,v1p, LocalConnection)#temp"""
connectBidir(v1,v2, ConvConnection, positive_only=False)
connectCtx(v2,v1p, LocalConnection)
v2.connectFromP(v1p, ConvConnection)
#connect(v1p,v2, ConvConnection)
connect(v2p,v1p, LocalConnection)
connectBidir(v2,v3, ConvConnection)
connectCtx(v3,v2p, LocalConnection)
#connect(v2p,v3, ConvConnection)
v3.connectFromP(v2p, ConvConnection)
connect(v3p,v2p, LocalConnection)
"""connectBidir(v3,v4, LocalConnection)
connectCtx(v4,v3p, LocalConnection)
connect(v3p,v4, LocalConnection)
connect(v4p,v3p, LocalConnection)
connectBidir(v2,v4, LocalConnection)
connectCtx(v4,v2p, LocalConnection)
connect(v2p,v4, LocalConnection)"""

e1=Layer((16,8,8))
"""connectBidir(v3,e1, LocalConnection)
connectCtx(e1,v3p, LocalConnection)
connect(v3p,e1, LocalConnection)"""

a1=Layer((16,16))
a1p=PLayer((16,16),a1)
a2=Layer((16,16))
"""_, back = connectBidir(v4,a1, DenseConnection)
#with torch.no_grad(): back.param *= 0.2
connectCtx(a1,v4p, DenseConnection)
connect(v4p,a1, DenseConnection)
connectBidir(a1,a2, DenseConnection)
connectCtx(a2,a1p, DenseConnection)
connect(a1p,a2, DenseConnection)

conn = connect(a1,a1, DenseConnection)
with torch.no_grad(): conn.param *= 0.1
conn = connect(a2,a2, DenseConnection)
with torch.no_grad(): conn.param *= 0.1"""


fig, axs = plt.subplots(5, 4)
plt.tight_layout(pad=0)
plt.subplots_adjust(wspace=0.05, hspace=0.05)
for ax in axs.flat: ax.tick_params(labelbottom=False, labelleft=False)

def showAt(ax, v, title, clamp=True):
	ax.clear()
	ax.set_title(title, fontsize=7)
	ax.axis('off')
	if v is None: return
	v = v.detach().float()
	if v.dim() == 3:
		C, H, W = v.shape
		if C == 3:  # show as RGB
			img = v.permute(1,2,0)
			#lo, hi = img.min(), img.max()
			ax.imshow(img.clamp(0,1).numpy())
		else:  # tile each channel as grayscale in a grid
			cols = int(C**0.5 + 0.999)
			rows = (C + cols - 1) // cols
			grid = torch.zeros(rows * H, cols * W)
			for c in range(C):
				r, cc = divmod(c, cols)
				ch = v[c]
				#lo, hi = ch.min(), ch.max()
				grid[r*H:(r+1)*H, cc*W:(cc+1)*W] = ch #(ch - lo) / (hi - lo + 1e-8)
			if clamp:
				ax.imshow(grid.numpy(), cmap='viridis', vmin=0, vmax=1)
			else:
				ax.imshow(grid.numpy())
	else:  # 1D or 2D weights/grads
		img = v.reshape(1,-1) if v.dim()==1 else v.reshape(v.shape[0],-1)
		hi = img.abs().max().clamp(min=1e-8)
		ax.imshow(img.numpy(), vmin=-hi, vmax=hi, cmap='RdBu', aspect='auto')

def layerV(l):
	v = l.v.detach()
	if isinstance(l.shape, tuple): v = v.reshape(l.shape)
	return v

def plotThem(i):
	showAt(axs[0,0], layerV(v1),  "v1 (input)")
	showAt(axs[0,1], layerV(v1p), "v1p (pred)")
	showAt(axs[0,2], v2.connections[v1].param.flatten(0,1),  "weights", False)
	showAt(axs[1,0], layerV(v2),  "v2")
	showAt(axs[1,1], layerV(v2p), "v2p (pred)")
	showAt(axs[2,0], layerV(v3),  "v3")
	showAt(axs[2,1], layerV(v3p), "v3p (pred)")
	#showAt(axs[3,0], v1p.ctxConnections[v2].param.flatten(0,1),  "weights", False)
	#showAt(axs[3,1], (v1p.ctxConnections[v2].param.grad.flatten(0,1) if v1p.ctxConnections[v2].param.grad is not None else tensor([404])),  "weights grad", False)
	#showAt(axs[3,2], (v2.connections[v1].param.grad.flatten(0,1) if v2.connections[v1].param.grad is not None else tensor([404])),  "weights grad", False)
	#showAt(axs[3,0], layerV(v4),  "v4")
	#showAt(axs[3,1], layerV(v4p),  "v4p")
	#showAt(axs[3,2], layerV(e1),  "e1")
	showAt(axs[1,2], layerV(a1), "a1")
	showAt(axs[1,3], layerV(a1p), "a1p")
	showAt(axs[2,2], layerV(a2),  "a2")
	showAt(axs[4,0], v2.avgForwardInput.reshape(v2.shape),  "v2 f avg")
	showAt(axs[4,1], v2.avgBackInput.reshape(v2.shape),  "v2 avgBackInput")
	showAt(axs[4,2], v2.avgPInput.reshape(v2p.shape),  "v2 avgPInput")
	#showAt(axs[4,2], a1.avgAct.reshape(a1.shape),  "a1 avg")
	axs[0,3].clear(); axs[0,3].axis('off')
	loss_val = toOptimize.item() if toOptimize is not None and toOptimize.grad_fn is None else (toOptimize.detach().item() if toOptimize is not None else float('nan'))
	axs[0,3].text(0.1, 0.5, f"loss: {loss_val}\ni: {i}", transform=axs[0,3].transAxes)
	# similarity matrix
	axs[3,3].clear(); axs[3,3].set_title('video similarity', fontsize=7); axs[3,3].axis('off')
	if len(video_sigs) >= 2:
		sigs = torch.stack(video_sigs)
		sigs = sigs / sigs.norm(dim=1, keepdim=True).clamp(min=1e-8)
		sim = (sigs @ sigs.T).numpy()
		axs[3,3].imshow(sim, vmin=-1, vmax=1, cmap='RdBu', aspect='auto')
		axs[3,3].axis('on')
	plt.pause(0.25)

#continue: https://huggingface.co/datasets?modality=modality:video&sort=trending&search=mine
class VideoDataset:
	def __init__(self, frame_shape):
		self.frame_shape = frame_shape[1:]  # (H, W)
		self.video_id = 0
		# aisuko/ucf101-subset: 405 real UCF-101 clips, raw AVI bytes, no login needed
		hf = load_dataset("aisuko/ucf101-subset", split="train", streaming=True)
		self._iter = iter(hf)
		self._frames = []
		self._frame_idx = 0
		self._load_next()

	def _decode(self, video_bytes, skip=10):
		H, W = self.frame_shape
		container = av.open(io.BytesIO(video_bytes))
		stream = container.streams.video[0]
		stream.thread_type = 'AUTO'  # use all available CPU threads for decoding
		frames = []
		for i, frame in enumerate(container.decode(stream)):
			if i % skip != 0: continue
			# reformat resizes inside av — no separate interpolate needed
			img = torch.from_numpy(frame.reformat(width=W, height=H, format='rgb24').to_ndarray())
			frames.append(img.permute(2,0,1).float() / 255.0)  # (C,H,W)
		if not frames: return None
		return torch.stack(frames)  # (T,C,H,W)

	def _load_next(self):
		while True:
			item = next(self._iter)
			video_bytes = item.get('avi')  # raw AVI bytes field
			if not video_bytes: continue
			frames = self._decode(video_bytes)
			if frames is not None:
				self._frames = frames
				self._frame_idx = 0
				self.video_id += 1
				return

	def next(self):
		if self._frame_idx >= len(self._frames):
			self._load_next()
		frame = self._frames[self._frame_idx].flatten()
		self._frame_idx += 1
		return frame

torch.manual_seed(123)
video_data = VideoDataset(v1.shape)

# --- Activation similarity tracking ---
SIM_LAYER    = a1    # layer whose activations represent each video
MAX_SIGS     = 20    # max videos to keep in the matrix
video_sigs   = []    # (N, size) mean activation per video
_sig_vid_id  = -1
_sig_acts    = []

def update_sigs():
	global _sig_vid_id, _sig_acts
	if video_data.video_id != _sig_vid_id:
		if _sig_acts:
			sig = torch.stack(_sig_acts).mean(0).detach()
			video_sigs.append(sig)
			if len(video_sigs) > MAX_SIGS: video_sigs.pop(0)
		_sig_vid_id = video_data.video_id
		_sig_acts   = []
	_sig_acts.append(SIM_LAYER.v.detach().clone())

# Adam state
adam_lr = 0.0001
adam_b1, adam_b2, adam_eps = 0.5, 0.999, 1e-8
adam_m = [torch.zeros_like(p) for p in params]
adam_v = [torch.zeros_like(p) for p in params]
adam_t = 0

for i in range(8000):
	v = video_data.next()
	v1.v = v
	toOptimize = None
	for l in reversed(sLayers): l.update()
	toOptimize = tensor(0.0)
	for l in sLayers: l.update()
	for l in reversed(pLayers): l.update()
	if toOptimize.grad_fn is not None: toOptimize.backward()
	for l in layers:
		#if hasattr(l,"postUpdate"): l.postUpdate()
		l.prevV = l.v
	for l in layers: l.v = l.v.detach()
	update_sigs()

	plotThem(i)
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
		for l in layers:
			for conn in l.connections.values():
				conn.clampWeights()
			for conn in l.backConnections.values():
				conn.clampWeights()

	if i % 35 == 34:
		for l in layers:
			l.homeostaticAdjust()
	for l in layers: l.homeostaticAdjust2()
