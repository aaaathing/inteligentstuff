import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Module:
    def __init__(self):
        self.modules = []

    def add(self, m):
        self.modules.append(m)
        return m

    def forward(self, net):
        for m in self.modules: m.forward(net)
    def backward(self, net):
        for m in reversed(self.modules): m.backward(net)
    def onPlusPhaseEnd(self, net): pass
    def _onPlusPhaseEnd(self, net):
        self.onPlusPhaseEnd(net)
        for m in self.modules: m._onPlusPhaseEnd(net)
    def onMinusPhaseEnd(self, net): pass
    def _onMinusPhaseEnd(self, net):
        self.onMinusPhaseEnd(net)
        for m in self.modules: m._onMinusPhaseEnd(net)

def flattenShape(shape):
	if isinstance(shape, tuple):
		prod = 1
		for s in shape: prod *= s
		return prod
	else: return shape
def initWeightsSparse(shape):
	chance = min(1.0 / flattenShape(shape[1:]) * 10.0, 1.5)
	w = torch.rand(shape)
	w = (w - (1.0-chance)).clamp_min(0.0) / chance
	w /= w.sum(dim=0, keepdim=True) + 0.0001
	return w

class FullConnection:
    def __init__(self, in_shape, out_shape, initWeights=initWeightsSparse):
        self.weight = nn.Parameter(initWeights((*out_shape, *in_shape)), False)
    def forward(self, x):
        return x @ self.weight.T
    def backward(self, x):
        return x @ self.weight

class ConvConnection:
    def __init__(self, in_shape, out_shape, kernel_size=3, stride=1, padding=1, initWeights=initWeightsSparse):
        self.weight = nn.Parameter(
            initWeights((out_shape[2], in_shape[2], kernel_size, kernel_size)),
            False)
        self.stride = stride
        self.padding = padding
    def forward(self, x):
        return F.conv2d(x.permute(2,0,1).unsqueeze(0), self.weight, stride=self.stride, padding=self.padding).squeeze(0).permute(1,2,0)
    def backward(self, x):
        return F.conv_transpose2d(x.permute(2,0,1).unsqueeze(0), self.weight.transpose(0,1), stride=self.stride, padding=self.padding).squeeze(0).permute(1,2,0)

class Layer(Module):
    def __init__(self, shape):
        super().__init__()
        if not isinstance(shape,tuple): shape = (shape,)
        self.shape = shape
        self.activation = torch.zeros(shape)
        self.forward_input = torch.zeros(shape)
        self.backward_input = torch.zeros(shape)
        self.external_input = None
        self.connections = []
        self.back_connections = []

    def connectFrom(self, from_layer, connection_class, bidirectional=False, *args, **kwargs):
        """Create and attach a connection to another layer."""
        conn = connection_class(from_layer.shape, self.shape, *args, **kwargs)
        self.connections.append((from_layer, conn))
        if bidirectional:
            from_layer.back_connections.append((self, conn))
        return conn
    
    def updateActivation(self):
        input = self.forward_input + self.backward_input
        if self.external_input is not None: input += self.external_input

        feedforwardInhibition = input.mean().clamp_min(0.0) # inhibit it so that only some of them are active
        if len(self.shape) > 1:
            p = input.view(self.shape).mean(dim=1, keepdim=True).clamp_min(0.0)
            feedforwardInhibition = torch.maximum(feedforwardInhibition, p) # by using the maximum of column and layer inhibition, only the most active ones in the most active column are active
        input -= (feedforwardInhibition).expand(self.shape).flatten()

        self.activation = torch.tanh(input.clamp_min(0.0)*8.0)

    def forward(self, net):
        self.forward_input.zero_()
        for layer, conn in self.connections:
            self.forward_input += conn.forward(layer.activation)

        self.updateActivation()
        return self.activation
    
    def backward(self, net):
        self.backward_input.zero_()
        for layer, conn in self.back_connections:
            self.backward_input += conn.backward(layer.activation)
        
        self.updateActivation()
        return self.activation


class DeeppredLayers(Module):
    def __init__(self, shape, connectTo: list["DeeppredLayers"], connection_class, backward_connection_class, *args, **kwargs):
        super().__init__()
        self.layer = self.add(Layer(shape))
        self.ct = self.add(Layer(shape))
        self.p = self.add(Layer(shape))
        self.ct.connectFrom(self.layer, connection_class, *args, **kwargs)
        for lower in connectTo:
            self.layer.connectFrom(lower.layer, connection_class, bidirectional=True, *args, **kwargs)
            lower.p.connectFrom(self.ct, backward_connection_class, bidirectional=True, *args, **kwargs)
            self.layer.connectFrom(lower.p, connection_class, *args, **kwargs)
    def forward(self, net):
        self.layer.forward(net)
        self.p.forward(net) # P -> CT
        if net.plus_phase: self.p.activation = self.layer.activation.clone()
    def backward(self, net):
        self.ct.backward(net) # P <- CT
        self.p.backward(net)
        self.layer.backward(net)
    def onPlusPhaseEnd(self, net):
        self.ct.forward(net)


class Network(Module):
    def __init__(self):
        super().__init__()
        self.l1 = self.add(DeeppredLayers(10, [], FullConnection, FullConnection))
        self.l2 = self.add(DeeppredLayers(10, [self.l1], FullConnection, FullConnection))
        
    def step(self):
        self.l1.layer.external_input = tensor([0,1,0,0,0,0,0,0,0,0],dtype=torch.float)

        self.plus_phase = False
        self.forward(self)
        self.backward(self)
        self._onMinusPhaseEnd(self)
        self.plus_phase = True
        self.forward(self)
        self.backward(self)
        self._onPlusPhaseEnd(self)


net=Network()

fig, axs = plt.subplots(4, 4)
def plotAt(x,y, v, title):
	if isinstance(v,Layer): v=[v.activation]
	if len(axs[x,y].images): axs[x,y].images[0].set_data(v)
	else: axs[x,y].imshow(v, vmin=0,vmax=1)
	axs[x,y].set_title(title)
def plotThem():
    plotAt(0,0, net.l1.layer, "l1.layer")
    plotAt(0,1, net.l1.ct, "l1.ct")
    plotAt(0,2, net.l1.p, "l1.p")
    plotAt(1,0, net.l2.layer, "l2.layer")
    plotAt(1,1, net.l2.ct, "l2.ct")
    plotAt(1,2, net.l2.p, "l2.p")

for i in range(10):
    net.step()
    plotThem()
    plt.pause(2)
