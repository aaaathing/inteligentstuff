import torch
from torch import tensor
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


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
	w /= w.sum(dim=tuple(range(1,w.ndim)), keepdim=True) + 0.0001
	return w

class FullConnection(nn.Module):
    def __init__(self, in_shape, out_shape, initWeights=initWeightsSparse):
        super().__init__()
        if not isinstance(in_shape, tuple): in_shape = (in_shape,)
        if not isinstance(out_shape, tuple): out_shape = (out_shape,)
        self.weight = nn.Parameter(initWeights((flattenShape(out_shape), flattenShape(in_shape))), False)
        self.out_shape = out_shape
        self.in_shape = in_shape
        self.input = torch.zeros(in_shape)
    def forward(self, x):
        self.input = x
        return (x.flatten() @ self.weight.T).view(self.out_shape)
    def backward(self, x):
        return (x.flatten() @ self.weight).view(self.in_shape)
    def multEachW(self, sender_factor, receiver_factor):
        return (receiver_factor.flatten().unsqueeze(-1) @ sender_factor.flatten().unsqueeze(0)).view(self.weight.shape)

class ConvConnection(nn.Module):
    def __init__(self, in_shape, out_shape, kernel_size=3, stride=1, padding=1, initWeights=initWeightsSparse):
        super().__init__()
        self.weight = nn.Parameter(
            initWeights((out_shape[2], in_shape[2], kernel_size, kernel_size)),
            False)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input = torch.zeros(in_shape)
    def forward(self, x):
        self.input = x
        return F.conv2d(x.permute(2,0,1).unsqueeze(0), self.weight, stride=self.stride, padding=self.padding).squeeze(0).permute(1,2,0)
    def backward(self, x):
        return F.conv_transpose2d(x.permute(2,0,1).unsqueeze(0), self.weight, stride=self.stride, padding=self.padding).squeeze(0).permute(1,2,0)
    def multEachW(self, sender_factor, receiver_factor):
        grad = F.unfold(sender_factor.permute(2,0,1).unsqueeze(0), self.kernel_size, stride=self.stride, padding=self.padding)
        grad = grad * receiver_factor.permute(2,0,1).flatten(1)[:,None,:]
        grad = grad.mean(dim=2).view(receiver_factor.shape[2], sender_factor.shape[2], self.kernel_size, self.kernel_size)
        return grad

#x=[1.0,2.0];x0=[0,0];x1=[1.,2.]
#print(learn(tensor([[x1,x0,x0],[x0,x0,x0],[x0,x0,x0]]), tensor([[x,x0,x0],[x0,x0,x0],[x0,x0,x0]])))


class Layer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        if not isinstance(shape,tuple): shape = (shape,)
        self.shape = shape
        self.activation = torch.zeros(shape)
        self.forward_input = torch.zeros(shape)
        self.backward_input = torch.zeros(shape)
        self.minusPhaseActivation = torch.zeros(shape)
    
    def updateActivation(self):
        input = self.forward_input + self.backward_input*0.2

        feedforwardInhibition = ((input.mean()*0.7+input.amax()*0.3) - 0.05).clamp_min(0.0) # inhibit it so that only some of them are active
        if len(self.shape) > 1:
            p = ((input.mean(dim=-1, keepdim=True)*0.7+input.amax(dim=-1, keepdim=True)*0.3) - 0.05).clamp_min(0.0)
            feedforwardInhibition = torch.maximum(feedforwardInhibition, p) # by using the maximum of column and layer inhibition, only the most active ones in the most active column are active
        input -= feedforwardInhibition

        self.activation = torch.tanh(input.clamp_min(0.0)*8.0)

    def forward(self, forward_input, net):
        self.forward_input.zero_()
        for i in forward_input:
            self.forward_input += i
        self.updateActivation()
        return self.activation
    
    def backward(self, backward_input, net):
        self.backward_input.zero_()
        for i in backward_input:
            self.backward_input += i
        self.updateActivation()
        return self.activation
    
    def onMinusPhaseEnd(self, net):
        self.minusPhaseActivation = self.activation

#def hebblearn(layer, conn):#this test
#    if not hasattr(layer,"avg"):layer.avg=torch.zeros(layer.shape)
#    layer.avg+=(layer.activation-layer.avg)*0.2
#    conn.weight += (conn.multEachW(conn.input, layer.activation-layer.avg))*0.1
def boostActivity(self,net,conns):
    if not hasattr(self, "avgOutputActivity"):
        self.avgOutputActivity = torch.zeros(self.shape)
    target = self.avgOutputActivity.mean(dim=-1,keepdim=True) # for each pool
    self.avgOutputActivity += (self.activation - self.avgOutputActivity) * net.longAvgSpeed
    #actPct = self.avgOutputActivity / self.avgOutputActivity.sum(dim=-1,keepdim=True)
    for conn in conns:
        conn.weight += conn.multEachW(torch.ones_like(conn.input),((target - self.avgOutputActivity))) * conn.weight * net.lr


class DeeppredLayers(nn.Module):
    def __init__(self, shape, connectTo:list["DeeppredLayers"], connection_class, backward_connection_class, *args, **kwargs):
        super().__init__()
        self.layer = Layer(shape)
        self.ct = Layer(shape)
        self.p = Layer(shape)
        self.layer_connections = []
        self.layer_back_connections = []
        self.lower_p_to_layer_connections = []
        self.ct_connection = connection_class(shape, shape, *args, **kwargs)
        self.prev_ct_input = torch.zeros(shape)
        self.higher_ct_to_p_connections = []
        self.lower_p_back_to_ct_connections = []
        for lower in connectTo:
            layer_conn = connection_class(lower.layer.shape, self.layer.shape, *args, **kwargs)
            self.layer_connections.append(layer_conn)
            lower.layer_back_connections.append(layer_conn)
            self.lower_p_to_layer_connections.append(connection_class(lower.p.shape, self.layer.shape, *args, **kwargs))
            ct_to_p_conn = backward_connection_class(self.ct.shape, lower.p.shape, *args, **kwargs)
            lower.higher_ct_to_p_connections.append(ct_to_p_conn)
            self.lower_p_back_to_ct_connections.append(ct_to_p_conn)
    def ct_forward(self, net):
        self.ct.forward([self.ct_connection.forward(self.prev_ct_input)], net)
        return self.ct.activation
    def forward(self, lower, higher_ct, net, external_input=None):
        input = [conn.forward(lower[i].layer.activation) for i, conn in enumerate(self.layer_connections)]
        input += [conn.forward(lower[i].p.activation) for i, conn in enumerate(self.lower_p_to_layer_connections)]
        if external_input is not None: input.append(external_input)
        self.layer.forward(input, net)
        self.p.forward([conn.forward(higher_ct[i].ct.activation) for i, conn in enumerate(self.higher_ct_to_p_connections)], net) # P -> CT
        if net.plus_phase: self.p.activation = self.layer.activation.clone()
        #return self.layer.activation, self.p.activation
    def backward_p(self, net):
        self.p.backward([], net)
        if net.plus_phase: self.p.activation = self.layer.activation.clone()
        return self.p.activation
    def backward(self, higher_layer, lower_p, net):
        self.ct.backward([conn.backward(lower_p[i].ct.activation) for i, conn in enumerate(self.lower_p_back_to_ct_connections)], net) # P <- CT
        self.layer.backward([conn.backward(higher_layer[i].layer.activation) for i, conn in enumerate(self.layer_back_connections)], net)
        return self.layer.activation
    def onPlusPhaseEnd(self, net):
        for conn in self.layer_connections:
            conn.weight += conn.multEachW(conn.input, self.layer.activation-self.layer.minusPhaseActivation) * net.lr
        for conn in self.lower_p_to_layer_connections:
            conn.weight += conn.multEachW(conn.input, self.layer.activation-self.layer.minusPhaseActivation) * net.lr
        boostActivity(self.layer,net,self.layer_connections+self.lower_p_to_layer_connections)
        for conn in self.higher_ct_to_p_connections:
            conn.weight += conn.multEachW(conn.input, self.p.activation-self.p.minusPhaseActivation) * net.lr
        self.ct_connection.weight += self.ct_connection.multEachW(self.ct_connection.input, self.ct.activation-self.ct.minusPhaseActivation) * net.lr
        self.prev_ct_input = self.layer.activation.clone()

def addTrace(trace, x, decay):
    trace += (x - trace) * decay
class DecideModule(nn.Module):
  def __init__(self, shape, connectTo):
    super().__init__()
    assert len(shape)==2
    self.mtxGo = Layer(shape)
    self.mtxNoGo = Layer(shape)
    self.patchD1 = Layer(shape)
    self.patchD2 = Layer(shape)
    self.go_connection = [FullConnection(l.shape, shape) for l in connectTo]
    self.nogo_connection = [FullConnection(l.shape, shape) for l in connectTo]
    self.d1_connection = [FullConnection(l.shape, shape) for l in connectTo]
    self.d2_connection = [FullConnection(l.shape, shape) for l in connectTo]
    self.go_connection_trace = [torch.zeros(conn.weight.shape) for conn in self.go_connection]
    self.nogo_connection_trace = [torch.zeros(conn.weight.shape) for conn in self.nogo_connection]
    self.d1_connection_trace = [torch.zeros(conn.weight.shape) for conn in self.d1_connection]
    self.d2_connection_trace = [torch.zeros(conn.weight.shape) for conn in self.d2_connection]

  def forward(self, input, net):
    go_input = [conn.forward(input[i].activation) for i, conn in enumerate(self.go_connection)]
    self.mtxGo.forward(go_input, net)
    nogo_input = [conn.forward(input[i].activation) for i, conn in enumerate(self.nogo_connection)]
    self.mtxNoGo.forward(nogo_input, net)
    d1_input = [conn.forward(input[i].activation) for i, conn in enumerate(self.d1_connection)]
    self.patchD1.forward(d1_input, net)
    d2_input = [conn.forward(input[i].activation) for i, conn in enumerate(self.d2_connection)]
    self.patchD2.forward(d2_input, net)

    decision = (self.mtxGo.activation.mean(1, keepdim=True) - self.mtxNoGo.activation.mean(1, keepdim=True)).clamp_min(0.0)
    d1 = self.patchD1.activation.mean(1, keepdim=True)
    d2 = self.patchD2.activation.mean(1, keepdim=True)
    for i, (go_conn, nogo_conn, d1_conn, d2_conn) in enumerate(zip(self.go_connection, self.nogo_connection, self.d1_connection, self.d2_connection)):
        goOffTrace = go_conn.multEachW(input[i].activation, torch.where(decision<0.1, (d2 - d1) * self.mtxGo.activation, 0.0)) * 0.1
        nogoOffTrace = nogo_conn.multEachW(input[i].activation, torch.where(decision<0.1, (d2 - d1) * self.mtxNoGo.activation, 0.0)) * 0.1
        addTrace(self.go_connection_trace[i], go_conn.multEachW(input[i].activation, decision * ((1.0-d1)+d2) * self.mtxGo.activation) + goOffTrace, net.traceDecay)
        addTrace(self.nogo_connection_trace[i], nogo_conn.multEachW(input[i].activation, decision * ((1.0-d1)+d2) * self.mtxNoGo.activation) + nogoOffTrace, net.traceDecay)
        addTrace(self.d1_connection_trace[i], d1_conn.multEachW(input[i].activation, decision * self.patchD1.activation), net.traceDecay)
        addTrace(self.d2_connection_trace[i], d2_conn.multEachW(input[i].activation, decision * self.patchD2.activation), net.traceDecay)
    return decision
  def onPlusPhaseEnd(self, net):
    if net.reward:
      for i, (go_conn, nogo_conn, d1_conn, d2_conn) in enumerate(zip(self.go_connection, self.nogo_connection, self.d1_connection, self.d2_connection)):
        go_conn.weight += self.go_connection_trace[i] * net.reward * net.lr
        nogo_conn.weight -= self.nogo_connection_trace[i] * net.reward * net.lr
        d1_conn.weight += self.d1_connection_trace[i] * net.reward * net.lr
        d2_conn.weight -= self.d2_connection_trace[i] * net.reward * net.lr


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.v1 = DeeppredLayers((10,10,3), [], ConvConnection, ConvConnection)
        self.v2 = DeeppredLayers((10,10,3), [self.v1], ConvConnection, ConvConnection)
        self.v3 = DeeppredLayers((10,10,3), [self.v2], ConvConnection, ConvConnection)
        self.sem1 = DeeppredLayers((10,100), [self.v3], FullConnection, FullConnection)
        self.decide = DecideModule((5,36), [self.sem1.layer])
        
    def forward(self):
        self.v2.ct_forward(self)
        self.v1.forward([], [self.v2], self, external_input=self.input)
        self.v3.ct_forward(self)
        self.v2.forward([self.v1], [self.v3], self)
        self.sem1.ct_forward(self)
        self.v3.forward([self.v2], [self.sem1], self)
        self.sem1.forward([self.v3], [], self)
        self.decide.forward([self.sem1.layer], self)
    def backward(self):
        self.sem1.backward_p(self)
        self.v3.backward_p(self)
        self.sem1.backward([], [self.v3], self)
        self.v2.backward_p(self)
        self.v3.backward([self.sem1], [self.v2], self)
        self.v1.backward_p(self)
        self.v2.backward([self.v3], [self.v1], self)
        self.v1.backward([self.v2], [], self)

    input1=(torch.rand((10,10,3))>0.9).float()
    input2=(torch.rand((10,10,3))>0.9).float()
    def step(self):
        self.lr = 0.01
        self.traceDecay = 0.1
        self.reward = 1.0 if torch.rand(1).item() > 0.5 else 0.0
        self.longAvgSpeed = 0.05

        self.input = self.input1 if self.reward else self.input2

        self.plus_phase = False
        for i in range(4):
            self.forward()
            self.backward()
        for m in self.modules():
            if hasattr(m,"onMinusPhaseEnd"): m.onMinusPhaseEnd(self)
        self.plus_phase = True
        for i in range(4):
            self.forward()
            self.backward()
        for m in self.modules():
            if hasattr(m,"onPlusPhaseEnd"): m.onPlusPhaseEnd(self)


net=Network()

fig, axs = plt.subplots(4, 4)
def plotAt(x,y, v, title):
    if isinstance(v,Layer): v = v.activation
    if len(v.shape)==3: v=v.flatten(1)
    elif len(v.shape)==1: v=[v]
    
    if len(axs[x,y].images): axs[x,y].images[0].set_data(v)
    else: axs[x,y].imshow(v, vmin=0,vmax=1)
    axs[x,y].set_title(title)
def plotThem():
    axs[0,3].clear()
    axs[0,3].text(0.1,0.1, f"reward: {net.reward}")
    plotAt(0,0, net.input, "input")
    plotAt(1,0, net.v1.layer, "v1.layer")
    plotAt(1,1, net.v1.ct, "v1.ct")
    plotAt(1,2, net.v1.p.minusPhaseActivation, "v1.p minus")
    plotAt(2,0, net.v2.layer, "v2.layer")
    plotAt(2,1, net.v3.layer, "v3.layer")
    plotAt(2,2, net.sem1.layer, "sem1.layer")
    plotAt(3,0, net.decide.mtxGo, "decide Go")
    plotAt(3,1, net.decide.mtxNoGo, "decide NoGo")
    plotAt(3,2, net.decide.patchD1, "decide D1")
    plotAt(3,3, net.decide.patchD2, "decide D2")

for i in range(100):
    net.step()
    plotThem()
    plt.pause(0.1)
