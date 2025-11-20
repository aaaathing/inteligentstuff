import torch
from torch import tensor
import torch.nn as nn

class A(nn.Module):
	def __init__(self):
		super().__init__()
		self.l=nn.ModuleList()
		self.l2=self.l
		self.x=tensor([])
		self.xs:list[torch.Tensor]=[]
	def c(self,a):
		self.l.append(a)
		self.xs.append(tensor([]))
	def forward(self,x):
		for i, m in enumerate(self.l):
			x=m(x)
			self.xs[i]=x
		self.x=x
		return x
	def dostuff(self):
		self.x.fill_(123)
	@torch.jit.export
	def testsame(self):
		with torch.no_grad():
			self.l[0].weight.fill_(1234)
		self.xs.append(tensor([123,321,10]))
		self.xs = [x+1 for x in self.xs]
		for i in self.l:
			if hasattr(i,"in_features"):
				self.xs.append(tensor([66,i.in_features]))
		return self.l2[0].weight
	@torch.jit.export
	def testarg(self, o:"N"):
		return o.idx
def geta(a:nn.ModuleList):
	for i in a:
		return i.weight
	
class N(nn.Module):
	def __init__(self):
		super().__init__()
		self.a=A()
		self.a.c(nn.Linear(10,10))
		self.a.c(nn.ReLU())
		self.a2=A()
		self.a2.c(nn.Linear(10,10))
		self.idx=0
	def forward(self,x):
		#for i in self.a.l:x=i(x)
		#x=self.a.l[self.idx](x)
		self.idx+=1
		return self.a(x)
	@torch.jit.export
	def what(self):
		for i in self.children(): i.dostuff()
	@torch.jit.export
	def geta(self):return geta(self.a.l)

a=torch.jit.script(N())
print(a(torch.rand(10)))
print(a.a.x)
a.what()
print(a.a.x)
print(a.a.xs)
print(a.a.testsame())
print(a.a.xs)
print(a.a.testarg(a))
print(a.geta())
