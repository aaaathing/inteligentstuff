# it is so frustrating to make it torchscipt compatible (which required explicit connections) and have seperate forward and backward pass and still have different connection types and learning rules


v1=...
v1_to_v2=ConvConnection(3)
v2_back_to_v1=ConvBackConnection(3)
v2_to_v1_p=ConvConnection(3)
v2=...
v2_to_v3=ConvConnection(3)
v3_back_to_v2=ConvBackConnection(3)
v3_to_v2_p=ConvConnection(3)
v3=...
v3_to_v4=ConvConnection(3)
v4_back_to_v3=ConvBackConnection(3)
v4_to_v3_p=ConvConnection(3)
v4=...

def forward(x):
	x = v1.forward(x)
	x = v2.forward(v1_to_v2())
	v1_p.forward(v2_to_v1_p(x))
	x = v3.forward(v2_to_v3())
	v2_p.forward(v3_to_v2_p(x))
	x = v4.forward(v3_to_v4())
	v3_p.forward(v4_to_v3_p(x))
def backward():
	x = v4.backward()
	x = v3.backward(v4_back_to_v3())
	x = v2.backward(v3_back_to_v2())
	x = v1.backward(v2_back_to_v1())
