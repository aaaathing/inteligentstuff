#%%
import torch
from torch import tensor
import torch.nn.functional as F
import sys
#sys.path.append("..")
#sys.path.append("../AbSViT")
sys.path.append("AbSViT")
import AbSViT.models.absvit

device=torch.device("cpu")

# %%

model = 'absvit_tiny_patch16_224'
resume = 'https://berkeley.box.com/shared/static/7415yz4d1l5z0ur6x32k35f8y99zgynq.pth'

model = AbSViT.models.absvit.absvit_tiny_patch8_224_gap()
model.to(device)
model.eval()
checkpoint = torch.hub.load_state_dict_from_url(resume, map_location=device)
model.load_state_dict(checkpoint['model'])

#%%
if False:
	# %%
	import urllib.request
	url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
	labels = urllib.request.urlopen(url).readlines()

	import torchvision.transforms.v2 as transforms
	from PIL import Image
	from pathlib import Path
	def getImage(path):
		transform = transforms.Compose([ transforms.Resize((224, 224)), transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True) ])
		return transform(Image.open(Path.home()/path).convert('RGB')).unsqueeze(0)  # Shape: [1, 3, 224, 224]

	input = getImage("Downloads/Screenshot 2025-04-10 10.50.49 AM.jpg")
	input2 = getImage("Downloads/b.jpg")

# %%
#def forward(input, channel_attention, spatial_attention):
#	x, _, out_var = model.forward_features(input)
#	for c in torch.linalg.vector_norm(x[0],dim=0).topk(10).indices:
#		print(labels[c])
#
#	cos_sim = F.normalize(x, dim=-1) @ F.normalize(model.prompt[..., None], dim=1)
#	mask = cos_sim.clamp(0, 1)
#	x = x * mask
#	top_down_transform = model.prompt[..., None] @ model.prompt[..., None].transpose(-1, -2)
#	x = x @ top_down_transform * 5
#	td = model.feedback(x)
#
#	att = out_var[-1-3].norm(dim=-1)[:, model.num_prefix_tokens:]
#	L = att.shape[-1]
#	att = att[0].view(int(L**0.5), int(L**0.5))
#	att = (att - (0.7*att.max() + 0.3*att.min())).clamp(0)
#	fig, axs = plt.subplots(1,3)
#	axs[0].imshow(att.detach().cpu().numpy())
#
#	mask_vis = mask[0, model.num_prefix_tokens:].squeeze().view(int(L**0.5), int(L**0.5))
#	mask_vis = (mask_vis - (0.7*mask_vis.max() + 0.3*mask_vis.min())).clamp(0)
#	axs[1].imshow(mask_vis.detach().cpu().numpy())
#
#	x, _, out_var = model.forward_features(input, td)
#	print("\nwith attention")
#	for c in torch.linalg.vector_norm(x[0],dim=0).topk(10).indices:
#		print(labels[c])
#    
#	att = out_var[-1-3].norm(dim=-1)[:, model.num_prefix_tokens:]
#	L = att.shape[-1]
#	att = att[0].view(int(L**0.5), int(L**0.5))
#	att = (att - (0.7*att.max() + 0.3*att.min())).clamp(0)
#	axs[2].imshow(att.detach().cpu().numpy())


# %%
#model.prompt.requires_grad_(False).zero_()
#model.prompt[20:30]=1
#forward(input)
# %%

def rescale(x):
	min = x.min()
	max = x.max()
	return x.clamp(0)/max if max>0 else x.clamp(0)
	#if min == max:
	#	max = max+1
	#return (x - min) / (max - min)

model.prevX = None

@torch.no_grad
def run(input, channel_attention=None, spatial_attention=None):
	if len(input) == 3:
		input = input[None,...]
	
	td = None
	if model.prevX is not None and channel_attention is not None:
		spatial_attention = torch.cat((torch.ones(model.num_prefix_tokens), spatial_attention))
		x = model.prevX * (channel_attention[None,None,:] + spatial_attention[None,:,None]) + 0.2
		x /= x.max()
		td = model.feedback(x)

	x, _, _ = model.forward_features(input, td)
	model.prevX = x
	
	ca = torch.mean(x[0], dim=0)
	ca[149] = 0
	ca[181] = 0
	sa = torch.mean(x[0,model.num_prefix_tokens:], dim=1)
	return rescale(ca), rescale(sa)

# %%
if False:
	#%%
	import matplotlib.pyplot as plt
	(ca,sa) = run(input,ca,sa)
	#ca=torch.tanh(ca)
	#sa=torch.tanh(sa)
	fig, axs = plt.subplots(2)
	axs[0].imshow(sa.view(28,28).detach(), vmin=0, vmax=1)
	axs[1].imshow(ca.view(12,16).detach(), vmin=0, vmax=1)
# %%
