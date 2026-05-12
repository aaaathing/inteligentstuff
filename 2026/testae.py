import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import av, io
from datasets import load_dataset

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
video_data = VideoDataset((3,16,16))


fig, axs = plt.subplots(2, 2)
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


e1=nn.Conv2d(3,6,9,padding=4)
d1=nn.Conv2d(6,3,9,padding=4)
for i in range(1000):
	v=video_data.next().reshape(3,16,16)*2-1
	a=e1(v+torch.rand((3,16,16))*0.1).reshape(6,16,16)
	a=F.relu(a)
	b=d1(a).reshape(3,16,16)
	F.mse_loss(b,v).backward()
	with torch.no_grad():
		for p in e1.parameters(): p-=p.grad*0.0001
		for p in d1.parameters(): p-=p.grad*0.0001
	showAt(axs[0,0],v,'v')
	showAt(axs[0,1],a,'a')
	showAt(axs[1,0],b,'b')
	plt.pause(0.25)