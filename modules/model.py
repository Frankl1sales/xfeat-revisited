# Model functions implemented from paper and modified from original

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicLayer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
		super().__init__()
		self.layer = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias),
			nn.BatchNorm2d(out_channels, affine=False),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
	  return self.layer(x)

class XFeatModel(nn.Module):
	
	def __init__(self, ablation_type='skip_default'):
		"""
        ablation_type options:
          - 'skip_default'      			: Original architecture (1 skip connection to block4)
          - 'skip_none'         			: Ablation 1 - Removes the single skip connection
          - 'skip_input'  					: Ablation 2 - Adds skip connections from the input to EACH block
          - 'skip_resnet' 					: Ablation 3 - Adds ResNet-like residual skip connections to each block
		  - 'non_parallel_keypoints' 		: Ablation 4 - Replace the parallel keypoint detector with a block that operates on encoder feature map
        """
		super().__init__()
		self.ablation_type = ablation_type
		valid_ablations =['skip_default', 'skip_none', 'skip_input', 'skip_resnet', 'non_parallel_keypoints']
		if self.ablation_type not in valid_ablations:
			raise ValueError(f"ablation_type must be one of {valid_ablations}")

		self.norm = nn.InstanceNorm2d(1)

		# CNN Backbone & Heads
		self.skip1 = nn.Sequential(
			nn.AvgPool2d(4, stride=4),
			nn.Conv2d(1, 24, 1)
		)

		if self.ablation_type == 'skip_input':
			# Adds skip connections routing from input `x` to the inputs of each block
			self.in_skip2 = nn.Conv2d(1, 4, 1)
			self.in_skip3 = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(1, 8, 1))
			self.in_skip4 = nn.Sequential(nn.AvgPool2d(4), nn.Conv2d(1, 24, 1))
			self.in_skip5 = nn.Sequential(nn.AvgPool2d(8), nn.Conv2d(1, 64, 1))
			self.in_skip6 = nn.Sequential(nn.AvgPool2d(16), nn.Conv2d(1, 64, 1))

		if self.ablation_type == 'skip_resnet':
			# Adds residual connections around each block (from block input to block output)
			self.res_skip1 = nn.Conv2d(1, 4, 1, stride=1)
			self.res_skip2 = nn.Conv2d(4, 8, 1, stride=2)
			self.res_skip3 = nn.Conv2d(8, 24, 1, stride=2)
			self.res_skip4 = nn.Conv2d(24, 64, 1, stride=2)
			self.res_skip5 = nn.Conv2d(64, 64, 1, stride=2)
			self.res_skip6 = nn.Conv2d(64, 128, 1, stride=2)

		self.block1 = nn.Sequential(
			BasicLayer(1, 4, stride=1),
			BasicLayer(4, 4, stride=1),
		)

		self.block2 = nn.Sequential(
			BasicLayer(4, 8, stride=2),
			BasicLayer(8, 8, stride=1),
		)

		self.block3 = nn.Sequential(
			BasicLayer(8, 24, stride=2),
			BasicLayer(24, 24, stride=1),
			BasicLayer(24, 24, stride=1),
		)

		self.block4 = nn.Sequential(
			BasicLayer(24, 64, stride=2),
			BasicLayer(64, 64, stride=1),
			BasicLayer(64, 64, stride=1),
		)

		self.block5 = nn.Sequential(
			BasicLayer(64, 64, stride=2),
			BasicLayer(64, 64, stride=1),
			BasicLayer(64, 64, stride=1),
		)

		self.block6 = nn.Sequential(
			BasicLayer(64, 128, stride=2),
			BasicLayer(128, 128, stride=1),
			BasicLayer(128, 128, stride=1),
		)

		self.block6_proj = nn.Conv2d(128, 64, 1)

		self.fusion_block = nn.Sequential(
			BasicLayer(64, 64, stride=1),
			BasicLayer(64, 64, stride=1),
			nn.Conv2d(64, 64, 1)
		)

		self.reliability_head = nn.Sequential(
			BasicLayer(64, 64, stride=1),
			BasicLayer(64, 64, stride=1),
			nn.Conv2d(64, 1, 1),
			nn.Sigmoid()
		)

		self.keypoint_head = nn.Sequential(
			BasicLayer(64, 64),
			BasicLayer(64, 64),
			BasicLayer(64, 64),
			nn.Conv2d(64, 65, 1)
		)

		# Fine Matcher MLP
		self.fine_matcher =  nn.Sequential(
			nn.Linear(128, 512),
			nn.BatchNorm1d(512, affine=False),
			nn.ReLU(inplace = True),
			nn.Linear(512, 512),
			nn.BatchNorm1d(512, affine=False),
			nn.ReLU(inplace = True),
			nn.Linear(512, 512),
			nn.BatchNorm1d(512, affine=False),
			nn.ReLU(inplace = True),
			nn.Linear(512, 512),
			nn.BatchNorm1d(512, affine=False),
			nn.ReLU(inplace = True),
			nn.Linear(512, 64),
		)

	def _unfold2d(self, x, ws = 2):
		B, C, H, W = x.shape
		x = x.unfold(2,  ws , ws).unfold(3, ws,ws)   \
			.reshape(B, C, H//ws, W//ws, ws**2)
		return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)


	def forward(self, x):
		# Input normalisation
		with torch.no_grad():
			x = x.mean(dim=1, keepdim = True)
			x = self.norm(x)

		# Main backbone modified by ablation type  
		if self.ablation_type == 'skip_none':
			x1 = self.block1(x)
			x2 = self.block2(x1)
			x3 = self.block3(x2)
			x4 = self.block4(x3)
			x5 = self.block5(x4)
			x6 = self.block6(x5)
            
		elif self.ablation_type == 'skip_input':
			x1 = self.block1(x)
			x2 = self.block2(x1 + self.in_skip2(x))
			x3 = self.block3(x2 + self.in_skip3(x))
			x4 = self.block4(x3 + self.in_skip4(x))
			x5 = self.block5(x4 + self.in_skip5(x))
			x6 = self.block6(x5 + self.in_skip6(x))
            
		elif self.ablation_type == 'skip_resnet':
			x1 = self.block1(x) + self.res_skip1(x)
			x2 = self.block2(x1) + self.res_skip2(x1)
			x3 = self.block3(x2) + self.res_skip3(x2)
			x4 = self.block4(x3) + self.res_skip4(x3)
			x5 = self.block5(x4) + self.res_skip5(x4)
			x6 = self.block6(x5) + self.res_skip6(x5)
		else:
			x1 = self.block1(x)
			x2 = self.block2(x1)
			x3 = self.block3(x2)
			x4 = self.block4(x3 + self.skip1(x))
			x5 = self.block5(x4)
			x6 = self.block6(x5)
			
		# Pyramid fusion: merge {1/8, 1/16, 1/32} at 1/8 resolution
		x5_up = F.interpolate(x5, (x4.shape[-2], x4.shape[-1]), mode='bilinear')
		x6_proj = self.block6_proj(x6)  # 128 → 64 channels
		x6_up = F.interpolate(x6_proj, (x4.shape[-2], x4.shape[-1]), mode='bilinear')
		feats = self.fusion_block(x4 + x5_up + x6_up)

		# Reliability map
		heatmap = self.reliability_head(feats)  
		#Keypoint map logits
		if self.ablation_type != 'non_parallel_keypoints':
			keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) 
		else:
			keypoints = self.keypoint_head(feats)

		# feats: torch.Tensor(B, 64, H/8, W/8)
		# keypoints: torch.Tensor(B, 65, H/8, W/8)
		# heatmap: torch.Tensor(B, 1, H/8, W/8)
		return feats, keypoints, heatmap
