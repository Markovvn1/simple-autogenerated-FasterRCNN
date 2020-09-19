import torch
import torch.nn as nn
from fvcore.nn import weight_init


class Conv2d(nn.Conv2d):

	def __init__(self, in_channels, out_channels, **kwargs):
		norm_cls = kwargs.pop("norm_cls", None)
		init = kwargs.pop("init", None)
		super().__init__(in_channels, out_channels, **kwargs)
		self.norm_cls = norm_cls
		self.norm = norm_cls(out_channels) if norm_cls is not None else lambda x: x

		if init is not None:
			weight_init.__getattribute__(init)(self)

		self.conv_args = (in_channels, out_channels)
		self.conv_kwargs = kwargs

	def forward(self, x):
		return self.norm(super().forward(x))

	def extract(self):
		"""Объединить conv и batchnorm в слой nn.Conv2d."""

		res = nn.Conv2d(*self.conv_args, **self.conv_kwargs)
		res.weight = self.weight
		res.bias = self.bias

		for p in res.parameters():
			p.requires_grad = False

		if self.norm_cls is None: return res

		res_scale = self.norm.weight * (self.norm.running_var + self.norm.eps).rsqrt()
		res_bias = self.norm.bias - self.norm.running_mean * res_scale

		res.weight.data *= res_scale.reshape(-1, 1, 1, 1)
		if res.bias is not None:
			res.bias.data *= res_scale
			res.bias.data += res_bias
		else:
			res.bias = torch.nn.Parameter(res_bias, requires_grad=False)

		return res

	@classmethod
	def extract_all(cls, module):
		if isinstance(module, Conv2d):
			return cls.extract(module)

		res = module
		for name, child in module.named_children():
			new_child = cls.extract_all(child)
			if new_child is not child:
				res.add_module(name, new_child)

		return res