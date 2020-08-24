import torch
import torch.nn as nn

class Conv2d(nn.Conv2d):

	def __init__(self, *args, **kwargs):
		norm = kwargs.pop("norm", None)
		assert len(args) <= 3
		super().__init__(*args, **kwargs)
		self.norm = norm

		self.conv_args = args
		self.conv_kwargs = kwargs

	def forward(self, x):
		x = super().forward(x)
		if self.norm is not None:
			x = self.norm(x)
		return x

	def extract(self):
		"""Объединить conv и batchnorm в слой nn.Conv2d."""

		res = nn.Conv2d(*self.conv_args, **self.conv_kwargs)
		res.weight = self.weight
		res.bias = self.bias

		for p in res.parameters():
			p.requires_grad = False

		if self.norm is None: return res

		if isinstance(self.norm, nn.BatchNorm2d):
			res_scale = self.norm.weight * (self.norm.running_var + self.norm.eps).rsqrt()
			res_bias = self.norm.bias - self.norm.running_mean * res_scale

			res.weight.data *= res_scale.reshape(-1, 1, 1, 1)
			if res.bias is not None:
				res.bias.data *= res_scale
				res.bias.data += res_bias
			else:
				res.bias = torch.nn.Parameter(res_bias, requires_grad=False)
		else:
			raise NotImplementedError()

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