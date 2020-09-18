import torch
import torch.nn as nn

class ModuleWithFreeze:

	def freeze(self):
		"""Freeze this module and return it."""

		for p in self.parameters():
			p.requires_grad = False

		return FrozenBatchNorm2d.freeze_all_batchnorms(self)


class FrozenBatchNorm2d(nn.Module):

	_version = 3

	def __init__(self, num_features, eps=1e-5):
		super().__init__()
		self.num_features = num_features
		self.eps = eps
		self.register_buffer("weight", torch.ones(num_features))
		self.register_buffer("bias", torch.zeros(num_features))
		self.register_buffer("running_mean", torch.zeros(num_features))
		self.register_buffer("running_var", torch.ones(num_features) - eps)

		self._update_params()

	def _update_params(self):
		self.res_scale = self.weight * (self.running_var + self.eps).rsqrt()
		self.res_bias = self.bias - self.running_mean * self.res_scale

		self.res_scale = self.res_scale.reshape(1, -1, 1, 1)
		self.res_bias = self.res_bias.reshape(1, -1, 1, 1)

	def forward(self, x):
		if x.device != self.res_scale.device: self._update_params()
		return x * self.res_scale + self.res_bias

	def _load_from_state_dict(
		self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
	):
		version = local_metadata.get("version", None)

		if version is None or version < 2:
			# No running_mean/var in early versions
			# This will silent the warnings
			if prefix + "running_mean" not in state_dict:
				state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
			if prefix + "running_var" not in state_dict:
				state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

		if version is not None and version < 3:
			# In version < 3, running_var are used without +eps.
			state_dict[prefix + "running_var"] -= self.eps

		super()._load_from_state_dict(
			state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
		)

		self._update_params()

	def __repr__(self):
		return f"FrozenBatchNorm2d(num_features={self.num_features}, eps={self.eps})"

	@classmethod
	def from_batchnorm(cls, module):
		"""
		Convert BatchNorm/SyncBatchNorm into FrozenBatchNorm.

		Args:
			module (nn.BatchNorm2d or nn.SyncBatchNorm)

		Returns:
			A frozen version of module
		"""

		assert isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm))
	 
		res = cls(module.num_features, module.eps)
		if module.affine:
			res.weight.data = module.weight.data.clone().detach()
			res.bias.data = module.bias.data.clone().detach()
		res.running_mean.data = module.running_mean.data
		res.running_var.data = module.running_var.data

		res._update_params()

		return res

	@classmethod
	def freeze_all_batchnorms(cls, module):
		if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
			return cls.from_batchnorm(module)
		
		res = module
		for name, child in module.named_children():
			new_child = cls.freeze_all_batchnorms(child)
			if new_child is not child:
				res.add_module(name, new_child)

		return res