import os

class ModuleBuilderBase:

	def __init__(self, all_modules):
		self.keys = all_modules
		self._modules = {}
		self._childs = {}

	def use(self, module, global_params, cfg):
		assert module in self.keys
		if module in self._modules:
			assert self._modules[module] == cfg, "notImplementedError"
			return {}
		self._modules[module] = cfg
		self._assert_cfg(module, cfg)
		res = self._dependencies(module, global_params, cfg)
		assert isinstance(res, dict)
		self._childs.update(res)
		return res

	def build(self, file_name, global_params):
		if not self._modules: return

		os.makedirs(os.path.dirname(file_name), exist_ok=True)
		with open(file_name, "w") as f:
			f.write(self._generate(global_params, self._modules, self._childs))

	def init_file(self):
		if not self._modules: return None
		return self._init_file(self._modules)
