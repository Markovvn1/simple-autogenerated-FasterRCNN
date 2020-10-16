import os
import sys
import shutil

# TODO: add option freeze_at
"""
Model:
	image size: (height, width)
	output already sorted
"""

def build_model(cfg, mode, output_dir, engine="pytorch"):
	assert mode in ["test", "train"]
	assert cfg["name"] in ["FASTER_RCNN"]

	input_point = {"FASTER_RCNN": "model.FasterRCNN"}[cfg["name"]]

	builder = Builder({"mode": mode}, engine)
	builder.add_module(input_point, cfg[cfg["name"]])
	builder.build(output_dir)

	with open(os.path.join(output_dir, "config.yaml"), "w") as f:
		f.write(_yaml_dump(cfg))


class Builder:

	"""
	builders (list): list of all module builders
	dependencies (dict[str, func]): map name of module to function 'use' of this module
	"""

	def __init__(self, global_params, engine):
		assert engine == "pytorch"

		self._global_params = global_params

		def find(d=None):
			res = []
			for i in os.listdir(d):
				if d is not None: i = os.path.join(d, i)
				if os.path.isfile(i):
					if i.endswith(".py"):
						res.append(i)
				else:
					res.extend(find(i))
			return res

		pkg_path = os.path.join("pkg", engine)

		last_dir = os.getcwd()
		last_sys_path = sys.path.copy()

		os.chdir(os.path.join(os.path.dirname(__file__), pkg_path))
		ls = find()
		os.chdir("../..")
		sys.path.append(os.getcwd())

		self.builders = {}
		self.builders_per_folder = {}
		for i in ls:
			path = os.path.join(pkg_path, i[:-3]).replace("/", ".")
			exec(f"from {path} import ModuleBuilder; self.builders[\"{i}\"] = ModuleBuilder()")
			dirname = os.path.dirname(i)
			if dirname not in self.builders_per_folder: self.builders_per_folder[dirname] = []
			self.builders_per_folder[dirname].append(self.builders[i])
		self.dependencies = {j: i.use for i in self.builders.values() for j in i.keys}

		sys.path = last_sys_path
		os.chdir(last_dir)

	def add_module(self, main_class, cfg):
		"""Добавить построение модуля main_class с параметрами cfg."""
		assert isinstance(main_class, str)

		def build_graph(t, cfg):
			assert t in self.dependencies, f"implementation for module {t} not found"
			for args in self.dependencies[t](t, self._global_params, cfg).items():
				build_graph(*args)

		build_graph(main_class, cfg)  # building dependencies graph

	def build(self, output_dir):
		"""Построить все добавленные модули."""
		assert output_dir

		shutil.rmtree(output_dir, ignore_errors=True)  # clean build directory
		os.makedirs(output_dir)
		last_dir = os.getcwd()  # save old dir
		os.chdir(output_dir)  # go to build dir

		# build all modules
		for file_name, item in self.builders.items():
			item.build(file_name, self._global_params)

		# create __init__.py files
		for dir_name, items in self.builders_per_folder.items():
			content = [i.init_file() for i in items]
			content = [i for i in content if i]
			if not content: continue
			with open(os.path.join(dir_name, "__init__.py"), "w") as f:
				f.write("\n".join(content) + "\n")

		os.chdir(last_dir)  # go to saved dir


def _yaml_dump(x, tab=""):
	if not isinstance(x, dict): return tab + str(x)
	res = []
	ntab = tab + "  "
	for k, v in x.items():
			if isinstance(v, dict):
					res.append(tab + str(k) + ":\n" + _yaml_dump(v, ntab) + "\n")
			else:
					res.append(tab + str(k) + ": " + (f"\"{v}\"" if isinstance(v, str) else str(v)) + "\n")
	return "".join(res)[:-1]
