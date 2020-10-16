import os
import sys
import shutil

# TODO: add option freeze_at
"""
Model:
	image size: (height, width)
	output already sorted
"""

class Builder:

	"""
	builders (list): list of all module builders
	dependencies (dict[str, func]): map name of module to function 'use' of this module
	"""

	def __init__(self, engine, output_dir):
		assert engine == "pytorch"
		self._output_dir = output_dir

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
			

	def build(self, main_class, global_params, cfg):
		assert isinstance(main_class, str)

		def build_graph(t, cfg):
			assert t in self.dependencies, f"implementation for module {t} not found"
			for args in self.dependencies[t](t, global_params, cfg).items():
				build_graph(*args)

		build_graph(main_class, cfg)  # building dependencies graph

		last_dir = os.getcwd()  # save old dir
		shutil.rmtree(self._output_dir, ignore_errors=True)  # clean build directory
		os.makedirs(self._output_dir)
		os.chdir(self._output_dir)  # go to build dir

		# build all modules
		for file_name, item in self.builders.items():
			item.build(file_name, global_params)

		# create __init__.py files
		for dir_name, items in self.builders_per_folder.items():
			content = [i.init_file() for i in items]
			content = [i for i in content if i]
			if not content: continue
			with open(os.path.join(dir_name, "__init__.py"), "w") as f:
				f.write("\n".join(content) + "\n")

		os.chdir(last_dir)  # go to saved dir