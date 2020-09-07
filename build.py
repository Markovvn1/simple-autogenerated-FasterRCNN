import os
import sys
import shutil
import yaml
import collections

from builders import build_resnet, build_fpn
from builders import build_rpn
from builders import build_model

def unknown_value(value):
	print(f"Неизвестное значение {value}\n")
	show_help_and_exit()

def show_help_and_exit(short=True):
	print(f"Использование: python3 {sys.argv[0]} MODE PATH_TO_CONFIG")
	
	if short:
		print("Для получения более подробной справки используйте -h")
	else:
		print("""
MODE:            В каком режиме будет использоваться сеть (train/test)
PATH_TO_CONFIG:  Путь до файла с конфигурацией сети""")
	exit()


if len(sys.argv) == 1: show_help_and_exit()
if sys.argv[1] == "-h" or sys.argv[1] == "--help": show_help_and_exit(False)
if len(sys.argv) != 3: show_help_and_exit()
if sys.argv[1] not in ["train", "test"]: unknown_value(sys.argv[1])
TEST_ONLY = sys.argv[1] == "test"

if not os.path.isfile(sys.argv[2]):
	print(f"Файл {sys.argv[2]} не существует")
	exit()

with open(sys.argv[2], "r") as f:
	cfg = yaml.safe_load(f)




shutil.rmtree('build', ignore_errors=True)  # clean build directory
os.makedirs("build")
os.chdir("build")

def safe_clone(name, test_only=None):
	os.makedirs(os.path.dirname(name), exist_ok=True)
	shutil.copyfile("../builders/libs_pytorch/"+name, name)

libs_maps = {
	"model.py": (build_model, cfg["MODEL"]),
	"parts/resnet.py": (build_resnet, cfg["MODEL"]["BACKBONE"]["RESNETS"]),
	"parts/fpn.py": (build_fpn, cfg["MODEL"]["BACKBONE"]["FPN"]),
	"parts/rpn.py": (build_rpn, cfg["MODEL"]["PROPOSAL_GENERATOR"]["RPN"]),
	"parts/backbone.py": (safe_clone, "parts/backbone.py"),
	"layers/conv_wrapper.py": (safe_clone, "layers/conv_wrapper.py"),
	"layers/freeze_batchnorm.py": (safe_clone, "layers/freeze_batchnorm.py"),
}

import_maps = {
	"parts/resnet.py": ["ResNet"],
	"parts/fpn.py": ["FPN"],
	"parts/rpn.py": ["RPN"],
	"parts/backbone.py": ["Backbone"],
	"layers/conv_wrapper.py": ["Conv2d"],
	"layers/freeze_batchnorm.py": ["ModuleWithFreeze", "FrozenBatchNorm2d"],
}

def create_with_dependencies(lib):
	if libs_maps[lib] is None: return
	func = libs_maps[lib]; libs_maps[lib] = None
	print(f"Generate {lib}")
	res = func[0](func[1], test_only=TEST_ONLY)
	if res is None: return
	for item in res:
		create_with_dependencies(item)

create_with_dependencies("model.py")

init_imports = collections.defaultdict(list)
for k in import_maps:
	if libs_maps[k] is not None: continue
	import_file = os.path.basename(k)
	assert import_file.endswith(".py")
	init_imports[os.path.dirname(k)].append(f"from .{import_file[:-3]} import " + ", ".join(import_maps[k]))

for path, v in init_imports.items():
	print(f"Generate {path}/__init__.py")
	with open(path+"/__init__.py", "w") as f:
		f.write("###  Automatically-generated file  ###\n\n")
		f.write("\n".join(v)+"\n")

print("Done.")