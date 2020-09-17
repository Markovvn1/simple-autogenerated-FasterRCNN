import os
import sys
import shutil
import yaml
import collections

# TODO:
"""
Model:
	image size: (height, width)
	output already sorted
"""

from builders import build_resnet, build_fpn
from builders import build_rpn
from builders import build_model
from builders import build_fast_rcnn
from builders import build_pooler

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

def safe_clone(name, dependencies=None, test_only=None):
	os.makedirs(os.path.dirname(name), exist_ok=True)
	shutil.copyfile("../builders/libs_pytorch/"+name, name)
	return dependencies

libs_maps = {
	"model.py": (build_model, cfg["MODEL"]),
	"parts/resnet.py": (build_resnet, cfg["MODEL"]["BACKBONE"]["RESNETS"]),
	"parts/fpn.py": (build_fpn, cfg["MODEL"]["BACKBONE"]["FPN"]),
	"parts/rpn.py": (build_rpn, cfg["MODEL"]["PROPOSAL_GENERATOR"]["RPN"]),
	"parts/fast_rcnn.py": (build_fast_rcnn, cfg["MODEL"]["ROI_HEAD"]["StandardROIHeads"]),
	"parts/backbone.py": (safe_clone, "parts/backbone.py"),
	"layers/conv_wrapper.py": (safe_clone, "layers/conv_wrapper.py"),
	"layers/freeze_batchnorm.py": (safe_clone, "layers/freeze_batchnorm.py"),
	"layers/pooler.py": (build_pooler, cfg["MODEL"]["ROI_HEAD"]["StandardROIHeads"]["POOLER"]),
	"utils/boxes.py": (safe_clone, "utils/boxes.py"),
	"utils/anchors.py": (safe_clone, "utils/anchors.py"),
	"utils/matcher.py": (safe_clone, "utils/matcher.py", ["utils/pairwise_iou.py"]),
	"utils/pairwise_iou.py": (safe_clone, "utils/pairwise_iou.py"),
	"utils/subsampler.py": (safe_clone, "utils/subsampler.py"),
	"utils/box_transform.py": (safe_clone, "utils/box_transform.py"),
}

import_maps = {
	"parts/resnet.py": "from .resnet import ResNet",
	"parts/fpn.py": "from .fpn import FPN",
	"parts/rpn.py": "from .rpn import RPN",
	"parts/fast_rcnn.py": "from .fast_rcnn import FastRCNNHead",
	"parts/backbone.py": "from .backbone import Backbone",
	"layers/conv_wrapper.py": "from .conv_wrapper import Conv2d",
	"layers/freeze_batchnorm.py": "from .freeze_batchnorm import ModuleWithFreeze, FrozenBatchNorm2d",
	"layers/pooler.py": "from .pooler import RoIPooler",
	"utils/boxes.py": "from . import boxes as Boxes",
	"utils/anchors.py": "from .anchors import Anchors, MultiAnchors",
	"utils/matcher.py": "from .matcher import Matcher",
	"utils/pairwise_iou.py": "from .pairwise_iou import pairwise_iou",
	"utils/subsampler.py": "from .subsampler import Subsampler",
	"utils/box_transform.py": "from .box_transform import BoxTransform",
}

def create_with_dependencies(lib):
	if libs_maps[lib] is None: return
	func = libs_maps[lib]; libs_maps[lib] = None
	print(f"Generate {lib}")
	res = func[0](*func[1:], test_only=TEST_ONLY)
	if res is None: return
	for item in res:
		create_with_dependencies(item)

create_with_dependencies("model.py")

init_imports = collections.defaultdict(list)
for k in import_maps:
	if libs_maps[k] is not None: continue
	import_file = os.path.basename(k)
	assert import_file.endswith(".py")
	init_imports[os.path.dirname(k)].append(import_maps[k])

for path, v in init_imports.items():
	print(f"Generate {path}/__init__.py")
	with open(path+"/__init__.py", "w") as f:
		f.write("###  Automatically-generated file  ###\n\n")
		f.write("\n".join(v)+"\n")

print("Done.")