import os
import sys
import shutil
import yaml

from builders import build_resnet, build_backbone, build_fpn


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

assert cfg["MODEL"]["backbone"] in ["resnet_fpn"]




shutil.rmtree('build', ignore_errors=True)  # clean build directory
os.makedirs("build")
os.chdir("build")

libs = set()
if cfg["MODEL"]["backbone"] == "resnet_fpn":
	libs.update(build_backbone())
	libs.update(build_resnet(cfg["MODEL"]["RESNETS"], test_only=TEST_ONLY, lib_prefix=".libs."))
	libs.update(build_fpn(cfg["MODEL"]["FPN"], test_only=TEST_ONLY, lib_prefix=".libs."))

if len(libs) != 0:
	os.makedirs("libs", exist_ok=True)

	for item in libs:
		shutil.copyfile("../libs/"+item, "libs/"+item)

print("Done.")