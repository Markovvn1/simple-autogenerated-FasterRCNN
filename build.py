import os
import sys
import yaml
from builder import build_model

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
mode = sys.argv[1]

if not os.path.isfile(sys.argv[2]):
	print(f"Файл {sys.argv[2]} не существует")
	exit()

with open(sys.argv[2], "r") as f:
	cfg = yaml.safe_load(f)

build_model(cfg, mode, output_dir="build")