#!/home/vova/anaconda3/bin/python3
import os
import sys
import json
import random
import shutil

from builder import build_model  # Переместите ../builder в web

def get_input():
	res = json.loads("".join(sys.argv[1:]))

	def norm_element(s):
		if not isinstance(s, str): return s
		try:
			return int(s)
		except ValueError:
			try:
				return float(s)
			except ValueError as e:
				return s

	def walk(t):
		for k, u in (enumerate(t) if isinstance(t, list) else t.items()):
			if isinstance(u, (dict, list)):
				walk(u)
			else:
				t[k] = norm_element(u)

	walk(res)
	return res


def choose_dir(temp_dir):
	while True:
		dir_name = os.path.join(temp_dir, str(random.randint(0, 999999)))
		try:
			os.makedirs(dir_name, exist_ok=True)
			return dir_name
		except PermissionError as e:
			raise PermissionError(13, "Failed to create 'workdir' folder")


os.chdir(os.path.dirname(__file__))  # go to home dir

base_folder = None

try:
	cfg = get_input()
	mode = cfg.pop("mode")
	engine = cfg.pop("engine")
	model_type = cfg.pop("_model_type").upper()
	cfg = {"name": model_type, model_type: cfg}

	base_folder = choose_dir("workdir/res")
	output_dir = os.path.join(base_folder, "model")
	base_folder = os.path.abspath(base_folder)

	# build net
	build_model(cfg, mode, output_dir, engine)

	print(base_folder, end="")

except Exception as e:
	if base_folder is not None:
		shutil.rmtree(base_folder, ignore_errors=True)

	tb = e.__traceback__
	while tb.tb_next is not None: tb = tb.tb_next

	fname = os.path.split(tb.tb_frame.f_code.co_filename)
	fname = os.path.join(os.path.basename(fname[0]), fname[1])
	# name = tb.tb_frame.f_code.co_name  # function name
	print(f"{fname}, line {tb.tb_lineno}: {repr(e)}")