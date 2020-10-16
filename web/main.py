#!/home/vova/anaconda3/bin/python3
import os
import sys
import json
import random

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


def yaml_dump(x, tab=""):
	if not isinstance(x, dict): return tab + str(x)
	res = []
	ntab = tab + "  "
	for k, v in x.items():
		if isinstance(v, dict):
			res.append(tab + str(k) + ":\n" + yaml_dump(v, ntab) + "\n")
		else:
			res.append(tab + str(k) + ": " + (f"\"{v}\"" if isinstance(v, str) else str(v)) + "\n")
	return "".join(res)[:-1]


def choose_dir(temp_dir):
	while True:
		dir_name = os.path.join(temp_dir, str(random.randint(0, 999999)))
		try:
			os.mkdir(dir_name)
			return dir_name
		except FileExistsError as e:
			pass


os.chdir(os.path.dirname(__file__))  # go to home dir

try:
	cfg = get_input()
	mode = cfg.pop("mode")
	engine = cfg.pop("engine")
	model_type = cfg.pop("_model_type").upper()
	cfg = {"name": model_type, model_type: cfg}

	base_folder = choose_dir("workdir/res")
	folder = lambda x: os.path.join(base_folder, x)
	os.mkdir(folder("build"))

	# save config
	with open(folder("build/config.yaml"), "w") as f:
		f.write(yaml_dump(cfg))

	print(os.path.abspath(base_folder), end="")

except Exception as e:
	print(repr(e))