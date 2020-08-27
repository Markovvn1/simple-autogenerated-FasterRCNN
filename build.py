import shutil
from builders import build_resnet

cfg = {
	"depth": 50,
	"norm": "BN",
	"num_groups": 1,
	"width_per_group": 64,
	"stem_out_channels": 64,
	"res2_out_channels": 256,
	"stride_in_1x1": True,
	"res5_dilation": 1
}

shutil.rmtree('build', ignore_errors=True)  # clean build directory

# build_resnet(cfg, test_only=False, integrate_backbone=False, lib_prefix=".libs.")
build_resnet(cfg, test_only=True, integrate_backbone=True, lib_prefix=".libs.")
print("Done.")