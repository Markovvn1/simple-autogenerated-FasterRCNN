from libs.resnet_builder import generate_resnet

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

print("Creating resnet.py")

with open("resnet.py", "w") as f:
	f.write(generate_resnet(cfg, test_only=False, lib_prefix="libs."))

print("Done.")