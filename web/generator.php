<?php

unset($_POST['x']);
unset($_POST['y']);

if ($_POST["_model_type"] == "faster_rcnn")
{
	if (isset($_POST['BACKBONE']['RESNET']))
	{
		$_POST['BACKBONE']['RESNET']['stride_in_1x1'] = isset($_POST['BACKBONE']['RESNET']['stride_in_1x1']);
	}
	if (isset($_POST['ROI_HEADS']['STANDARD']))
	{
		$_POST['ROI_HEADS']['STANDARD']['is_agnostic'] = isset($_POST['ROI_HEADS']['STANDARD']['is_agnostic']);
		$_POST['ROI_HEADS']['STANDARD']['TRAIN']['append_gt_to_proposal'] = isset($_POST['ROI_HEADS']['STANDARD']['TRAIN']['append_gt_to_proposal']);
		if (isset($_POST['ROI_HEADS']['STANDARD']['BOX_HEAD']['FastRCNNConvFC']))
		{
			if (!isset($_POST['ROI_HEADS']['STANDARD']['BOX_HEAD']['FastRCNNConvFC']['fc'])) $_POST['ROI_HEADS']['STANDARD']['BOX_HEAD']['FastRCNNConvFC']['fc'] = [];
			if (!isset($_POST['ROI_HEADS']['STANDARD']['BOX_HEAD']['FastRCNNConvFC']['conv'])) $_POST['ROI_HEADS']['STANDARD']['BOX_HEAD']['FastRCNNConvFC']['conv'] = [];
		}
	}
}

$request = json_encode($_POST);
$command = escapeshellcmd(realpath("main.py"));
$output = shell_exec($command." '".$request."'");
// echo nl2br($output);

if (is_dir($output))
{
	shell_exec("cd \"".$output."\" && zip -r9 build.zip build");  # create zip

	# load file
	header('Content-Type: application/zip');
	header("Content-Transfer-Encoding: Binary"); 
	header("Content-disposition: attachment; filename=\"build.zip\""); 
	readfile($output."/build.zip");

	shell_exec("rm -rf \"".$output."\"");  # delete all
}
else
	echo $output;
?>