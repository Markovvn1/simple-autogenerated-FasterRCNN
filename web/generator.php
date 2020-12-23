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
$output=null;
$retval=null;
exec("PYTHONIOENCODING=utf-8 ".$command." '".$request."'", $output, $retval);
if ($retval != 0 || count($output) != 1 || !is_dir($output[0]))
{
    echo "Python script failed with status code $retval\n";
    echo '<pre>'; print_r(implode("\n", $output)); echo '</pre>';
    exit;
}

$work_dir = $output[0];
$output=null;
exec("cd \"".$work_dir."\" && zip -r9 model.zip model", $output, $retval);  // create zip

if ($retval != 0)
{
    echo "Zip failed with status code $retval\n";
    echo '<pre>'; print_r(implode("\n", $output)); echo '</pre>';
    exit;
}

// load file
header('Content-Type: application/zip');
header("Content-Transfer-Encoding: Binary");
header("Content-disposition: attachment; filename=\"model.zip\"");
readfile($work_dir."/model.zip");

shell_exec("rm -rf \"".$work_dir."\"");  // delete all

?>