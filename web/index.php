<?php
if (!isset($_GET["model"]))
	$_GET["model"] = "faster_rcnn";

function is_model($model)
{
	return ($_GET["model"] == $model);
}
?>

<!DOCTYPE html>
<head>
	<title>Создание сети</title>
	<link rel="icon" href="favicon.ico">
	<link rel="stylesheet" href="css/style.css">
</head>
<body>
	<form name="model_type" action="index.php" method="GET">
		<dev class="header">
			<h1>Создание сети</h1>
			<select id="select_model" name="model">
				<option <?php if (is_model("faster_rcnn")) echo "selected"; ?> value="faster_rcnn">FasterRCNN</option>
				<option <?php if (is_model("yolo_v4")) echo "selected"; ?> value="yolo_v4">YOLOv4</option>
			</select>
		</dev>
	</form>

	<form name="config" action="generator.php" method="POST">
		<?php echo '<input name="_model_type" type="hidden" value="'.$_GET["model"].'">'; ?>

		<?php
		$file_name = "models/".$_GET["model"].".php";
		if (is_file($file_name))
			include $file_name;
		else
			echo "<h3>Error: model not found</h3>";
		?>

		<dev class="footer-wrapper">
			<dev class="footer">
				<input type="image" title="download" src="images/download.svg" height="20em">
				<select name="mode">
					<option selected value="train">train + test</option>
					<option value="test">test only</option>
				</select>
				<h3>mode: </h3>
				<select name="engine">
					<option selected value="pytorch">pytorch</option>
				</select>
				<h3>engine: </h3>
			</dev>
		</dev>
	</form>

	<script defer type="text/javascript" src="js/jquery.min.js"></script>
	<script defer type="text/javascript" src="js/main.js"></script>
</body>
