<dev class="panel ignore_enter">
	<dev class="column_panel">
		<dev class="row_panel">
			<h3 class="panel_name">Backbone</h3>
			<select class="select_type" name="BACKBONE[name]">
				<option selected value="RESNET">ResNet</option>
			</select>

			<dev id="RESNET" class="select_target panel_content">
				<?php include 'parts/resnet.php'; ?>
			</dev>
		</dev>
		<dev class="row_panel">
			<h3 class="panel_name">Neck</h3>
			<select class="select_type" name="NECK[name]">
				<option selected value="FPN">FPN</option>
			</select>

			<dev id="FPN" class="select_target panel_content">
				<?php include 'parts/fpn.php'; ?>
			</dev>
		</dev>
	</dev>
	<dev class="column_panel">
		<dev class="row_panel">
			<h3 class="panel_name">Proposal generator</h3>
			<select class="select_type" name="PROPOSAL_GENERATOR[name]">
				<option selected value="RPN">RPN</option>
			</select>

			<dev id="RPN" class="select_target panel_content">
				<?php include 'parts/rpn.php'; ?>
			</dev>
		</dev>
	</dev>
	<dev class="column_panel">
		<dev class="row_panel">
			<h3 class="panel_name">ROI heads</h3>
			<select class="select_type" name="ROI_HEADS[name]">
				<option selected value="STANDARD">Standard</option>
			</select>

			<dev id="STANDARD" class="select_target panel_content">
				<?php include 'parts/standard_roi_heads.php'; ?>
			</dev>
		</dev>
	</dev>
</dev>