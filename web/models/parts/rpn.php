<dev title="Минимальный размер коробки, которую может выдать данный модуль">min_box_size:</dev>
<input name="PROPOSAL_GENERATOR[RPN][min_box_size]" type="number" min="0" step="0.1" value="0" class="pos_float">
<br>
<dev title="Коэффициенты на которые будут умножаться значения локализации (xywh)">box_transform_weights:</dev><br>
<dev class="input_array">
	<input name="PROPOSAL_GENERATOR[RPN][box_transform_weights][]" type="number" min="0" step="any" value="1" class="pos_float"><input name="PROPOSAL_GENERATOR[RPN][box_transform_weights][]" type="number" min="0" step="any" value="1" class="pos_float"><input name="PROPOSAL_GENERATOR[RPN][box_transform_weights][]" type="number" min="0" step="any" value="1" class="pos_float"><input name="PROPOSAL_GENERATOR[RPN][box_transform_weights][]" type="number" min="0" step="any" value="1" class="pos_float">
</dev>
<dev title="Параметры генератора энкоров">ANCHOR_GENERATOR:</dev>
<dev class="panel_content">
	<dev title="Отношения сторон энкоров">ratios (<input type="number" min="1" step="1" value="3" class="pos_short">):</dev>
	<table class="input_table">
		<tr>
			<td><input name="PROPOSAL_GENERATOR[RPN][ANCHOR_GENERATOR][ratios][]" type="number" min="0.001" step="any" value="0.5" class="pos_float"></td>
			<td><input name="PROPOSAL_GENERATOR[RPN][ANCHOR_GENERATOR][ratios][]" type="number" min="0.001" step="any" value="1" class="pos_float"></td>
			<td><input name="PROPOSAL_GENERATOR[RPN][ANCHOR_GENERATOR][ratios][]" type="number" min="0.001" step="any" value="2" class="pos_float"></td>
		</tr>
	</table>
	<dev title="Размеры энкоров для каждого выхода Neck">sizes_per_out (<input type="number" min="1" step="1" value="1" class="pos_short">):</dev>
	<table class="input_table">
		<tr>
			<td><input name="PROPOSAL_GENERATOR[RPN][ANCHOR_GENERATOR][sizes][0][0]" type="number" min="0.001" step="any" value="32" class="pos_sfloat"></td>
			<td><input name="PROPOSAL_GENERATOR[RPN][ANCHOR_GENERATOR][sizes][1][0]" type="number" min="0.001" step="any" value="64" class="pos_sfloat"></td>
			<td><input name="PROPOSAL_GENERATOR[RPN][ANCHOR_GENERATOR][sizes][2][0]" type="number" min="0.001" step="any" value="128" class="pos_sfloat"></td>
			<td><input name="PROPOSAL_GENERATOR[RPN][ANCHOR_GENERATOR][sizes][3][0]" type="number" min="0.001" step="any" value="256" class="pos_float"></td>
			<td><input name="PROPOSAL_GENERATOR[RPN][ANCHOR_GENERATOR][sizes][4][0]" type="number" min="0.001" step="any" value="512" class="pos_float"></td>
		</tr>
	</table>
</dev>

<dev title="Параметры ошибок">LOSS:</dev>
<dev class="panel_content">
	<dev title="Вклад ошибки модуля в общую ошибку сети">global_weight</dev>:
	<input name="PROPOSAL_GENERATOR[RPN][LOSS][global_weight]" type="number" min="0" step="any" value="2" class="pos_float">
	<br>
	<dev title="Какую часть ошибки модуля занимает ошибка локализации">box_reg_weight</dev>:
	<input name="PROPOSAL_GENERATOR[RPN][LOSS][box_reg_weight]" type="number" min="0" max="1" step="any" value="0.5" class="pos_float">
	<br>
	<dev title="Тип ошибки локализации">bbox_reg_loss_type</dev>:
	<select class="select_type" name="PROPOSAL_GENERATOR[RPN][LOSS][bbox_reg_loss_type]">
		<option value="smooth_l1">Smooth L1</option>
		<option selected value="giou">GIoU</option>
	</select></h3>

	<dev id="smooth_l1" class="select_target panel_content hidden">
		<dev title="Параметр ошибки Smooth L1">smooth_l1_beta</dev>:
		<input name="PROPOSAL_GENERATOR[RPN][LOSS][smooth_l1_beta]" type="number" min="0" step="any" value="1" class="pos_float">
	</dev>
</dev>

<dev title="Параметры для обучения">TRAIN:</dev>
<dev class="panel_content">
	<dev title="Количество образцов, которое будет выбрано перед NMS">pre_topk:</dev>
	<input name="PROPOSAL_GENERATOR[RPN][TRAIN][pre_topk]" type="number" min="1" step="1" value="2000" class="pos_int">
	<br>
	<dev title="Все претенденты с IoU > nms_thresh будут отсеяны">nms_thresh:</dev>
	<input name="PROPOSAL_GENERATOR[RPN][TRAIN][nms_thresh]" type="number" min="0" max="1" step="any" value="0.7" class="pos_float">
	<br>
	<dev title="Количество образцов, которое будет выбрано после NMS">post_topk:</dev>
	<input name="PROPOSAL_GENERATOR[RPN][TRAIN][post_topk]" type="number" min="1" step="1" value="1000" class="pos_int">
	<br>
	<dev title="Количество изображений, используемых для обучения модуля">batch_size_per_image:</dev>
	<input name="PROPOSAL_GENERATOR[RPN][TRAIN][batch_size_per_image]" type="number" min="1" step="1" value="256" class="pos_int">
	<br>
	<dev title="Процент положительных изображений в минибатче">positive_fraction:</dev>
	<input name="PROPOSAL_GENERATOR[RPN][TRAIN][positive_fraction]" type="number" min="0" max="1" step="any" value="0.5" class="pos_float">
	<br>
	<dev title="Пороги для определения является ли anchor background или foreground">iou_thresholds:</dev>
	<dev class="input_inline_array">
		<input name="PROPOSAL_GENERATOR[RPN][TRAIN][iou_thresholds][]" type="number" min="0" max="1" step="any" value="0.3" class="pos_float"><input name="PROPOSAL_GENERATOR[RPN][TRAIN][iou_thresholds][]" type="number" min="0" max="1" step="any" value="0.7" class="pos_float">
	</dev>
</dev>

<dev title="Базовые параметры для тестирования (можно изменять программно)">TEST:</dev>
<dev class="panel_content">
	<dev title="Количество образцов, которое будет выбрано перед NMS">pre_topk:</dev>
	<input name="PROPOSAL_GENERATOR[RPN][TEST][pre_topk]" type="number" min="1" step="1" value="1000" class="pos_int">
	<br>
	<dev title="Все претенденты с IoU > nms_thresh будут отсеяны">nms_thresh:</dev>
	<input name="PROPOSAL_GENERATOR[RPN][TEST][nms_thresh]" type="number" min="0" max="1" step="any" value="0.7" class="pos_float">
	<br>
	<dev title="Количество образцов, которое будет выбрано после NMS">post_topk:</dev>
	<input name="PROPOSAL_GENERATOR[RPN][TEST][post_topk]" type="number" min="1" step="1" value="1000" class="pos_int">
</dev>