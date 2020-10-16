<dev title="Предсказывать положение для всех классов вместе или отдельно">is_agnostic:</dev>
<input name="ROI_HEADS[STANDARD][is_agnostic]" type="checkbox">
<br>
<dev title="Коэффициенты на которые будут умножаться значения локализации (xywh)">box_transform_weights:</dev><br>
<dev class="input_array">
	<input name="ROI_HEADS[STANDARD][box_transform_weights][]" type="number" min="0" step="any" value="10" class="pos_float"><input name="ROI_HEADS[STANDARD][box_transform_weights][]" type="number" min="0" step="any" value="10" class="pos_float"><input name="ROI_HEADS[STANDARD][box_transform_weights][]" type="number" min="0" step="any" value="5" class="pos_float"><input name="ROI_HEADS[STANDARD][box_transform_weights][]" type="number" min="0" step="any" value="5" class="pos_float">
</dev>

<dev title="Параметры модуля, который вырезает RoI фич">POOLER:</dev>
<select name="ROI_HEADS[STANDARD][POOLER][type]">
	<option value="RoIAlign">RoIAlign</option>
	<option selected value="RoIAlignV2">RoIAlignV2</option>
	<option value="RoIPool">RoIPool</option>
</select>
<dev class="panel_content">
	<dev title="Выходное разрешение">resolution:</dev>
	<dev class="input_inline_array">
		<input name="ROI_HEADS[STANDARD][POOLER][resolution][]" type="number" min="1" step="1" value="7" class="pos_int"><input name="ROI_HEADS[STANDARD][POOLER][resolution][]" type="number" min="1" step="1" value="7" class="pos_int">
	</dev>
	<br>
	<dev title="see torchvision.ops.roi_align">sampling_ratio:</dev>
	<input name="ROI_HEADS[STANDARD][POOLER][sampling_ratio]" type="number" step="1" value="-1" class="pos_int">
</dev>

<dev title="Параметры модуля уточнения коробок">BOX_HEAD:</dev>
<select class="select_type" name="ROI_HEADS[STANDARD][BOX_HEAD][name]">
	<option selected value="FastRCNNConvFC">FastRCNNConvFC</option>
</select>
<dev id="FastRCNNConvFC" class="panel_content">
	<dev title="Тип слоя нормализации для conv">norm:</dev>
	<select name="ROI_HEADS[STANDARD][BOX_HEAD][FastRCNNConvFC][norm]">
		<option selected value="None">None</option>
		<option value="BN">BatchNorm</option>
		<option value="FrozenBN">FrozenBN</option>
	</select>
	<br>
	<dev title="Количество каналов в conv-слоях">conv (<input type="number" min="0" step="1" value="0" class="pos_short">):</dev>
	<dev class="input_inline_array">
	</dev>
	<br>
	<dev title="Количество нейронов в fc-слоях">fc (<input type="number" min="0" step="1" value="2" class="pos_short">):</dev>
	<dev class="input_inline_array">
		<input name="ROI_HEADS[STANDARD][BOX_HEAD][FastRCNNConvFC][fc][]" type="number" min="1" step="1" value="1024" class="pos_int"><input name="ROI_HEADS[STANDARD][BOX_HEAD][FastRCNNConvFC][fc][]" type="number" min="1" step="1" value="1024" class="pos_int">
	</dev>
</dev>

<dev title="Параметры ошибок">LOSS:</dev>
<dev class="panel_content">
	<dev title="Вклад ошибки модуля в общую ошибку сети">global_weight:</dev>
	<input name="ROI_HEADS[STANDARD][LOSS][global_weight]" type="number" min="0" step="any" value="2" class="pos_float">
	<br>
	<dev title="Какую часть ошибки модуля занимает ошибка локализации">box_reg_weight:</dev>
	<input name="ROI_HEADS[STANDARD][LOSS][box_reg_weight]" type="number" min="0" max="1" step="any" value="0.5" class="pos_float">
	<br>
	<dev title="Тип ошибки локализации">bbox_reg_loss_type:</dev>
	<select class="select_type" name="ROI_HEADS[STANDARD][LOSS][bbox_reg_loss_type]">
		<option value="smooth_l1">Smooth L1</option>
		<option selected value="giou">GIoU</option>
	</select></h3>

	<dev id="smooth_l1" class="select_target panel_content ignore_enter hidden">
		<dev title="Параметр ошибки Smooth L1">smooth_l1_beta:</dev>
		<input name="ROI_HEADS[STANDARD][LOSS][smooth_l1_beta]" type="number" min="0" step="any" value="1" class="pos_float">
	</dev>
</dev>

<dev title="Параметры для обучения">TRAIN:</dev>
<dev class="panel_content">
	<dev title="Количество изображений, используемых для обучения модуля">batch_size_per_image:</dev>
	<input name="ROI_HEADS[STANDARD][TRAIN][batch_size_per_image]" type="number" min="1" step="1" value="512" class="pos_int">
	<br>
	<dev title="Процент положительных изображений в минибатче">positive_fraction:</dev>
	<input name="ROI_HEADS[STANDARD][TRAIN][positive_fraction]" type="number" min="0" max="1" step="any" value="0.25" class="pos_float">
	<br>
	<dev title="Добавлять таргеты к кандидатам. Ускоряет обучение этого модуля">append_gt_to_proposal</dev>
	<input name="ROI_HEADS[STANDARD][TRAIN][append_gt_to_proposal]" type="checkbox" checked>
	<br>
	<dev title="Пороги для определения является ли кандидат background или foreground">iou_thresholds:</dev>
	<dev class="input_inline_array">
		<input name="ROI_HEADS[STANDARD][TRAIN][iou_thresholds][]" type="number" min="0" max="1" step="any" value="0.5" class="pos_float"><input name="ROI_HEADS[STANDARD][TRAIN][iou_thresholds][]" type="number" min="0" max="1" step="any" value="0.5" class="pos_float">
	</dev>
</dev>

<dev title="Базовые параметры для тестирования (можно изменять программно)">TEST:</dev>
<dev class="panel_content">
	<dev title="Все претенденты с IoU > nms_thresh будут отсеяны">nms_thresh:</dev>
	<input name="ROI_HEADS[STANDARD][TEST][nms_thresh]" type="number" min="0" max="1" step="any" value="0.5" class="pos_float">
	<br>
	<dev title="Все претенденты с score < score_thresh будут отсеяны">score_thresh:</dev>
	<input name="ROI_HEADS[STANDARD][TEST][score_thresh]" type="number" min="0" max="1" step="any" value="0.85" class="pos_float">
</dev>