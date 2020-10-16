<dev title="Глубина сети">depth:</dev>
<select name="BACKBONE[RESNET][depth]">
	<option value="18">18</option>
	<option value="34">34</option>
	<option selected value="50">50</option>
	<option value="101">101</option>
	<option value="152">152</option>
</select>
<br>
<dev title="Тип слоя нормализации для conv">norm:</dev>
<select name="BACKBONE[RESNET][norm]">
	<option value="None">None</option>
	<option selected value="BN">BatchNorm</option>
	<option value="FrozenBN">FrozenBN</option>
</select>
<br>
<dev title="Количество групп для сверточных 3x3 слоев">num_groups:</dev>
<input name="BACKBONE[RESNET][num_groups]" type="number" min="1" step="1" value="1" class="pos_int">
<br>
<dev title="Количество каналов в каждой группе">width_per_group:</dev>
<input name="BACKBONE[RESNET][width_per_group]" type="number" min="1" step="1" value="64" class="pos_int">
<br>
<dev title="Количество каналов на выходе слоя stem">stem_out_channels:</dev>
<input name="BACKBONE[RESNET][stem_out_channels]" type="number" min="1" step="1" value="64" class="pos_int">
<br>
<dev title="Количество каналов на выходе слоя res2">res2_out_channels:</dev>
<input name="BACKBONE[RESNET][res2_out_channels]" type="number" min="1" step="1" value="256" class="pos_int">
<br>
<dev title="Будет ли stride проходить в слое 1x1 или же в слое 3x3">stride_in_1x1:</dev>
<input name="BACKBONE[RESNET][stride_in_1x1]" type="checkbox" checked>
<br>
<dev title="dilation в слое res5">res5_dilation:</dev>
<select name="BACKBONE[RESNET][res5_dilation]">
	<option selected value="1">1</option>
	<option value="2">2</option>
</select>
<br>
<dev title="Какие слои будут возвращаться данным модулем">out_features:</dev>
<select multiple name="BACKBONE[RESNET][out_features][]" size="1" class="ms_horizontal">
	<option value="stem">stem</option>
	<option selected value="res2">res2</option>
	<option selected value="res3">res3</option>
	<option selected value="res4">res4</option>
	<option selected value="res5">res5</option>
</select>