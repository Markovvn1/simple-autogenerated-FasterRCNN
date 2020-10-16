<dev title="Тип слоя нормализации для conv">norm:</dev>
<select name="NECK[FPN][norm]">
	<option selected value="None">None</option>
	<option value="BN">BatchNorm</option>
	<option value="FrozenBN">FrozenBN</option>
</select>
<br>
<dev title="Способ объединения слоев">fuse_type:</dev>
<select name="NECK[FPN][fuse_type]">
	<option selected value="sum">sum</option>
	<option value="avg">avg</option>
</select>
<br>
<dev title="Количество слоев на выходе">out_channels:</dev>
<input name="NECK[FPN][out_channels]" type="number" min="1" step="1" value="256" class="pos_int">
<br>