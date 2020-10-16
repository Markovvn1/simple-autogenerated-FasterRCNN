function type_onchange()
{
	self = $(this);
	panels = self.siblings(".select_target"); //TODO
	target = panels.filter("#"+self.val())

	panels.addClass("hidden");
	if (target != null) target.removeClass("hidden");
}

$(document).ready(function() {
	$("#select_model").change(function() {document.forms["model_type"].submit();});
	$(".select_type").change(type_onchange);
	$(".ignore_enter").keypress(function(e) { return e.keyCode != 13; })
	$(".pos_int, .pos_float").change(function() {
		self = $(this);
		if (self.val() == "") self.val(0);
	});
});