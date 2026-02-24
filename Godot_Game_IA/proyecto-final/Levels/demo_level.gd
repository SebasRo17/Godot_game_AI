extends Node2D

var episode_finished := false
var run_time := 0.0

# Called when the node enters the scene tree for the first time.
func _ready():
	$goal_area_2d.goal_reached_signal.connect(_on_goal_reached)

func _on_goal_reached():
	print("EPISODIO TERMINADO (VICTORIA)")
	episode_finished = true
	

func _physics_process(delta):
	if episode_finished:
		return

	run_time += delta
	$CanvasLayer/Control/TimerLabel.text = _format_time(run_time)

func _format_time(t: float) -> String:
	var minutes = int(t) / 60
	var seconds = int(t) % 60
	var millis = int((t - int(t)) * 1000)
	return "%02d:%02d.%03d" % [minutes, seconds, millis]


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
