extends Area2D
class_name GoalArea2DCustom

signal player_reached_goal(player: Node)
signal goal_reached_signal

@export var one_shot: bool = false
@export var respawn_delay: float = 0.0
@export var debug_color: Color = Color(0, 1, 0, 0.25)

var _triggered := false
var goal_reached := false

func _ready() -> void:
	add_to_group("GOAL")
	monitoring = true
	monitorable = true
	if not is_connected("body_entered", Callable(self, "_on_body_entered")):
		connect("body_entered", Callable(self, "_on_body_entered"))
	if get_node_or_null("CollisionShape2D") == null:
		push_warning("[GoalArea2D] Falta CollisionShape2D.")
	if has_node("ColorRect"):
		$ColorRect.color = debug_color

func _on_body_entered(body: Node) -> void:
	if body is CharacterBody2D and not goal_reached:
		goal_reached = true
		print("GOAL ALCANZADO POR:", body.name)
		emit_signal("goal_reached_signal")

func reset_goal() -> void:
	_triggered = false
