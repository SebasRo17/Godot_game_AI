extends Area2D
class_name CheckpointArea2D

@export var snap_to_floor: bool = true   # Ajusta Y unos px arriba si quieres evitar overlaps

signal checkpoint_reached(pos: Vector2)

func _ready() -> void:
	monitoring = true
	monitorable = true
	connect("body_entered", Callable(self, "_on_body_entered"))

func _on_body_entered(body: Node) -> void:
	if body is CharacterBody2D and body.has_method("set_respawn"):
		var p := global_position
		if snap_to_floor:
			p.y -= 4.0
		body.set_respawn(p)
		emit_signal("checkpoint_reached", p)
