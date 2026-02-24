# Scripts/SpawnPoint.gd
extends Marker2D

@export var spawn_id: int = 1
@export var y_lift: float = 4.0

func _ready() -> void:
	add_to_group("SPAWN")

func get_spawn_position() -> Vector2:
	return global_position + Vector2(0, -y_lift)
