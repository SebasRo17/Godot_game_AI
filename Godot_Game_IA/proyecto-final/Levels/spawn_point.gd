extends Node2D
class_name SpawnPoint

@export var spawn_id: int = 1

func get_spawn_position() -> Vector2:
	return global_position
