extends Marker2D
class_name SpawnMarker

@export var spawn_id: int = 2       # 1 = Spawn1, 2 = Spawn2, etc.
@export var y_lift: float = 4.0     # levanta un poco al jugador para evitar encaje

func get_spawn_position() -> Vector2:
	return global_position + Vector2(0, -y_lift)
