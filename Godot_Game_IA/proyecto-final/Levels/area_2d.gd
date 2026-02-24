extends Area2D
class_name GoalArea2D

signal goal_reached

@export var target_spawn_path: NodePath
@export var target_spawn_id: int = 2
@export var set_player_respawn: bool = true
@export var y_lift: float = 4.0
@export var debug_prints: bool = false

func _ready() -> void:
	set_monitoring(true)
	set_monitorable(true)
	if not is_connected("body_entered", Callable(self, "_on_body_entered")):
		connect("body_entered", Callable(self, "_on_body_entered"))

func _on_body_entered(body: Node) -> void:
	if not (body is CharacterBody2D):
		return
	var player := body as CharacterBody2D

	var spawn := _resolve_target_spawn()
	if spawn:
		var pos: Vector2
		if spawn.has_method("get_spawn_position"):
			pos = (spawn as SpawnPoint).get_spawn_position()
		else:
			pos = (spawn as Node2D).global_position + Vector2(0, -y_lift)

		player.global_position = pos
		if "velocity" in player:
			player.velocity = Vector2.ZERO
		if set_player_respawn and player.has_method("set_respawn"):
			player.set_respawn(pos)

		if debug_prints:
			print("[GoalArea2D:", name, "] → teleported to:", pos)

	emit_signal("goal_reached")


func _resolve_target_spawn() -> Node:
	if target_spawn_path != NodePath(""):
		var n := get_node_or_null(target_spawn_path)
		if n: return n
	var root := get_tree().current_scene
	if not root: return null
	return _find_spawn_by_id(root, target_spawn_id)

func _find_spawn_by_id(n: Node, id: int) -> Node:
	if n is SpawnPoint and (n as SpawnPoint).spawn_id == id:
		return n
	for c in n.get_children():
		var r := _find_spawn_by_id(c, id)
		if r: return r
	return null
