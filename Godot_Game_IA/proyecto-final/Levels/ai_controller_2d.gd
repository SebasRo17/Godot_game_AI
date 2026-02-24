extends Node2D
class_name AIController2D2

enum ControlModes {HUMAN, TRAINING, ONNX_INFERENCE}

@export var control_mode: ControlModes = ControlModes.TRAINING
@export var player_path: NodePath
@export var goal_path: NodePath
@export var death_y: float = 2000.0
@export var reset_after: int = 1000

var heuristic := "human"
var done := false
var reward := 0.0
var needs_reset := false
var _level: Node = null
var _prev_x := 0.0
var _stuck_steps := 0
var previous_on_wall := false
var _last_move_dir := 0
var _oscillation_count := 0
var current_zone: int = 0
var prev_zone: int = 0
var no_progress_steps: int = 0
var last_death_zone: int = -1

var _player: CharacterBody2D
var _goal: Area2D

var _action: Dictionary = { "move": 0, "jump": false, "dash": false }

var _goal_reached := false
var _prev_goal_dist := INF


func _ready() -> void:
	add_to_group("AGENT")
	_setup_player_and_goal()
	_reset_goal_distance_cache()
	_level = get_tree().current_scene

	var zones = get_tree().get_nodes_in_group("PROGRESS_ZONE")
	for z in zones:
		if z == null:
			continue
		var zone_id := _parse_zone_id(z.name) # z.name = "Zone_3"
		var cb := Callable(self, "_on_zone_entered").bind(zone_id)
		if not z.is_connected("body_entered", cb):
			z.connect("body_entered", cb)

	print("[AIController2D] ready -> player:%s goal:%s" % [str(_player), str(_goal)])

func _parse_zone_id(zone_name: String) -> int:
	if zone_name.begins_with("Zone_"):
		return int(zone_name.split("_")[1])
	return 0

func _on_zone_entered(body: Node, zone_id: int) -> void:
	if body != _player:
		return
	current_zone = zone_id
	print("Zona actual:", current_zone)

func _physics_process(_delta: float) -> void:
	if _level and "episode_finished" in _level and _level.episode_finished:
		return
	# Si Sync pide reset, reseteamos aquí
	if needs_reset:
		_do_reset()
		return

	if not _player:
		return

	# Aplicar SIEMPRE la última acción recibida desde Python
	if _player.has_method("apply_ai_action"):
		_player.apply_ai_action(
			int(_action["move"]),
			bool(_action["jump"]),
			bool(_action["dash"])
		)

	# checar caída
	if _player.position.y > death_y:
		done = true
		needs_reset = true
		
	if _player.is_dead:
		done = true
		needs_reset = true


# ───────── CONFIGURACIÓN PLAYER / GOAL ─────────
func _setup_player_and_goal() -> void:
	# PLAYER
	if player_path != NodePath(""):
		_player = get_node_or_null(player_path) as CharacterBody2D
	if _player == null:
		_player = _autodetect_player()

	# GOAL
	if goal_path != NodePath(""):
		_goal = get_node_or_null(goal_path) as Area2D
	if _goal == null:
		_goal = _autodetect_goal()

	if _goal and not _goal.is_connected("body_entered", Callable(self, "_on_goal_body_entered")):
		_goal.connect("body_entered", Callable(self, "_on_goal_body_entered"))


func _autodetect_player() -> CharacterBody2D:
	var root := get_tree().current_scene
	if root == null:
		return null

	# 1º: nodo llamado "Player"
	if root.has_node("Player"):
		var n := root.get_node("Player")
		if n is CharacterBody2D:
			return n

	# 2º: primer CharacterBody2D que encuentre
	for c in root.get_children():
		if c is CharacterBody2D:
			return c
	return null


func _autodetect_goal() -> Area2D:
	var root := get_tree().current_scene
	if root == null:
		return null

	# 1º: grupo "GOAL"
	var group := get_tree().get_nodes_in_group("GOAL")
	if group.size() > 0 and group[0] is Area2D:
		return group[0]

	# 2º: primera Area2D que encuentre
	for c in root.get_children():
		if c is Area2D:
			return c
	return null


# ───────── API PARA SYNC / PYTHON ─────────
func get_info() -> Dictionary:
	return {
		"observation_space": get_obs_space(),
		"action_space": get_action_space()
	}

func get_obs() -> Dictionary:
	if _player == null:
		return {"obs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

	var on_floor: float = 1.0 if _player.is_on_floor() else 0.0

	var can_dash: float = 1.0
	if "canDash" in _player:
		can_dash = 1.0 if _player.canDash else 0.0

	var zone_norm: float = clampf(float(current_zone) / 10.0, 0.0, 1.0)

	var wall_left: bool = false
	var wall_right: bool = false

	if _player.LeftRaycast:
		wall_left = _player.LeftRaycast.is_colliding()
	if _player.RightRaycast:
		wall_right = _player.RightRaycast.is_colliding()

	var goal_dx: float = 0.0
	var goal_dy: float = 0.0
	if _goal:
		goal_dx = (_goal.global_position.x - _player.global_position.x) / 1000.0
		goal_dy = (_goal.global_position.y - _player.global_position.y) / 1000.0

	var time_norm: float = 0.0
	var level = get_tree().current_scene
	if level and "run_time" in level:
		time_norm = clampf(level.run_time / 30.0, 0.0, 1.0)

	return {
		"obs": [
			_player.global_position.x,
			_player.global_position.y,
			_player.velocity.x,
			_player.velocity.y,
			on_floor,
			can_dash,
			1.0 if wall_left else 0.0,
			1.0 if wall_right else 0.0,
			zone_norm,
			time_norm,
			goal_dx,
			goal_dy,
			float(prev_zone) / 10.0
		]
	}


func get_obs_space() -> Dictionary:
	return {
		"obs": {
			"size": [13],
			"space": "box"
		}
	}

func get_action_space() -> Dictionary:
	return {
		"move": {"size": 3, "action_type": "discrete"}, # izq / nada / der
		"jump": {"size": 2, "action_type": "discrete"},
		"dash": {"size": 2, "action_type": "discrete"}
	}

func set_action(action: Dictionary) -> void:
	if action.is_empty():
		return

	var raw_move: int = int(action.get("move", 1))
	if raw_move == 0:
		_action["move"] = -1
	elif raw_move == 1:
		_action["move"] = 0
	else:
		_action["move"] = 1
	_action["jump"] = bool(action.get("jump", false))
	_action["dash"] = bool(action.get("dash", false))

	# DEBUG: ver que llegan acciones desde Python
	print("[AIController2D] action:", _action)


func get_reward() -> float:
	var r: float = 0.0
	if not _player:
		return r

	var wall_left: bool = false
	var wall_right: bool = false

	if _player.LeftRaycast:
		wall_left = _player.LeftRaycast.is_colliding()
	if _player.RightRaycast:
		wall_right = _player.RightRaycast.is_colliding()

	var near_wall: bool = wall_left or wall_right
	var move_dir: int = int(_action.get("move", 0))

	var goal_is_below: bool = false
	if _goal:
		goal_is_below = (_goal.global_position.y > _player.global_position.y + 50.0)

	# 0️⃣ progreso en X
	var dx: float = _player.global_position.x - _prev_x
	_prev_x = _player.global_position.x
	r += clampf(dx / 60.0, -0.2, 0.2)

	# 1️⃣ zonas
	if current_zone > prev_zone:
		r += 3.0
		no_progress_steps = 0
	elif current_zone < prev_zone:
		r -= 1.5
	else:
		no_progress_steps += 1
	prev_zone = current_zone

	# 2️⃣ tiempo suave
	r -= 0.0005

	# 3️⃣ caída sin pared → castigo
	if _player.velocity.y > 350.0 and (not near_wall) and (not goal_is_below):
		r -= 0.03

	# 4️⃣ caída + pared → premia empujar hacia pared
	if _player.velocity.y > 150.0 and near_wall:
		if wall_left and move_dir == -1:
			r += 0.03
		elif wall_right and move_dir == 1:
			r += 0.03

	# 5️⃣ estancamiento
	if no_progress_steps > 180:
		r -= 0.02

	# 6️⃣ muerte
	if _player.global_position.y > death_y or _player.is_dead:
		r -= 12.0
		done = true
		needs_reset = true
		last_death_zone = current_zone

	# 7️⃣ meta
	if _goal_reached:
		r += 40.0
		_goal_reached = false
		done = true

	reward = r
	return r


func get_done() -> bool:
	var is_done := done
	if _level and "episode_finished" in _level:
		is_done = is_done or _level.episode_finished
	if is_done:
		print("[AIController2D] DONE detectado")
	return is_done

func reset() -> void:
	_do_reset()


func _do_reset() -> void:
	needs_reset = false
	done = false
	reward = 0.0
	current_zone = 0
	prev_zone = 0
	no_progress_steps = 0
	_goal_reached = false
	_player.is_dead = false
	_reset_goal_distance_cache()
	_prev_x = _player.global_position.x

	if not _player:
		return

	# Curriculum spawn (fase actual)
	if get_tree().current_scene.has_node("SpawnPoint_B"):
		var sp := get_tree().current_scene.get_node("SpawnPoint_B")
		_player.global_position = sp.global_position
		_player.velocity = Vector2.ZERO
	else:
		# Fallback normal
		if _player.has_method("respawn"):
			_player.respawn()
		else:
			_player.velocity = Vector2.ZERO

func zero_reward() -> void:
	reward = 0.0

func set_done_false() -> void:
	done = false

func set_heuristic(h) -> void:
	heuristic = str(h)


# ───────── UTILIDADES / SEÑALES ─────────
func _reset_goal_distance_cache() -> void:
	if _player and _goal:
		_prev_goal_dist = _goal.global_position.distance_to(_player.global_position)
	else:
		_prev_goal_dist = INF


func _on_goal_body_entered(body: Node) -> void:
	if _player and body == _player:
		_goal_reached = true


