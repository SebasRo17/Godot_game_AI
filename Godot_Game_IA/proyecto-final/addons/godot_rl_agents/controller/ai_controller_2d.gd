extends Node2D
class_name AIController2D

enum ControlModes {
	INHERIT_FROM_SYNC, 
	HUMAN, 
	TRAINING, 
	ONNX_INFERENCE, 
	RECORD_EXPERT_DEMOS 
}

@export var control_mode: ControlModes = ControlModes.INHERIT_FROM_SYNC
@export var onnx_model_path: String = ""
@export var reset_after: int = 1000

@export_group("Record expert demos mode options")
@export var expert_demo_save_path: String
@export var remove_last_episode_key: InputEvent
@export var action_repeat: int = 1
@export_group("Multi-policy mode options")
@export var policy_name: String = "shared_policy"

var onnx_model: ONNXModel
var heuristic: String = "human"
var done: bool = false
var reward: float = 0.0
var n_steps: int = 0
var needs_reset: bool = false

var _player: CharacterBody2D

func _ready():
	add_to_group("AGENT")

func init(player: CharacterBody2D):
	_player = player

func _physics_process(delta):
	n_steps += 1
	if n_steps > reset_after:
		needs_reset = true
		
	# Implement AI logic to control the player
	handle_ai_control()

func handle_ai_control():
	# AI behavior decision logic
	# Adjust the action based on the player's state
	if _player != null and _player.is_on_floor():
	# Your existing code # Assuming you have a way to check if the player is on the floor
		if Input.is_action_pressed("right"):
			set_action({"move": 1}) # Move right
		elif Input.is_action_pressed("left"):
			set_action({"move": -1}) # Move left
		else:
			set_action({"move": 0}) # Idle

		if Input.is_action_just_pressed("jump") and _player.currentState != "jump" and _player.currentState != "fall":
			set_action({"jump": true})
			
		if Input.is_action_just_pressed("dash"):
			set_action({"dash": true})

func get_obs() -> Dictionary:
	# Implement your observation logic here
	assert(false, "the get_obs method is not implemented when extending from ai_controller")
	return {"obs": []}

func get_reward() -> float:
	# Implement your reward logic here
	assert(false, "the get_reward method is not implemented when extending from ai_controller")
	return 0.0

func get_action_space() -> Dictionary:
	assert(false, "the get_action_space method is not implemented when extending from ai_controller")
	return {
		"example_actions_continous": {"size": 2, "action_type": "continuous"},
		"example_actions_discrete": {"size": 2, "action_type": "discrete"},
	}

func set_action(action) -> void:
	# Implement your action setting logic here
	assert(false, "the set_action method is not implemented when extending from ai_controller")

func get_obs_space():
	var obs = get_obs()
	return {
		"obs": {"size": [len(obs["obs"])], "space": "box"},
	}

func reset():
	n_steps = 0
	needs_reset = false

func reset_if_done():
	if done:
		reset()

func set_heuristic(h):
	heuristic = h

func get_done() -> bool:
	return done

func set_done_false():
	done = false

func zero_reward():
	reward = 0.0
