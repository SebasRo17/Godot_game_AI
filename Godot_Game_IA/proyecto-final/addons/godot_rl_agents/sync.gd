extends Node

# --fixed-fps 2000 --disable-render-loop

enum ControlModes {HUMAN, TRAINING, ONNX_INFERENCE}
@export var control_mode: ControlModes = ControlModes.HUMAN
@export_range(1, 10, 1, "or_greater") var action_repeat := 8
@export_range(1, 10, 1, "or_greater") var speed_up = 1
@export var onnx_model_path := ""

@onready var start_time = Time.get_ticks_msec()

const MAJOR_VERSION := "0"
const MINOR_VERSION := "7" 
const DEFAULT_PORT := "11008"
const DEFAULT_SEED := "1"
var stream : StreamPeerTCP = null
var connected = false
var message_center
var should_connect = true
var agents
var need_to_send_obs = false
var args = null
var initialized = false
var just_reset = false
var onnx_model = null
var n_action_steps = 0

var _action_space : Dictionary
var _obs_space : Dictionary

# Called when the node enters the scene tree for the first time.
func _ready():
	process_mode = Node.PROCESS_MODE_ALWAYS
	await get_tree().root.ready
	get_tree().set_pause(true)
	_initialize()
	await get_tree().create_timer(1.0).timeout
	get_tree().set_pause(false)
	print("SCENE:", get_tree().current_scene.name)
	
func _initialize():
	_get_agents()
	_obs_space = agents[0].get_obs_space()
	_action_space = agents[0].get_action_space()
	args = _get_args()
	# Si viene con --port=XXXX, asumimos que es modo entrenamiento
	if args.has("port"):
		control_mode = ControlModes.TRAINING
	print("MODE:", control_mode)
	print("PORT:", _get_port())
	Engine.physics_ticks_per_second = _get_speedup() * 60 # Replace with function body.
	Engine.time_scale = _get_speedup() * 1.0
	prints("physics ticks", Engine.physics_ticks_per_second, Engine.time_scale, _get_speedup(), speed_up)
	
	_set_heuristic("human")
	match control_mode:
		ControlModes.TRAINING:
			connected = connect_to_server()
			if connected:
				_set_heuristic("model")
				_handshake()
				_send_env_info()  
			else:
				push_warning("Couldn't connect to Python server, using human controls instead. ",
				"Did you start the training server using e.g. `gdrl` from the console?")
		ControlModes.ONNX_INFERENCE:
				assert(FileAccess.file_exists(onnx_model_path), "Onnx Model Path set on Sync node does not exist: %s" % onnx_model_path)
				onnx_model = ONNXModel.new(onnx_model_path, 1)
				_set_heuristic("model")	
	
	_set_seed()
	_set_action_repeat()
	initialized = true  

func _physics_process(_delta):
	if connected:
		# IMPORTANTÍSIMO: refresca el buffer de red
		stream.poll()

		# 1) Si toca enviar step (respuesta al action anterior)
		if need_to_send_obs:
			need_to_send_obs = false
			var reward = _get_reward_from_agents()
			var done = _get_done_from_agents()
			var obs = _get_obs_from_agents()
			_send_dict_as_json_message({
				"type": "step",
				"obs": obs,
				"reward": reward,
				"done": done
			})

		# 2) Procesar mensajes entrantes (reset/action/call/close)
		if stream.get_available_bytes() >= 4:
			get_tree().set_pause(true)
			var handled = handle_message()
			if not handled:
				get_tree().set_pause(false)

		return

	# ---- NO CONECTADO (humano u ONNX) ----
	if n_action_steps % action_repeat != 0:
		n_action_steps += 1
		return

	n_action_steps += 1

	if onnx_model != null:
		var obs : Array = _get_obs_from_agents()
		var actions = []

		for o in obs:
			var action = onnx_model.run_inference(o["obs"], 1.0)
			action["output"] = clamp_array(action["output"], -1.0, 1.0)
			actions.append(_extract_action_dict(action["output"]))

		_set_agent_actions(actions)
		need_to_send_obs = true
		_reset_agents_if_done()
	else:
		_reset_agents_if_done()

func _extract_action_dict(action_array: Array):
	var index = 0
	var result = {}
	for key in _action_space.keys():
		var size = _action_space[key]["size"]
		if _action_space[key]["action_type"] == "discrete":
			result[key] = round(action_array[index])
		else:
			result[key] = action_array.slice(index,index+size)
		index += size
		
	return result

func _get_agents():
	agents = get_tree().get_nodes_in_group("AGENT")

func _set_heuristic(heuristic):
	for agent in agents:
		agent.set_heuristic(heuristic)

func _handshake():
	print("performing handshake")
	
	var json_dict = _get_dict_json_message()
	assert(json_dict["type"] == "handshake")
	var major_version = json_dict["major_version"]
	var minor_version = json_dict["minor_version"]
	if major_version != MAJOR_VERSION:
		print("WARNING: major verison mismatch ", major_version, " ", MAJOR_VERSION)  
	if minor_version != MINOR_VERSION:
		print("WARNING: minor verison mismatch ", minor_version, " ", MINOR_VERSION)
		
	print("handshake complete")

func _get_dict_json_message():
	# esperar header (4 bytes)
	while stream.get_available_bytes() < 4:
		stream.poll()
		if stream.get_status() != 2:
			print("Server disconnected")
			return null
		OS.delay_usec(50)

	var msg_len := stream.get_u32()

	# esperar payload completo
	while stream.get_available_bytes() < msg_len:
		stream.poll()
		if stream.get_status() != 2:
			print("Server disconnected")
			return null
		OS.delay_usec(50)

	var got = stream.get_data(msg_len) # [err, PackedByteArray]
	var err = got[0]
	if err != OK:
		print("Error reading payload:", err)
		return null

	var payload: PackedByteArray = got[1]
	var message := payload.get_string_from_utf8()

	var json_data = JSON.parse_string(message)
	if json_data == null:
		print("Invalid JSON received:", message)
		return null

	return json_data

func _send_dict_as_json_message(dict):
	var json_text := JSON.stringify(dict)
	var data: PackedByteArray = json_text.to_utf8_buffer()
	stream.put_u32(data.size())
	stream.put_data(data)

func _send_env_info():
	var json_dict = _get_dict_json_message()
	assert(json_dict["type"] == "env_info")

		
	var message = {
		"type" : "env_info",
		"observation_space": _obs_space,
		"action_space":_action_space,
		"n_agents": len(agents)
		}
	_send_dict_as_json_message(message)

func connect_to_server():
	print("Waiting for one second to allow server to start")
	OS.delay_msec(1000)
	print("trying to connect to server")
	stream = StreamPeerTCP.new()
	
	# "localhost" was not working on windows VM, had to use the IP
	var ip = "127.0.0.1"
	var port = _get_port()
	var connect = stream.connect_to_host(ip, port)
	# stream.set_no_delay(true) # TODO check if this improves performance or not
	stream.poll()
	# Fetch the status until it is either connected (2) or failed to connect (3)
	while stream.get_status() < 2:
		stream.poll()
	return stream.get_status() == 2

func call_on_path(node_path: String, method: String, args: Array = []):
	var n = get_node_or_null(node_path)
	if n == null:
		return {"ok": false, "error": "node_not_found", "path": node_path}
	if not n.has_method(method):
		return {"ok": false, "error": "method_not_found", "method": method}
	var result = n.callv(method, args)
	return {"ok": true, "result": result}

func _get_args():
	print("getting command line arguments")
	var arguments = {}
	for argument in OS.get_cmdline_args():
		print(argument)
		if argument.find("=") > -1:
			var key_value = argument.split("=")
			arguments[key_value[0].lstrip("--")] = key_value[1]
		else:
			# Options without an argument will be present in the dictionary,
			# with the value set to an empty string.
			arguments[argument.lstrip("--")] = ""

	return arguments   

func _get_speedup():
	print(args)
	return args.get("speedup", str(speed_up)).to_int()

func _get_port():    
	return args.get("port", DEFAULT_PORT).to_int()

func _set_seed():
	var _seed = args.get("env_seed", DEFAULT_SEED).to_int()
	seed(_seed)

func _set_action_repeat():
	action_repeat = args.get("action_repeat", str(action_repeat)).to_int()
	
func disconnect_from_server():
	stream.disconnect_from_host()



func handle_message() -> bool:
	# get json message: reset, step, close
	var message = _get_dict_json_message()
	
	if message == null:
		return false   # No llegó nada válido todavía

	if not message.has("type"):
		print("Invalid message received: ", message)
		return false
		
	if message["type"] == "close":
		print("received close message, closing game")
		get_tree().quit()
		get_tree().set_pause(false) 
		return true
		
	if message["type"] == "reset":
		print("resetting all agents")
		_reset_all_agents()
		# IMPORTANTE: forza que los agentes ya estén listos para dar obs
		# Si tu agente necesita 1 tick para resetear, entonces aquí debes llamar agent.reset() real
		# o hacer que get_obs() devuelva obs válido aunque needs_reset=tr		
		var obs = _get_obs_from_agents()
		_send_dict_as_json_message({"type":"reset","obs": obs})
		get_tree().set_pause(false)
		return true

		
	if message["type"] == "call":
		if message.has("path"):
			var ret = call_on_path(
				String(message["path"]),
				String(message.get("method", "")),
				message.get("args", [])
			)
			_send_dict_as_json_message({"type":"call","returns":[ret]})
			return true

		var method = message["method"]
		var returns = _call_method_on_agents(method)
		_send_dict_as_json_message({"type":"call","returns": returns})
		return true

	
	if message["type"] == "action":
		var action = message["action"]
		_set_agent_actions(action) 
		need_to_send_obs = true
		get_tree().set_pause(false) 
		return true
		
	print("message was not handled")
	return false

func _call_method_on_agents(method):
	var returns = []
	for agent in agents:
		returns.append(agent.call(method))
		
	return returns


func _reset_agents_if_done():
	for agent in agents:
		if agent.get_done(): 
			agent.set_done_false()

func _reset_all_agents():
	for agent in agents:
		agent.needs_reset = true
		#agent.reset()   

func _get_obs_from_agents():
	var obs = []
	for agent in agents:
		obs.append(agent.get_obs())

	return obs
	
func _get_reward_from_agents():
	var rewards = [] 
	for agent in agents:
		rewards.append(agent.get_reward())
		agent.zero_reward()
	return rewards    
	
func _get_done_from_agents():
	var dones = [] 
	for agent in agents:
		var done = agent.get_done()
		if done: agent.set_done_false()
		dones.append(done)
	return dones    
	
func _set_agent_actions(actions):
	for i in range(len(actions)):
		agents[i].set_action(actions[i])
	
func clamp_array(arr : Array, min:float, max:float):
	var output : Array = []
	for a in arr:
		output.append(clamp(a, min, max))
	return output
	
func start_training_mode():
	if connected:
		return

	print("Connecting to Python server from UI...")

	connected = connect_to_server()

	if connected:
		_set_heuristic("model")
		_handshake()
		_send_env_info()
	else:
		print("FAILED to connect to server")


func start_human_mode():
	if connected:
		disconnect_from_server()
	connected = false
	_set_heuristic("human")
