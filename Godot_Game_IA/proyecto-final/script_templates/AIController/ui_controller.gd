extends CanvasLayer

@export var sync_path: NodePath
@onready var sync_node := get_node(sync_path)

@onready var toggle_button = $"Control/Jugar con IA"

var ia_active := false

func _ready():
	toggle_button.pressed.connect(_on_button_pressed)
	update_text()

func _on_button_pressed():
	set_ai_enabled(!ia_active) # reutiliza la misma lógica

func update_text():
	toggle_button.text = "Jugar sin IA" if ia_active else "Jugar con IA"

func set_ai_enabled(enable: bool):
	if ia_active == enable:
		return

	ia_active = enable

	if ia_active:
		print("IA ACTIVADA")

		# Para jugar normal: conecta si hace falta, SIN _initialize()
		sync_node.control_mode = sync_node.ControlModes.TRAINING
		if not sync_node.connected:
			sync_node.start_training_mode()

	else:
		print("MODO HUMANO")

		sync_node.control_mode = sync_node.ControlModes.HUMAN
		if sync_node.connected:
			sync_node.disconnect_from_server()
		sync_node._set_heuristic("human") # opcional, pero útil

	update_text()
