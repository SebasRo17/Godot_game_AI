extends Node

var tcp_server : TCP_Server
var client : StreamPeerTCP
var port = 11008  # ¡Debe coincidir con el puerto del script Python!

func _ready():
	# Iniciar servidor TCP
	tcp_server = TCP_Server.new()
	var err = tcp_server.listen(port, "127.0.0.1")  # Localhost
	if err == OK:
		print("Servidor Godot iniciado en puerto ", port)
	else:
		push_error("Error al iniciar el servidor: ", err)

func _process(delta):
	if tcp_server.is_connection_available():
		# Aceptar conexión de Python
		client = tcp_server.take_connection()
		print("Cliente de Python conectado: ", client.get_connected_host())

	if client != null and client.get_status() == StreamPeerTCP.STATUS_CONNECTED:
		# Enviar observaciones al cliente (Python)
		var obs = _get_observations()  # Tus datos del entorno
		client.put_utf8_string(JSON.stringify(obs) + "\n")

		# Recibir acciones desde Python
		if client.get_available_bytes() > 0:
			var action_str = client.get_utf8_string()
			var action = JSON.parse_string(action_str)
			_apply_action(action)  # Ejecutar la acción en Godot

func _get_observations():
	# Lógica para obtener observaciones (ej: posición de nodos)
	return {
		"position": $Player.position,
		"velocity": $Player.linear_velocity
	}

func _apply_action(action):
	# Lógica para aplicar acciones (ej: mover jugador)
	$Player.move(action["direction"])
