extends Area2D

func _ready():
	add_to_group("TRAINING_BLOCK")

func _on_body_entered(body):
	if body is CharacterBody2D:
		# Solo afecta durante entrenamiento
		if get_tree().get_nodes_in_group("AGENT").size() > 0:
			body.is_dead = true
