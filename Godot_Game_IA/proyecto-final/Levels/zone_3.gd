extends Area2D

@export var zone_id: int = 3

func _ready() -> void:
	add_to_group("PROGRESS_ZONE")
	connect("body_entered", Callable(self, "_on_body_entered"))

func _on_body_entered(body: Node) -> void:
	if body is CharacterBody2D:
		var controller = get_tree().get_first_node_in_group("AGENT")
		if controller:
			controller.current_zone = zone_id
