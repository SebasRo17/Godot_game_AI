extends CharacterBody2D
# Player controlable por humano O por IA (via apply_ai_action).

signal died

# ─────────────────────────────────────────────────────────
# ESTADOS
# ─────────────────────────────────────────────────────────
var states := ["idle", "run", "dash", "fall", "jump", "double_jump", "wall_slide", "wall_jump"]
var currentState: String = states[0]
var previousState: String = ""

# ─────────────────────────────────────────────────────────
# NODOS
# ─────────────────────────────────────────────────────────
@onready var PlayerSprite: Sprite2D = $Sprite2D
@onready var Anim: AnimationPlayer = $AnimationPlayer
@onready var RightRaycast: RayCast2D = $RightRaycast
@onready var LeftRaycast: RayCast2D = $LeftRaycast

# ─────────────────────────────────────────────────────────
# SQUASH & STRETCH
# ─────────────────────────────────────────────────────────
var recoverySpeed: float = 0.03
var landingSquash: float = 1.5
var landingStretch: float = 0.5
var jumpingSquash: float = 0.5
var jumpingStretch: float = 1.5

# ─────────────────────────────────────────────────────────
# INPUT (HUMANO + IA)
# ─────────────────────────────────────────────────────────
var movementInput: int = 0        # valor final que usan los estados
var lastDirection: int = 1

var _human_move: int = 0
var _ai_move: int = 0

var isJumpPressed: int = 0
var isJumpReleased: int = 0
var isDashPressed: int = 0

var _ai_prev_jump: bool = false
var _ai_prev_dash: bool = false

# Coyote / Jump buffer
var coyoteStartTime: int = 0
var elapsedCoyoteTime: int = 0
var coyoteDuration: int = 100
var jumpInput: int = 0
var is_dead: bool = false

# ─────────────────────────────────────────────────────────
# MOVIMIENTO
# ─────────────────────────────────────────────────────────
var currentSpeed: float = 0.0
var maxSpeed: float = 190.0
var acceleration: float = 25.0
var decceleration: float = 40.0
var airFriction: float = 60.0

# Dash
var dashSpeed: float = 400.0
var dashDurration: int = 180
var canDash: bool = true
var dashStartTime: int = 0
var elapsedDashTime: int = 0
var dashDirection: int = 1

# Gravedad / salto
var gravity: float = 500.0
var jumpBufferStartTime: int = 0
var elapsedJumpBuffer: int = 0
var jumpBuffer: int = 100

var jumpHeight: float = 160.0
var jumpVelocity: float = 0.0
var doubleJumpHeight: float = 100.0
var doubleJumpVelocity: float = 0.0
var isDoubleJumped: bool = false
var wallSlideSpeed: float = 50.0
var wallJumpHeight: float = 128.0
var wallJumpVelocity: float = 0.0

# ─────────────────────────────────────────────────────────
# VIDA / RESPAWN
# ─────────────────────────────────────────────────────────
@export var hp_max: int = 1
var hp: int = 1
@export var respawn_point: Vector2 = Vector2.ZERO

func _ready() -> void:
	jumpVelocity = -sqrt(2.0 * gravity * float(jumpHeight))
	doubleJumpVelocity = -sqrt(2.0 * gravity * float(doubleJumpHeight))
	wallJumpVelocity = -sqrt(2.0 * gravity * float(wallJumpHeight))

	if respawn_point == Vector2.ZERO:
		respawn_point = global_position
	hp = hp_max

# ─────────────────────────────────────────────────────────
# API DE IA (godot_rl_agents llama a esto)
# ─────────────────────────────────────────────────────────
func apply_ai_action(move: int, jump: bool, dash: bool) -> void:
	# IA propone movimiento; humano tiene prioridad si pulsa
	_ai_move = clampi(move, -1, 1)

	var jump_press: bool = jump and not _ai_prev_jump
	var jump_release: bool = (not jump) and _ai_prev_jump

	if jump_press:
		isJumpPressed = 1
	if jump_release:
		isJumpReleased = 1
	# Evitar spam de salto en el aire (solo IA)
	if not is_on_floor() and currentState != "wall_slide" and currentState != "fall":
		isJumpPressed = 0



	var dash_press: bool = dash and not _ai_prev_dash
	if dash_press:
		isDashPressed = 1

	_ai_prev_jump = jump
	_ai_prev_dash = dash

# ─────────────────────────────────────────────────────────
# FÍSICA
# ─────────────────────────────────────────────────────────
func _physics_process(delta: float) -> void:
	if get_tree().current_scene.episode_finished:
		return

	get_input()  # mezcla humano + IA

	apply_gravity(delta)
	call(currentState + "_logic", delta)

	move_and_slide()

	# señales de 1 frame
	isJumpPressed = 0
	isJumpReleased = 0
	isDashPressed = 0

	recover_sprite_scale()
	PlayerSprite.flip_h = (lastDirection < 0)
	
	

# ─────────────────────────────────────────────────────────
# INPUT HUMANO (MEZCLADO CON IA)
# ─────────────────────────────────────────────────────────
func get_input() -> void:
	# Movimiento humano
	var right_strength: float = Input.get_action_strength("right")
	var left_strength: float = Input.get_action_strength("left")
	var axis: float = right_strength - left_strength

	if axis > 0.0:
		_human_move = 1
	elif axis < 0.0:
		_human_move = -1
	else:
		_human_move = 0

	var final_move: int = _human_move if _human_move != 0 else _ai_move
	movementInput = final_move

	if movementInput != 0:
		lastDirection = movementInput

	# Jump humano
	if Input.is_action_just_pressed("jump"):
		isJumpPressed = 1
	if Input.is_action_just_released("jump"):
		isJumpReleased = 1

	# Coyote
	if jumpInput == 0 and isJumpPressed == 1:
		jumpInput = 1
		coyoteStartTime = Time.get_ticks_msec()

	elapsedCoyoteTime = Time.get_ticks_msec() - coyoteStartTime
	if jumpInput != 0 and elapsedCoyoteTime > coyoteDuration:
		jumpInput = 0

	# Dash humano
	if Input.is_action_just_pressed("dash"):
		isDashPressed = 1

# ─────────────────────────────────────────────────────────
# AUX FÍSICA + SPRITE
# ─────────────────────────────────────────────────────────
func apply_gravity(delta: float) -> void:
	if currentState != "dash":
		velocity.y += gravity * delta

func recover_sprite_scale() -> void:
	PlayerSprite.scale.x = move_toward(PlayerSprite.scale.x, 1.0, recoverySpeed)
	PlayerSprite.scale.y = move_toward(PlayerSprite.scale.y, 1.0, recoverySpeed)

func set_state(new_state: String) -> void:
	previousState = currentState
	currentState = new_state
	if previousState != "":
		call(previousState + "_exit_logic")
	if currentState != "":
		call(currentState + "_enter_logic")

func move_horizontally(_subtractor) -> void:
	currentSpeed = move_toward(currentSpeed, maxSpeed, acceleration)
	velocity.x = currentSpeed * movementInput

func squash_stretch(squash: float, stretch: float) -> void:
	PlayerSprite.scale = Vector2(squash, stretch)

func jump(jump_force: float) -> void:
	velocity.y = 0.0
	velocity.y = jump_force
	canDash = true
	squash_stretch(jumpingSquash, jumpingStretch)

# ─────────────────────────────────────────────────────────
# ESTADOS
# ─────────────────────────────────────────────────────────
func idle_enter_logic() -> void:
	Anim.play("Idle")

func idle_logic(_delta: float) -> void:
	if jumpInput:
		jump(jumpVelocity)
		set_state("jump")

	if isDashPressed:
		set_state("dash")

	if movementInput != 0:
		set_state("run")

	velocity.x = move_toward(velocity.x, 0.0, decceleration)

func idle_exit_logic() -> void:
	currentSpeed = 0.0

func run_enter_logic() -> void:
	Anim.play("Run")

func run_logic(_delta: float) -> void:
	if jumpInput:
		jump(jumpVelocity)
		set_state("jump")

	if isDashPressed:
		set_state("dash")

	if not is_on_floor():
		jumpBufferStartTime = Time.get_ticks_msec()
		set_state("fall")
		return

	if movementInput == 0:
		set_state("idle")
	else:
		move_horizontally(0)

func run_exit_logic() -> void:
	pass

func fall_enter_logic() -> void:
	Anim.play("Fall")

func fall_logic(_delta: float) -> void:
	move_horizontally(airFriction)
	elapsedJumpBuffer = Time.get_ticks_msec() - jumpBufferStartTime

	if isJumpPressed:
		if not isDoubleJumped and elapsedJumpBuffer > jumpBuffer:
			jump(doubleJumpVelocity)
			set_state("double_jump")
		if elapsedJumpBuffer < jumpBuffer:
			if previousState == "run":
				jump(jumpVelocity)
				set_state("jump")
			if previousState == "wall_slide":
				jump(wallJumpVelocity)
				set_state("wall_jump")

	if isDashPressed and canDash:
		set_state("dash")

	if is_on_floor():
		set_state("run")
		isDoubleJumped = false
		squash_stretch(landingSquash, landingStretch)

	if (LeftRaycast.is_colliding() and movementInput == -1) or (RightRaycast.is_colliding() and movementInput == 1):
		set_state("wall_slide")

func fall_exit_logic() -> void:
	jumpBufferStartTime = 0

func dash_enter_logic() -> void:
	dashDirection = lastDirection
	dashStartTime = Time.get_ticks_msec()
	velocity = Vector2.ZERO
	Anim.play("Idle")
	PlayerSprite.modulate = Color.PURPLE

func dash_logic(_delta: float) -> void:
	elapsedDashTime = Time.get_ticks_msec() - dashStartTime
	velocity.x += dashSpeed * dashDirection
	if elapsedDashTime > dashDurration:
		set_state(previousState)

func dash_exit_logic() -> void:
	velocity = Vector2.ZERO
	if not is_on_floor():
		canDash = false
	PlayerSprite.modulate = Color.WHITE

func jump_enter_logic() -> void:
	Anim.play("Jump")

func jump_logic(_delta: float) -> void:
	move_horizontally(airFriction)

	if velocity.y < 0.0:
		if isJumpReleased:
			velocity.y *= 0.5
		if isJumpPressed and not isDoubleJumped:
			jump(doubleJumpVelocity)
			set_state("double_jump")
		if isDashPressed and canDash:
			set_state("dash")
		if is_on_ceiling():
			set_state("fall")
	else:
		set_state("fall")

func jump_exit_logic() -> void:
	pass

func double_jump_enter_logic() -> void:
	isDoubleJumped = true
	Anim.play("Double Jump")

func double_jump_logic(_delta: float) -> void:
	move_horizontally(airFriction)

	if velocity.y < 0.0:
		if isJumpReleased:
			velocity.y *= 0.5
		if isDashPressed and canDash:
			set_state("dash")
		if is_on_ceiling():
			set_state("fall")
	else:
		set_state("fall")

func double_jump_exit_logic() -> void:
	pass

func wall_slide_enter_logic() -> void:
	velocity = Vector2.ZERO
	Anim.play("Wall Slide")

func wall_slide_logic(_delta: float) -> void:
	velocity.y = wallSlideSpeed

	if (LeftRaycast.is_colliding() and movementInput != -1) or (RightRaycast.is_colliding() and movementInput != 1):
		jumpBufferStartTime = Time.get_ticks_msec()
		set_state("fall")

	if (not LeftRaycast.is_colliding() and movementInput == -1) or (not RightRaycast.is_colliding() and movementInput == 1):
		set_state("fall")

	if is_on_floor():
		jumpBufferStartTime = Time.get_ticks_msec()
		set_state("idle")

	if isDashPressed:
		set_state("dash")

	if isJumpPressed:
		jump(wallJumpVelocity)
		set_state("wall_jump")

func wall_slide_exit_logic() -> void:
	isDoubleJumped = false

func wall_jump_enter_logic() -> void:
	currentSpeed = 0.0
	Anim.play("Jump")

func wall_jump_logic(_delta: float) -> void:
	move_horizontally(airFriction)

	if velocity.y < 0.0:
		if isJumpReleased:
			velocity.y *= 0.5
		if isJumpPressed and not isDoubleJumped:
			jump(doubleJumpVelocity)
			set_state("double_jump")
		if isDashPressed:
			set_state("dash")
		if is_on_ceiling():
			set_state("fall")
	else:
		set_state("fall")

func wall_jump_exit_logic() -> void:
	canDash = true

# VIDA / RESPAWN
func take_damage(amount: int = 1) -> void:
	hp -= amount
	if hp <= 0:
		die()

func die() -> void:
	emit_signal("died")
	respawn()

func set_respawn(pos: Vector2) -> void:
	respawn_point = pos

func respawn() -> void:
	hp = hp_max
	global_position = respawn_point
	velocity = Vector2.ZERO
	currentState = "idle"
	previousState = ""
	currentSpeed = 0.0
	is_dead = false

func on_goal_reached() -> void:
	pass
