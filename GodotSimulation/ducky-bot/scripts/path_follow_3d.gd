extends PathFollow3D

@export var speed: float = 0.2
# When false, the follower drives to the end of the path and stops there
# (instead of looping back to the start). Default true keeps other maps' NPCs
# looping; set it false in the Inspector for an NPC that should stop at the end.
@export var loop_path: bool = true
var running: bool = true

func _ready() -> void:
	rotation_mode = PathFollow3D.ROTATION_Y
	loop = loop_path

func _process(delta: float) -> void:
	if not running:
		return
	progress += speed * delta
	# reached the end of a non-looping path -> clamp and stop
	if not loop_path and progress_ratio >= 1.0:
		progress_ratio = 1.0
		running = false
