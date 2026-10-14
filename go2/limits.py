"""Per-tool-call motion limits. Several calls in sequence may go further in total."""
import math

MAX_SPEED_MPS = 0.8
MAX_DURATION_S = 5.0
MAX_DISTANCE_M = 4.0
MAX_TURN_RATE_RPS = 1.0
MAX_TURN_RATE_DPS = math.degrees(MAX_TURN_RATE_RPS)          # ~57.3 °/s
MAX_TURN_DEG = MAX_TURN_RATE_DPS * MAX_DURATION_S             # ~286° in one call

DEFAULT_SPEED_MPS = 0.2       # minimum default; distance moves use max(0.2, distance / 5 s)
DEFAULT_TURN_RATE_RPS = 0.5   # minimum default; angle turns use max(0.5, angle / 5 s)

MAX_BODY_ROLL_DEG = 43.0      # SportClient Euler limits: roll/pitch ±0.75 rad, yaw ±0.6 rad
MAX_BODY_PITCH_DEG = 43.0
MAX_BODY_YAW_DEG = 34.0

MAX_WAIT_S = 30.0
MAX_GOTO_LEG_M = MAX_DISTANCE_M
CONFIRMATION_TTL_S = 60.0


def default_speed(distance_m=None):
    """Speed when the user gave none: the gentle default, raised just enough to finish in 5 s."""
    if distance_m is None:
        return DEFAULT_SPEED_MPS
    return min(MAX_SPEED_MPS, max(DEFAULT_SPEED_MPS, distance_m / MAX_DURATION_S))


def default_turn_rate(angle_rad=None):
    if angle_rad is None:
        return DEFAULT_TURN_RATE_RPS
    return min(MAX_TURN_RATE_RPS, max(DEFAULT_TURN_RATE_RPS, angle_rad / MAX_DURATION_S))
