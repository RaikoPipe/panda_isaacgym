from dataclasses import dataclass
from dataclasses import field

@dataclass
class PoseHandlerConfig:

    """Configuration for the pose generator."""

    range_radius: tuple[float, float] = (0.1, 0.5)
    """The range of the radius for the hollow sphere."""

    range_euler: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        'roll': (0.0, 0.0),
        'pitch': (-3.14, 3.14),
        'yaw': (-3.14, 3.14)
    })

    # whether to convert the quaternion to a standard form where the real part is non-negative
    make_quat_unique: bool = False

