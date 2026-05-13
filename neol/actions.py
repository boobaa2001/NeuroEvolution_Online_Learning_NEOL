import numpy as np


def to_env_action(acts, action_space):
    from gymnasium import spaces

    acts = np.asarray(acts, dtype=np.float32).reshape(-1)

    if isinstance(action_space, spaces.Box):
        expected = int(np.prod(action_space.shape))

        if acts.size != expected:
            raise ValueError(
                f"Network produced {acts.size} outputs, "
                f"but Box action space expects {expected}: {action_space}"
            )

        arr = acts.reshape(action_space.shape)
        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)

        finite_bounds = np.isfinite(low) & np.isfinite(high)
        scaled = np.where(
            finite_bounds,
            low + (arr + 1.0) * 0.5 * (high - low),
            arr,
        )
        return np.clip(scaled, action_space.low, action_space.high)

    if isinstance(action_space, spaces.Discrete):
        if acts.size != action_space.n:
            raise ValueError(
                f"Network produced {acts.size} outputs, "
                f"but Discrete action space expects {action_space.n}: "
                f"{action_space}"
            )

        start = getattr(action_space, "start", 0)
        return int(np.argmax(acts)) + int(start)

    raise NotImplementedError(f"Unsupported action space: {action_space}")
