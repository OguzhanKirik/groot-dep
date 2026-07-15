"""Small, dependency-light episode recorder for Adam-U simulation teleoperation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class AdamUTeleopRecorder:
    """Accumulate synchronized samples and write successful episodes to HDF5."""

    output_path: str | Path
    task: str
    control_dt: float
    _samples: dict[str, list[np.ndarray]] = field(default_factory=dict, init=False)
    _episode_index: int = field(default=0, init=False)

    REQUIRED_WIDTHS = {
        "observation.state.body": 19,
        "observation.state.hands": 12,
        "action.body": 19,
        "action.hands": 12,
        "observation.right_wrist_pose": 9,
        "observation.object_pose": 7,
    }

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path).expanduser().resolve()
        if self.control_dt <= 0:
            raise ValueError("control_dt must be positive")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            import h5py

            with h5py.File(self.output_path, "r") as handle:
                self._episode_index = len(handle.get("episodes", {}))

    @property
    def sample_count(self) -> int:
        return len(next(iter(self._samples.values()), []))

    def clear(self) -> None:
        self._samples.clear()

    def append(self, sample: dict[str, Any]) -> None:
        converted = {key: np.asarray(value) for key, value in sample.items()}
        missing = set(self.REQUIRED_WIDTHS) - set(converted)
        if missing:
            raise ValueError(f"Teleop sample is missing required fields: {sorted(missing)}")
        for key, width in self.REQUIRED_WIDTHS.items():
            value = converted[key]
            if value.shape != (width,) or not np.isfinite(value).all():
                raise ValueError(f"{key} must be finite with shape ({width},), got {value.shape}")
        for key, value in converted.items():
            if key.startswith("observation.images."):
                if value.ndim != 3 or value.shape[-1] not in (3, 4):
                    raise ValueError(f"{key} must have shape (H, W, 3/4), got {value.shape}")
                value = value[..., :3].astype(np.uint8, copy=False)
            elif not np.issubdtype(value.dtype, np.number):
                raise ValueError(f"Unsupported non-numeric sample field: {key}")
            self._samples.setdefault(key, []).append(value.copy())
        lengths = {len(values) for values in self._samples.values()}
        if len(lengths) != 1:
            raise RuntimeError("All recorded fields must have the same number of samples")

    def save_success(self) -> str:
        if self.sample_count == 0:
            raise ValueError("Cannot save an empty teleoperation episode")
        import h5py

        episode_name = f"episode_{self._episode_index:06d}"
        with h5py.File(self.output_path, "a") as handle:
            episodes = handle.require_group("episodes")
            if episode_name in episodes:
                raise RuntimeError(f"Episode already exists: {episode_name}")
            episode = episodes.create_group(episode_name)
            episode.attrs["task"] = self.task
            episode.attrs["success"] = True
            episode.attrs["control_dt"] = self.control_dt
            episode.attrs["num_samples"] = self.sample_count
            for key, values in self._samples.items():
                array = np.stack(values)
                kwargs = {"compression": "gzip", "compression_opts": 4} if array.ndim >= 3 else {}
                episode.create_dataset(key, data=array, **kwargs)
        self._episode_index += 1
        self.clear()
        return episode_name

