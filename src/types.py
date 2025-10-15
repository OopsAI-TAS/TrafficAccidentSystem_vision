from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import numpy as np

BBox = Tuple[float, float, float, float]

@dataclass
class TrackState:
    tid: int
    cls: str
    bbox: BBox
    center: Tuple[int, int]
    speed: float = 0.0
    heading: float = 0.0
    lane_id: int = -1
    flags: Dict[str, bool] = field(default_factory=dict)
    history: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class FrameState:
    idx: int
    H: int
    W: int
    road_mask: np.ndarray
    lane_mask: np.ndarray
    lane_labels: Optional[np.ndarray] = None
    signal_state: Optional[str] = None
    