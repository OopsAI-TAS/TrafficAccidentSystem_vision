from typing import Dict, Any, List
from src.types import TrackState

def to_json_record(frame_idx, track, events):
    return {
        "frame": int(frame_idx),
        "track_id": int(track.tid),
        "class": track.cls,
        "bbox": [float(x) for x in track.bbox],
        "center": [float(track.center[0]), float(track.center[1])],
        "speed_px_s": float(track.speed),
        "flags": {
            "on_road": bool(getattr(track, "on_road", False)),
            "on_lane": bool(getattr(track, "on_lane", False))
        },
        # 이벤트는 그대로 넣기 (conf 강제 안 함)
        "events": [dict(e) for e in events]
    }
