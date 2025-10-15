import numpy as np
from collections import defaultdict, deque

def bbox_footpoint(box):
    x1,y1,x2,y2 = box
    cx = 0.5*(x1+x2); fy = y2
    return cx, fy

def safe_int_clip(x, lo, hi):
    return max(lo, min(int(x), hi))

class LaneChangeEngine:
    """
    각 트랙의 lane_id를 프레임별 기록 → 안정화 → 변경 감지
    lane_labels: connectedComponents 결과 (lane=1/0 이 아니라 라벨 맵)
    """
    def __init__(self, persist=8, stable_min=3, cool=20):
        self.history = {}  # tid -> deque of lane_ids
        self.persist = persist
        self.stable_min = stable_min
        self.cool = cool
        self.last_emit = defaultdict(lambda:-9999)
        self.curr_lane = {}  # tid -> stable lane id

    def _lane_at(self, lane_labels, x, y):
        if lane_labels is None: return -1
        H, W = lane_labels.shape[:2]
        xi = safe_int_clip(x, 0, W-1); yi = safe_int_clip(y, 0, H-1)
        return int(lane_labels[yi, xi])

    def update_tracks(self, frame_idx, tracks, lane_labels):
        events = []
        for t in tracks:
            fx, fy = bbox_footpoint(t.bbox)
            lid = self._lane_at(lane_labels, fx, fy)

            dq = self.history.setdefault(t.tid, deque(maxlen=self.persist))
            dq.append(lid)

            # 안정화된 현재 레인(최빈값)
            if len(dq) >= self.stable_min:
                vals, cnts = np.unique(list(dq), return_counts=True)
                stable = int(vals[np.argmax(cnts)])
            else:
                stable = lid

            prev = self.curr_lane.get(t.tid, stable)
            self.curr_lane[t.tid] = stable
            t.lane_id = stable  # 디버그용

            # 변경 감지 (이전 안정 레인과 다르고, -1 제외, 쿨다운)
            if prev != stable and prev!=-1 and stable!=-1:
                if frame_idx - self.last_emit[t.tid] >= self.cool:
                    events.append({"type":"lane_change", "id":int(t.tid),
                                   "from": int(prev), "to": int(stable), "frame": int(frame_idx)})
                    self.last_emit[t.tid] = frame_idx
        return events