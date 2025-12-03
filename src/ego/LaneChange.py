import numpy as np

class LaneChange:
    def __init__(self, persist=8, cooldown=20, delta_thr=0.08):
        self.persist = persist
        self.cooldown = cooldown
        self.delta_thr = delta_thr
        self.hist = []         # 최근 balance 값들
        self.last_change = -9999
        self.prev_sign = None

    def update(self, frame_idx, drivable_mask):
        h, w = drivable_mask.shape[:2]
        m = (drivable_mask>0).astype(np.uint8)
        left = m[:, :w//2].sum(dtype=np.float64)
        right= m[:, w//2:].sum(dtype=np.float64)
        total = left + right + 1e-6
        balance = (right - left) / total   # (+) 오른쪽 점유>왼쪽, (-) 반대
        self.hist.append(balance)
        if len(self.hist) > max(2*self.persist, 30):
            self.hist.pop(0)

        # 충분한 변화가 누적된 구간만 체크
        if len(self.hist) < self.persist:
            return None

        recent = self.hist[-self.persist:]
        mean_bal = sum(recent)/len(recent)
        cur_sign = 1 if mean_bal > +self.delta_thr else (-1 if mean_bal < -self.delta_thr else 0)

        # sign이 0에서 벗어나고, 이전과 반대 부호로 충분히 유지되면 lane change
        evt = None
        if self.prev_sign is not None and cur_sign!=0 and cur_sign != self.prev_sign:
            if frame_idx - self.last_change >= self.cooldown:
                direction = "R" if cur_sign>0 else "L"  # 오른쪽 점유↑ → 우측 차선으로 이동
                evt = {"type": f"lane_change_{direction}",
                       "frame": int(frame_idx),
                       "balance": float(mean_bal)}
                self.last_change = frame_idx

        if cur_sign != 0:
            self.prev_sign = cur_sign

        return evt
