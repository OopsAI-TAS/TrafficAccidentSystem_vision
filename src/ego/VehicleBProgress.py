CLASSES = {
    "stopping":"0",
    "starting":"1",
    "left_turn":"2",
    "right_turn":"3",
    "straight":"4",
}
class VehicleBProgress:
    def __init__(self, fps, tau_deg=12, v_stop=0.15, dv_start=0.3, n_stop=10, n_start=6, cooldown=20):
        self.fps = fps
        self.tau = tau_deg
        self.v_stop = v_stop * fps/30.0
        self.dv_start = dv_start * fps/30.0
        self.n_stop = n_stop
        self.n_start = n_start
        self.cooldown = cooldown

        self.speed_hist = []
        self.last_state = "straight"
        self.last_emit = -9999

    def update(self, frame_idx, cam_state, phi_deg, lane_change_evt=None):
        """
        cam_state: {"yaw_deg": signed, "speed": ema translation(px)}
        phi_deg: 차로 주방향
        """
        if cam_state is None:  # 첫 프레임 등
            return None

        yaw = float(cam_state["yaw_deg"])     # 부호 있음
        v   = float(cam_state["speed"])
        self.speed_hist.append(v)
        if len(self.speed_hist) > 30:
            self.speed_hist.pop(0)

        # 1) stop / start
        stopping = False
        if len(self.speed_hist) >= self.n_stop and all(s < self.v_stop for s in self.speed_hist[-self.n_stop:]):
            stopping = True

        starting = False
        if len(self.speed_hist) >= self.n_start:
            # 최근 n_start 평균이 이전 n_start 평균보다 dv_start 이상 증가
            cur = sum(self.speed_hist[-self.n_start:]) / self.n_start
            prev = sum(self.speed_hist[-2*self.n_start:-self.n_start]) / max(1, self.n_start) if len(self.speed_hist)>=2*self.n_start else 0.0
            if (cur - prev) > self.dv_start and cur > self.v_stop:
                starting = True

        # 2) 회전 vs 차선방향
        delta = yaw - float(phi_deg)  # 좌(+)/우(-)
        # wrap to [-180,180]
        delta = ( (delta + 180) % 360 ) - 180

        # 3) 상태 결정 우선순위
        state = "straight"
        reason = {}

        if stopping:
            state = CLASSES.get("stopping")
            reason["stop_frames"] = self.n_stop
        elif starting:
            state = CLASSES.get("starting")
            reason["dv_start"] = round(cur - prev, 3)
        elif lane_change_evt is not None and frame_idx - self.last_emit >= self.cooldown:
            state = lane_change_evt["type"]
            reason["balance"] = lane_change_evt["balance"]
            self.last_emit = frame_idx
        elif delta > self.tau:
            state = CLASSES.get("left_turn")
            reason["delta_deg"] = round(delta, 2)
        elif delta < -self.tau:
            state = CLASSES.get("right_turn")
            reason["delta_deg"] = round(delta, 2)
        else:
            state = CLASSES.get("straight")
            reason["delta_deg"] = round(delta, 2)

        self.last_state = state
        return {
            "pred": state,
            "theta_deg": round(float(yaw), 2),
            "phi_deg": round(float(phi_deg), 2),
            "speed_proxy": round(v, 3),
            "reason": reason
        }
