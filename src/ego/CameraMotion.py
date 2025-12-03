import cv2, math

class CameraMotion:
    """
    프레임간 전역 변환(회전+평행이동) 추정 → 흔들림 스코어 계산
    """
    def __init__(self, fps, ema=0.6, trans_thr=3.5, rot_thr_deg=1.2, cool=10):
        self.prev_gray = None
        self.fps = fps
        self.ema = ema
        self.cool = cool
        self.last_emit = -9999
        self.state = {"t_ema":0.0, "r_ema":0.0, "yaw_deg": 0.0, "speed":0.0}  
        self.trans_thr = trans_thr * fps/30.0     # fps 보정
        self.rot_thr   = rot_thr_deg * fps/30.0   # fps 보정

    def update(self, frame_idx, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return None, None  # 초기 프레임

        # 특징점 추적
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=8)
        if p0 is None or len(p0)<20:
            self.prev_gray = gray
            return None, getattr(self, "state", {})
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None,
                                               winSize=(21,21), maxLevel=3,
                                               criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        good0 = p0[st[:,0]==1]; good1 = p1[st[:,0]==1]
        self.prev_gray = gray
        if len(good0)<15:
            self.prev_gray = gray
            return None, getattr(self, "state", {})

        # 전역 변환 추정 (유사변환: 회전+스케일+이동)
        M, inliers = cv2.estimateAffinePartial2D(good0, good1, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, refineIters=15)
        if M is None:
            return None, getattr(self, "state", {})

        dx, dy = M[0,2], M[1,2]
        # 회전 추정 (scale 무시, 각도만)
        yaw = math.degrees(math.atan2(M[1,0], M[0,0]))
        tmag = math.hypot(dx, dy)

        prev_yaw = 0.0 if not hasattr(self, "state") else float(self.state.get("yaw_deg", 0.0))
        dyaw = yaw - prev_yaw   
        # EMA
        self.state["t_ema"] = self.ema*self.state["t_ema"] + (1-self.ema)*math.hypot(dx,dy)
        self.state["r_ema"] = self.ema*self.state["r_ema"] + (1-self.ema)*abs(dyaw)
        self.state["yaw_deg"] = yaw
        self.state["speed"] = self.state["t_ema"]

        # 이벤트 판정
        evt = None
        if (self.state["t_ema"] >= self.trans_thr or self.state["r_ema"] >= self.rot_thr):
            if frame_idx - self.last_emit >= self.cool:
                evt = {"type":"camera_shake",
                       "t_px": float(self.state["t_ema"]),
                       "r_deg": float(self.state["r_ema"]),
                       "frame": int(frame_idx)}
                self.last_emit = frame_idx
        return evt, dict(self.state)