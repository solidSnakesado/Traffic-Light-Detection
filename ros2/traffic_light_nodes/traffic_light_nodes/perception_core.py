"""
perception_core.py - 인식 처리(ROS2와 무관): 프레임 1장 -> 1. 검출 -> 3. 판별 -> 2. 대상선택
- tools/record_states.py 와 같은 모듈, 같은 순서로 처리 -> 같은 영상이면 같은 결과
- perception_node.py 가 사용 (영상 직접 읽기 / 영상 토픽 구독 둘 다 이 클래스 사용)
- fill_msg: 처리 결과를 TrafficLightState 메시지 필드에 채움 (record_states.py 의 csv 열과 같은 값)
"""

# =================== 표준 라이브러리 ===========================
import time                                                                             # 프레임 처리 시간 측정

# ===================== 패키지 모듈 ============================
from traffic_light_nodes.repo_path import setup_repo_path                               # 저장소 경로 설정

# =============================================================
# 인식기: 모델을 한 번 불러 두고 프레임마다 process() 호출
# =============================================================
class Perception:
    def __init__(self, repo_dir="", model="", cls_model="", det_classes=None,
                 imgsz=1280, cls_imgsz=224, conf=0.25, device=None,
                 min_aspect=1.5, max_cy=0.55, min_area=300):
        self.repo       = setup_repo_path(repo_dir)                                     # 저장소 모듈을 불러올 수 있게 경로 추가
        from detector       import TrafficLightDetector                                 # 1. 검출기 (경로 설정 뒤에 불러옴)
        from selector       import TargetSelector, SelectorConfig                       # 2. 대상 선택기, 기준값
        from flow_judges    import HsvJudge, ModelJudge                                 # 3. 판별기 (record_states.py 와 같은 코드)

        self.detector   = TrafficLightDetector(model_path=model, conf=conf, imgsz=imgsz, 
                                               target_classes=det_classes or None,      # 빈 값이면 모든 클래스
                                               device=device)
        cfg             = SelectorConfig(min_aspect=min_aspect, max_center_y=max_cy, min_area=min_area)
        self.selector   = TargetSelector(cfg)                                           # edge_margin 등 나머지는 기본값
        self.judges     = ModelJudge(cls_model, cls_imgsz, device) if cls_model else HsvJudge()
        self.name       = self.judges.name                                              # 판별기 이름 (로그 표시용)

    def process(self, frame):
        """BGR 프레임 1장 -> (검출 목록, 선택 결과, 처리 시간 ms)"""
        t0      = time.perf_counter()
        dets    = self.detector.detect(frame)                                           # 1. 검출
        res     = self.selector.select(dets, frame.shape, self.judges.bind(frame))      # 3. 후보 판별 + 2. 대상 1개 선택
        return dets, res, (time.perf_counter() - t0) * 1000

# =============================================================
# 처리 결과 -> TrafficLightState 필드 (header.stamp 는 노드에서 채움)
# =============================================================
def fill_msg(msg, frame, video_time, dets, res):
    d               = res.target                                                        # 대상 박스 (없으면 None)
    msg.frame       = int(frame)                                                        # 영상 프레임 번호
    msg.video_time  = float(video_time)                                                 # 영상 시각 (초)
    msg.state       = res.target_state                                                  # 원시 상태 (6개 /UNKNOWN / NONE)
    msg.conf        = float(d.conf) if d is not None else 0.0                           # 대상이 없으면 0
    msg.x1, msg.y1  = (int(d.x1), int(d.y1)) if d is not None else (0, 0)               # 대상 박스 좌표 (없으면 0)
    msg.x2, msg.y2  = (int(d.x2), int(d.y2)) if d is not None else (0, 0)               
    msg.n_det       = len(dets)                                                         # 검출 박스 수
    msg.n_cand      = len(res.candidates)                                               # selector 1단계 통과 후보 수
    return msg