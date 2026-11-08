"""
perception_core.py - 인식 처리(ROS2와 무관): 프레임 1장 -> 1. 검출 -> 3. 판별 -> 2. 대상선택
- tools/record_states.py 와 같은 모듈, 같은 순서로 처리 -> 같은 영상이면 같은 결과
- perception_node.py 가 사용 (영상 직접 읽기 / 영상 토픽 구독 둘 다 이 클래스 사용)
- fill_msg: 처리 결과를 TrafficLightState 메시지 필드에 채움 (record_states.py 의 csv 열과 같은 값)
"""

# =================== 표준 라이브러리 ===========================
import time                                                     # 프레임 처리 시간 측정

# ===================== 패키지 모듈 ============================
from traffic_light_nodes.repo_path import setup_repo_path       # 저장소 경로 설정

# =============================================================
# 인식기: 모델을 한 번 불러 두고 프레임마다 process() 호출
# =============================================================
class Perception:
    def __init__(self, repo_dir="", model="", cls_model="", det_classes=None,
                 imgsz=1280, cls_imgsz=224, conf=0.25, device=None,
                 min_aspect=1.5, max_cy=0.55, min_area=300):
        pass

    def process(self, frame):
        pass

# =============================================================
# 처리 결과 -> TrafficLightState 필드 (header.stamp 는 노드에서 채움)
# =============================================================
def fill_msg(msg, frame, video_time, dets, res):
    pass