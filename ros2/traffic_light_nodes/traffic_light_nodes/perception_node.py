"""
perception_node.py - R2 인식 노드: 영상 -> 1. 검출 -> 3. 판별 -> 2. 대상 선택 -> TrafficLightState (/traffic_light/raw)
- 입력 2가지
    source 지정 : 영상 파일 (또는 카메라 번호) 을 직접 읽음 (기본, 큰 영상 메시지를 보내지 않음)
    source 빈 값: image_topic (sensor_msgs/Image) 구독 (확장용: 다른 장치의 카메라 영상)
- 직접 읽기의 pace
    all     : 모든 프레임을 순서대로 처리 (record_states.py 와 비교 검증용)
    realtime: 영상 시각에 맞춰 처리, 늦으면 프레임을 건너뜀 (주행용)
- wait_sub true 면 /traffic_light/raw 구독자가 생길 때까지 기다린 뒤 시작 (기록 노드가 첫 프레임을 놓치지 않게)

실행 (작업 공간 적용 후):
  ros2 run traffic_light_nodes perception_node --ros-args -p source:=<영상 경로> \
      -p model:=<검출 best.pt 또는 .engine> -p cls_model:=<분류 best.pt 또는 .engine>
"""

# =================== 표준 라이브러리 ===========================
import time                                                                 # 실시간 처리 간격, 대기
from collections                            import Counter                  # 상태별 프레임 수
from pathlib                                import Path                     # 경로 다루기 (~ 확장)

# =================== 외부 라이브러리 ===========================
import cv2                                                                  # 영상 읽기

# =================== ROS2 ====================================
import rclpy                                                                # ROS2 Python 클라이언트
from rclpy.node                             import Node                     # 노드 기본 클래스
from sensor_msgs.msg                        import Image                    # 구독 모드 입력
from traffic_light_msgs.msg                 import TrafficLightState        # 발행할 인식 결과

# =================== 패키지 모듈 ===============================
from traffic_light_nodes.perception_core    import Perception, fill_msg     # 인식 처리 (ROS2 와 무관)

PACES = ("all", "realtime")                                                 # 사용할 수 있는 pace 값

# =============================================================
# 인식 노드
# =============================================================
class PerceptionNode(Node):
    def __init__(self):
        pass

    # ===== 직접 읽기: 프레임 1장 처리 (main 반복해서 호출) =====
    def step(self):
        pass

    # ===== 구독 모드: 영상 메시지 1개 처리 =====
    def on_image(self, msg):
        pass

    # ===== 처리 + 발행 + 통계 =====
    def handle(self, i, t, frame):
        pass

    # ===== 종료 요약 (record_states.py 와 같은 항목) =====
    def finish(self, reason):
        pass 

# =============================================================
# 실행 진입 점 (직접 읽기는 영상이 끝나면 종료, 구독 모드는 Ctrl+C 까지)
# =============================================================
def main(args=None):
    pass

if __name__ == "__main__":
    main()