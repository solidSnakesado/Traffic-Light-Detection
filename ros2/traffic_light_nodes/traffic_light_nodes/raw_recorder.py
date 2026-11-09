"""
raw_recorder.py - R2 기록 노드: /traffic_light/raw (TrafficLightState) 를 받아 csv 로 저장
- csv 열은 tools/record_states.py 의 열 중 메시지에 있는 것만 (같은 이름, 같은 반올림)
    frame, t, state, n_det, n_cand, conf, x1, y1, x2, y2
    (cands, rejects, ms 는 메시지에 없어서 기록하지 않음)
- 첫 메시지를 받은 뒤 idle_sec 동안 메시지가 없으면 저장하고 종료 (0 = Ctrl+C 까지)
- 같은 프레임이 두 번 오면 그대로 두 줄 기록 (비교 도구 compare_states.py 가 마지막 줄을 씀)

실행 (인식 노드보다 먼저 또는 함께):
  ros2 run traffic_light_nodes raw_recorder --ros-args -p csv:=<저장할 csv 경로>
"""

# =================== 표준 라이브러리 ===========================
import csv                                                      # 결과를 csv 로 저장
import time                                                     # 마지막 수신 시각

from collections                import Counter                  # 상태별 개수
from pathlib                    import Path                     # 경로 다루기 (~ 확장)

# =================== ROS2 ====================================
import rclpy                                                    # ROS2 Python 클라이언트
from rclpy.node                 import Node                     # 노드 기본 클래스
from traffic_light_msgs.msg     import TrafficLightState        # 받을 인식 결과

FIELDS = ["frame", "t", "state", "n_det", "n_cand",             # csv 열 이름 (record_states.py 와 같은 이름)
          "conf", "x1", "y1", "x2", "y2"]

# =============================================================
# 기록 노드
# =============================================================
class RawRecorder(Node):
    def __init__(self):
        pass

    # ===== 메시지 1개 -> csv 한 줄 =====
    def on_raw(self, m):
        pass

    # ===== 종료 조건 확인 (main 반복에서 호출) =====
    def check_idle(self):
        pass

    # ===== 저장하고 요약 =====
    def close(self):
        pass

# =============================================================
# 실행 진입 점
# =============================================================
def main(args=None):
    pass

if __name__ == "__main__":
    main()