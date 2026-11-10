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
import csv                                                                              # 결과를 csv 로 저장
import time                                                                             # 마지막 수신 시각

from collections                import Counter                                          # 상태별 개수
from pathlib                    import Path                                             # 경로 다루기 (~ 확장)

# =================== ROS2 ====================================
import rclpy                                                                            # ROS2 Python 클라이언트
from rclpy.node                 import Node                                             # 노드 기본 클래스
from traffic_light_msgs.msg     import TrafficLightState                                # 받을 인식 결과

FIELDS = ["frame", "t", "state", "n_det", "n_cand",                                     # csv 열 이름 (record_states.py 와 같은 이름)
          "conf", "x1", "y1", "x2", "y2"]

# =============================================================
# 기록 노드
# =============================================================
class RawRecorder(Node):
    def __init__(self):
        super().__init__("raw_recorder")
        self.declare_parameter("csv", "")                                               # 저장할 csv 경로
        self.declare_parameter("raw_topic", "/traffic_light/raw")                       # 받을 토픽
        self.declare_parameter("idle_sec", 3.0)                                         # 이 시간 동안 메시지가 없으면 종료
        p = lambda name: self.get_parameter(name).value
        if not p("csv"):
            raise SystemExit("csv 파라미터가 필요합니다 (-p csv:=<저장할 csv 경로>)")
        self.path   = Path(p("csv")).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)                             # 저장 폴더가 없으면 생성
        self.file   = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
        self.writer.writeheader()                                                       # 첫 줄 = 열 이름
        self.idle   = float(p("idle_sec"))
        self.n      = 0                                                                 # 받은 메시지 수
        self.states = Counter()                                                         # 상태별 개수
        self.last   = None                                                              # 마지막 수신 시각 (monotonic)
        self.done   = False
        self.create_subscription(TrafficLightState, p("raw_topic"), self.on_raw, 100)   # 넉넉히 보관 (놓치지 않게)
        self.get_logger().info(f"{p('raw_topic')} -> {self.path} (idle {self.idle:.0f} s)")

    # ===== 메시지 1개 -> csv 한 줄 =====
    def on_raw(self, m):
        has = m.n_cand > 0                                                              # 후보가 있으면 대상 박스도 있음 (없으면 빈 값, record_states 와 같음)
        box = [m.x1, m.y1, m.x2, m.y2] if has else ["", "", "", ""]
        self.writer.writerow(dict(zip(FIELDS, [
            m.frame, round(m.video_time, 4), m.state, m.n_det, m.n_cand,
            round(m.conf, 3) if has else "", *box])))
        self.n += 1
        self.states[m.state] += 1
        self.last = time.monotonic()
        if self.n % 300 == 0:                                                           # 300개 마다 진행 상황
            self.get_logger().info(f"{self.n} messages (frame {m.frame})")

    # ===== 종료 조건 확인 (main 반복에서 호출) =====
    def check_idle(self):
        if self.idle > 0 and self.last is not None and time.monotonic() - self.last > self.idle:
            self.done = True

    # ===== 저장하고 요약 =====
    def close(self):
        self.file.close()
        self.get_logger().info(f"saved {self.n} rows -> {self.path}")
        self.get_logger().info("state: " + ", ".join(f"{k} {v}" for k, v in self.states.most_common()))

# =============================================================
# 실행 진입 점
# =============================================================
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RawRecorder()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
            node.check_idle()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()