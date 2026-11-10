"""
smoother_node.py - R3 안정화 노드: /traffic_light/raw (원시 상태) -> smoother.py -> /traffic_light/state (확정 상태)
- 받은 메시지마다 StateSmoother.update(영상 시각, 원시 상태) 를 호출 (tools/replay_smoother.py 와 같은 계산)
- 발행 메시지는 받은 메시지를 그대로 복사하고 state 만 확정 상태로 바꿈 (frame, video_time, 박스 등은 그대로)
- 확정 상태가 바뀔 때마다 로그 (프레임, 영상 시각, 이전 -> 새 상태)
- 영상 시각이 뒤로 가면 (새 영상 / 다시 시작) smoother 를 처음 상태로 되돌림
- 기준값은 smoother.py 시작값이 기본 (파라미터로 바꿀 수 있음), fps = 실제 처리 fps (최소 관측 수 계산용)

실행 (작업 공간 적용 후, 인식 노드와 함께):
  ros2 run traffic_light_nodes smoother_node
  기준값 바꾸기: --ros-args -p hold:=1.0 -p red_hold:=true
"""

# =================== ROS2 ====================================
import rclpy                                                                            # ROS2 Python 클라이언트
from rclpy.node                     import Node                                         # 노드 기본 클래스
from traffic_light_msgs.msg         import TrafficLightState                            # 받고 보내는 메시지 (원시 / 확정 공용)

# ===================== 패키지 모듈 ============================
from traffic_light_nodes.repo_path  import setup_repo_path                              # 저장소 경로 설정

# =============================================================
# 안정화 노드
# =============================================================
class SmootherNode(Node):
    def __init__(self):
        super().__init__("smoother_node")
        defaults = dict(raw_topic="/traffic_light/raw", state_topic="/traffic_light/state",
                        repo_dir="", fps=60.0,                                          # fps =  실제 처리 fps (영상 전체 처리면 영상 fps)
                        t_stop=0.2, t_go=0.5, ratio=0.7, min_obs=0.3,                   # smoother.py 시작값 (2026-11-01 결정)
                        hold=1.5, red_hold=False)
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value                                 # 파라미터 값 읽기

        setup_repo_path(p("repo_dir"))                                                  # 저장소의 smoother.py 를 불러올 수 있게
        from smoother import StateSmoother, SmootherConfig                              # 4. 안정화 모듈 (경로 설정 뒤에 불러옴)
        cfg             = SmootherConfig(t_stop=p("t_stop"), t_go=p("t_go"), ratio=p("ratio"), min_obs=p("min_obs"),
                                         hold=p("hold"), fps=p("fps"), red_hold=bool(p("red_hold")))
        self.smoother   = StateSmoother(cfg)
        self.pub        = self.create_publisher(TrafficLightState, p("state_topic"), 100)
        self.create_subscription(TrafficLightState, p("raw_topic"), self.on_raw, 100)   # 넉넉히 보관 (놓치지 않게) 
        self.n          = 0                                                             # 받은 메시지 수
        self.changes    = 0                                                             # 확정 상태가 바뀐 횟수
        self.last_t     = None                                                          # 마지막 영상 시각 (뒤로 가면 다시 시작)
        self.get_logger().info(f"{p('raw_topic')} -> {p('state_topic')}, " + ", ".join(
            f"{k} {getattr(cfg, k)}" for k in ("t_stop", "t_go", "ratio", "min_obs", "hold", "fps", "red_hold")))

    # ===== 원시 상태 1개 -> 확정 상태 발행 =====
    def on_raw(self, m):
        if self.last_t is not None and m.video_time < self.last_t:                      # 영상 시각이 뒤로 감 = 새로 시작
            self.get_logger().info(f"video time went back ({self.last_t:.2f} -> {m.video_time:.2f} s) -> reset")
            self.summary()
            self.smoother.reset()
            self.n, self.changes = 0, 0
        self.last_t     = m.video_time
        prev            = self.smoother.state
        res             = self.smoother.update(m.video_time, m.state)                   # 4. 안정화 (영상 시각 기준)
        m.state         = res.state                                                     # 받은 메시지에서 state 만 확정 상태로
        m.header.stamp  = self.get_clock().now().to_msg()                               # 발행 시각
        self.pub.publish(m)
        self.n += 1
        if res.changed:                                                                 # 확정 상태가 바뀜
            self.changes += 1
            self.get_logger().info(f"frame {m.frame} ({m.video_time:.2f} s): {prev} -> {res.state}")

    # ===== 요약 (종료 / 다시 시작할 때) =====
    def summary(self):
        self.get_logger().info(f"frames {self.n}, confirmed changes {self.changes}")

# =============================================================
# 실행 진입 점 (Ctrl+C 까지)
# =============================================================
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SmootherNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.summary()
            node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()