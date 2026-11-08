"""
env_check.py - R1 확인용 노드: ROS2 패키지에서 저장소 모듈, 모델 라이브러리, 메시지를 모두 쓸 수 있는 지 확인하고 끝남
- 확인 내용
    1. 저장소 폴더를 찾고 detector / selector / smoother / flow_judges 를 불러옴
    2. torch (GPU), ultralytics, numpy, cv2 버전
    3. traffic_light_msgs/TrafficLightState 메시지를 만들어 /traffic_light/env_check 로 1번 발행

실행:
  ros2 run traffic_light_nodes env_check
  ros2 run traffic_light_nodes env_check --ros-args -p repo_dir:=~/project/traffic_light
"""

# ======================== ROS2 ===============================
import rclpy                                                        # ROS2 Python 클라이언트
from rclpy.node                     import Node                     # 노드 기본 클래스
from traffic_light_msgs.msg         import TrafficLightState        # R1에서 만든 메시지

# ===================== 패키지 모듈 =============================
from traffic_light_nodes.repo_path  import setup_repo_path          # 저장소 경로 설정

# =============================================================
# 확인 노드
# =============================================================
class EnvCheck(Node):
    def __init__(self):
        super().__init__("env_check")
        self.declare_parameter("repo_dir", "")                      # 빈 값이면 파일 위치에서 찾음
        repo = setup_repo_path(self.get_parameter("repo_dir").value)
        self.get_logger().info(f"repo: {repo}")

        import detector, selector, smoother, flow_judges            # 저장소 모듈 (경로 설정 뒤에 불러옴)
        self.get_logger().info("modules: " + ", ".join(m.__name__ for m in (detector, selector, smoother, flow_judges)))
        
        import numpy, cv2, torch, ultralytics                       # 모델 실행에 필요한 라이브러리
        self.get_logger().info(f"numpy {numpy.__version__} | cv2 {cv2.__version__} | torch {torch.__version__} "
                               f"cuda {torch.cuda.is_available()} | ultralytics { ultralytics.__version__}")

        pub = self.create_publisher(TrafficLightState, "/traffic_light/env_check", 10)
        msg = TrafficLightState()                                   # 시험용 메시지 1개
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.state, msg.conf = smoother.NONE, 0.0                    # smoother 의 상태 문자열 사용
        pub.publish(msg)
        self.get_logger().info(f"published TrafficLightState state={msg.state} -> env check OK")

# =============================================================
# 실행 진입 점 (확인 후 바로 종료)
# =============================================================
def main():
    rclpy.init()
    node = EnvCheck()                                               # 생성자에서 확인이 모두 끝남
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()