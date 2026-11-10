"""
video_source.py - R2 영상 노드: 영상 파일 (또는 카메라) 을 한 프레임씩 sensor_msgs/Image 로 발행
- header.frame_id   = 프레임 번호 문자열 (0부터)
- header.stamp      = 영상 시각 (프레임 번호 / 영상 FPS, record_states.py 와 같은 시각), 카메라는 현재 시각
- mode
    lockstep: 프레임 N 을 보낸 뒤 /traffic_light/raw 에 frame == N 이 오면 N+1 발행 (누락 없음, 비교 검증용)
              wait_sec 동안 응답이 없으면 같은 프레임을 다시 발행
    realtime: 영상 FPS 주기로 계속 발행 (인식이 늦으면 인식 노드가 오래된 프레임을 버림, 주행용)
- 영상이 끝나거나 limit 프레임을 보내면 종료

실행 (작업 공간 적용 후):
  ros2 run traffic_light_nodes video_source --ros-args -p source:=<영상 경로> -p mode:=lockstep
  카메라: -p source:=0 -p mode:=realtime
"""

# =================== 표준 라이브러리 ===========================
import time                                                                     # 발행 간격, 응답 대기 시간 측정

from pathlib                    import Path                                     # 경로 다루기 (~ 확장)

# =================== 외부 라이브러리 ===========================
import cv2                                                                      # 영상 읽기

# =================== ROS2 ====================================
import rclpy                                                                    # ROS2 Python 클라이언트
from rclpy.node                 import Node                                     # 노드 기본 클래스
from builtin_interfaces.msg     import Time                                     # header.stamp 형식 (초 + 나노초)
from cv_bridge                  import CvBridge                                 # OpenCV 이미지 -> sensor_msgs/Image
from sensor_msgs.msg            import Image                                    # 발행할 영상 메시지
from traffic_light_msgs.msg     import TrafficLightState                        # lockstep 응답 (인식 결과)

MODES = ("lockstep", "realtime")                                                # 사용할 수 있는 mode 값

# =============================================================
# 영상 시각 (초) -> header.stamp
# =============================================================
def to_stamp(t):
    ns = int(round(t * 1e9))                                                    # 나노초 정수로 바꿔서 나눔 (반올림 오차 방지)
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)

# =============================================================
# 영상 노드
# =============================================================
class VideoSource(Node):
    def __init__(self):
        super().__init__("video_source")
        self.declare_parameter("source", "")                                    # 영상 경로 또는 카메라 번호 (예: 0)
        self.declare_parameter("mode", "lockstep")                              # lockstep / realtime
        self.declare_parameter("limit", 0)                                      # 보낼 프레임 수 (0 = 전부, 시험용)
        self.declare_parameter("fps", 0.0)                                      # 발행 FPS (0 = 영상 FPS, 카메라는 30)
        self.declare_parameter("wait_sec", 10.0)                                # lockstep 응답 대기 (첫 프레임은 모델 준비로 느림)
        self.declare_parameter("image_topic", "/camera/image_raw")              # 발행 토픽
        self.declare_parameter("raw_topic", "/traffic_light/raw")               # lockstep 응답 토픽
        p           = lambda name: self.get_parameter(name).value               # 파라미터 값 읽기
        src         = str(p("source"))
        self.mode   = p("mode")
        self.limit  = int(p("limit"))
        if not src:
            raise SystemExit("source 파라미터가 필요합니다 (-p source:=<영상 경로 또는 카메라 번호>)")
        if self.mode not in MODES:
            raise SystemExit(f"mode 는 {', '.join(MODES)} 중 하나: {self.mode}")

        self.is_camera  = src.isdigit()                                         # 숫자만 있으면 카메라 번호
        self.cap        = cv2.VideoCapture(int(src) if self.is_camera else str(Path(src).expanduser()))
        if not self.cap.isOpened():
            raise SystemExit(f"영상을 열 수 없습니다: {src}")
        video_fps       = 0.0 if self.is_camera else self.cap.get(cv2.CAP_PROP_FPS)
        self.fps        = float(p("fps")) or video_fps or 30.0                  # 지정 값 -> 영상 FPS -> 30 순서
        self.wait_sec   = float(p("wait_sec"))
        self.bridge     = CvBridge()
        self.pub        = self.create_publisher(Image, p("image_topic"), 1)     # 최신 1장만 보관

        self.i          = 0                                                     # 다음에 보낼 프레임 번호
        self.msg        = None                                                  # 마지막으로 보낸 메시지 (다시 보내기용)
        self.sent_at    = 0.0                                                   # 마지막 발행 시각 (monotonic)
        self.resent     = 0                                                     # 다시 보낸 횟수
        self.done       = False                                                 # 끝나면 main 반복 종료
        self.t0         = None                                                  # 첫 발행 시각 (처리 속도 계산)

        if self.mode == "lockstep":
            self.create_subscription(TrafficLightState, p("raw_topic"), self.on_raw, 10)
            self.create_timer(0.1, self.check_wait)                             # 0.1초마다 시작 조건, 응답 시간 초과 확인
        else:
            self.create_timer(1.0 / self.fps, self.publish_next)                # 영상 FPS 주기로 발행
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_camera else 0
        self.get_logger().info(f"source {src}: {self.fps:.2f} fps, {total} frames, mode {self.mode}, "
                               f"limit {self.limit or 'all'} -> {p('image_topic')}")

    # ===== 다음 프레임 1장 읽어서 발행 =====
    def publish_next(self):
        if self.done:
            return
        if self.limit and self.i >= self.limit:                                 # 시험용 프레임 수에 도달
            return self.finish("limit")
        ok, frame = self.cap.read()
        if not ok:                                                              # 영상 끝 (카메라는 읽기 실패)
            return self.finish("end of video")                  
        msg                 = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8") # OpenCV BGR 그대로
        msg.header.frame_id = str(self.i)                                       # 프레임 번호
        msg.header.stamp    = (self.get_clock().now().to_msg() if self.is_camera
                               else to_stamp(self.i / self.fps))                # 파일 = 영상 시각
        self.pub.publish(msg)
        self.msg, self.sent_at = msg, time.monotonic()
        self.t0 = self.t0 or self.sent_at
        self.i += 1
        if self.i % 300 == 0:                                                   # 300프레임마다 진행 상황
            self.get_logger().info(f"{self.i} frames, {self.rate():.1f} fps")

    # ===== lockstep: 인식 결과가 오면 다음 프레임 =====
    def on_raw(self, m):
        if not self.done and self.msg is not None and m.frame == self.i - 1:    # 방금 보낸 프레임의 결과만 인정
            self.publish_next()

    # ===== lockstep: 시작 조건, 응답 시간 초과 확인 =====
    def check_wait(self):
        if self.done:
            return
        if self.msg is None:                                                    # 아직 1장도 안 보냄
            if self.pub.get_subscription_count() > 0:                           # 인식 노드가 구독을 시작하면 첫 프레임
                self.publish_next()
            return
        if time.monotonic() - self.sent_at > self.wait_sec:                     # 응답 없음 -> 같은 프레임 다시 발행
            self.resent += 1
            self.get_logger().warn(f"no result for frame {self.i - 1} in {self.wait_sec:.1f} s -> resend")
            self.pub.publish(self.msg)
            self.sent_at = time.monotonic()

    # ===== 처리 속도 (첫 발행부터 지금까지) =====
    def rate(self):
        if self.t0 is None:
            return 0.0
        return self.i / max(time.monotonic() - self.t0, 1e-6)

    # ===== 종료 =====
    def finish(self, reason):
        self.done = True
        self.get_logger().info(f"done ({reason}): {self.i} frames ({self.i / self.fps:.1f} s video), "
                               f"{self.rate():.1f} fps, resent {self.resent}")

# =============================================================
# 실행 진입 점 (끝나면 스스로 종료)
# =============================================================
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VideoSource()
        while rclpy.ok() and not node.done:                                     # 영상이 끝날 때까지 콜백 처리
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.cap.release()
            node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()