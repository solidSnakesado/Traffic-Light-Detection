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
import time                                                                                 # 실시간 처리 간격, 대기

from collections                            import Counter                                  # 상태별 프레임 수
from pathlib                                import Path                                     # 경로 다루기 (~ 확장)

# =================== 외부 라이브러리 ===========================
import cv2                                                                                  # 영상 읽기

# =================== ROS2 ====================================
import rclpy                                                                                # ROS2 Python 클라이언트
from rclpy.node                             import Node                                     # 노드 기본 클래스
from sensor_msgs.msg                        import Image                                    # 구독 모드 입력
from traffic_light_msgs.msg                 import TrafficLightState                        # 발행할 인식 결과

# =================== 패키지 모듈 ===============================
from traffic_light_nodes.perception_core    import Perception, fill_msg                     # 인식 처리 (ROS2 와 무관)

PACES = ("all", "realtime")                                                                 # 사용할 수 있는 pace 값

# =============================================================
# 인식 노드
# =============================================================
class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        defaults = dict(source="", pace="all", limit=0, wait_sub=True,                      # 입력, 처리 방식
                        image_topic="/camera/image_raw", raw_topic="/traffic_light/raw",
                        repo_dir="", model="", cls_model="", det_classes=[0],               # 모델 (학습 모델 = 클래스 0)
                        imgsz=1280, cls_imgsz=224, conf=0.25, device="",                    # 빈 device = 자동 선택
                        min_aspect=1.5, max_cy=0.55, min_area=300)                          # selector 기준값 (record_states 와 같음)
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value                                     # 파라미터 값 읽기
        if not p("model"):
            raise SystemExit("model 파라미터가 필요합니다 (-p model:=<검출 가중치 경로>)")
        if p("pace") not in PACES:
            raise SystemExit(f"pace 는 {', '.join(PACES)} 중 하나: {p('pace')}")

        self.core   = Perception(repo_dir=p("repo_dir"), model=str(Path(p("model")).expanduser()),
                                 cls_model=str(Path(p("cls_model")).expanduser()) if p("cls_model") else "",
                                 det_classes=[c for c in p("det_classes") if c >= 0],       # -1 만 넣으면 모든 클래스
                                 imgsz=p("imgsz"), cls_imgsz=p("cls_imgsz"), conf=p("conf"),
                                 device=p("device") or None, min_aspect=p("min_aspect"),
                                 max_cy=p("max_cy"), min_area=p("min_area"))
        self.pub        = self.create_publisher(TrafficLightState, p("raw_topic"), 100)     # 작은 메시지, 넉넉히 보관
        self.pace       = p("pace")
        self.limit      = int(p("limit"))
        self.wait_sub   = bool(p("wait_sub"))
        self.states     = Counter()                                                         # 원시 상태별 프레임 수
        self.n, self.n_det, self.ms_sum, self.skipped = 0, 0, 0.0, 0                        # 처리 수, 검출 수, 처리 시간 합, 건너뛴 수
        self.done       = False                                                             # 직접 읽기가 끝나면 True
        self.t_start    = None                                                              # 처리 시작 시각 (monotonic)

        src = str(p("source"))
        if src:                                                                             # 직접 읽기
            cam         = src.isdigit()
            self.cap    = cv2.VideoCapture(int(src) if cam else str(Path(src).expanduser()))
            if not self.cap.isOpened():
                raise SystemExit(f"영상을 열 수 없습니다: {src}")
            self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30.0                            # 0으로 읽히면 30
            self.i      = 0                                                                 # 다음에 읽을 프레임 번호
            info = f"source {src}: {self.fps:.2f} fps, pace {self.pace}"
        else:                                                                               # 영상 토픽 구독
            from cv_bridge import CvBridge                                                  # 구독 모드에서만 필요
            self.cap, self.bridge = None, CvBridge()
            self.create_subscription(Image, p("image_topic"), self.on_image, 1)             # 최신 1장만 (늦으면 버림)
            info = f"subscribe {p('image_topic')}"
        self.get_logger().info(f"{info}, judge {self.core.name} -> {p('raw_topic')}")

    # ===== 직접 읽기: 프레임 1장 처리 (main 반복해서 호출) =====
    def step(self):
        if self.wait_sub and self.pub.get_subscription_count() == 0:                        # 기록 노드 등이 아직 없음
            time.sleep(0.1)
            return
        if self.limit and self.i >= self.limit:
            return self.finish("limit")
        self.t_start = self.t_start or time.monotonic()
        if self.pace == "realtime":
            due = int((time.monotonic() - self.t_start) * self.fps)                         # 지금 처리해야 할 프레임 번호
            if self.i > due:                                                                # 영상 시각보다 빠름 -> 기다림
                time.sleep(min((self.i - due) / self.fps, 0.01))
                return
            while self.i < due and self.cap.grab():                                         # 늦음 -> 디코딩만 하고 건너뜀
                self.i, self.skipped = self.i + 1, self.skipped + 1
        ok, frame = self.cap.read()
        if not ok:
            return self.finish("end of video")
        self.process_frame(self.i, self.i / self.fps, frame)                                # 시각 = 프레임 번호 / 영상 FPS
        self.i += 1

    # ===== 구독 모드: 영상 메시지 1개 처리 =====
    def on_image(self, msg):
        frame   = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        fid     = msg.header.frame_id                                                       # video_source 는 프레임 번호를 넣음
        i       = int(fid) if fid.isdigit() else self.n                                     # 없으면 받은 순서
        self.process_frame(i, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, frame)

    # ===== 처리 + 발행 + 통계 =====
    def process_frame(self, i, t, frame):
        dets, res, ms = self.core.process(frame)
        msg = fill_msg(TrafficLightState(), i, t, dets, res)
        msg.header.stamp = self.get_clock().now().to_msg()                                  # 발행 시각
        self.pub.publish(msg)
        self.states[res.target_state] += 1
        self.n, self.n_det, self.ms_sum = self.n + 1, self.n_det + len(dets), self.ms_sum + ms
        if self.n % 300 == 0:                                                               # 300프레임마다 진행 상황
            self.get_logger().info(f"{self.n} frames, {self.ms_sum / self.n:.1f} ms/frame, "
                                   f"skipped {self.skipped}")

    # ===== 종료 요약 (record_states.py 와 같은 항목) =====
    def finish(self, reason):
        self.done   = True
        n           = max(self.n, 1)
        wall        = time.monotonic() - (self.t_start or time.monotonic())
        self.get_logger().info(f"done ({reason}): frames {self.n}, skipped {self.skipped}, "
                               f"{self.ms_sum / n:.1f} ms/frame, {self.n / max(wall, 1e-6):.1f} fps")
        self.get_logger().info("raw state: " + ", ".join(f"{k} {v} ({v / n:.1%})"
                                                         for k, v in self.states.most_common()))
        self.get_logger().info(f"detections {self.n_det}")

# =============================================================
# 실행 진입 점 (직접 읽기는 영상이 끝나면 종료, 구독 모드는 Ctrl+C 까지)
# =============================================================
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PerceptionNode()
        if node.cap is None:
            rclpy.spin(node)
        while node.cap is not None and rclpy.ok() and not node.done:
            node.step()
            rclpy.spin_once(node, timeout_sec=0)                                            # 구독자 연결 등 내부 처리
        time.sleep(0.5)                                                                     # 마지막 메시지가 전달될 시간
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if node.cap is not None:
                node.cap.release()
            node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()