"""
detector_view.py - 1~2. 신호등 검출 + 대상 선택 결과 확인용 스크립트
- detector.py의 TrafficLightDetector 로 영상을 한 프레임씩 검출하고,
  selector.py의 TargetSelector 로 대상 신호등 1개를 골라 화면에 그려서 결과를 눈으로 확인
- 박스 색: 초록(볼드) = 최종 대상 / 노랑 = 후보 / 회색 = 탈락 (탈락 이유 표시)
- --save 옵션을 주면 박스가 그려진 영상을 mp4 파일로 저장

실행: 
      python detector_view.py --source drive.mp4                    (영상 파일)
      python detector_view.py --source drive.mp4 --save out.mp4     (박스 영상 저장)
      python detector_view.py --source drive.mp4 --min-aspect 1.8   (선택 기준값 조정)
      python detector_view.py --source 0                            (웹캠 0번)
종료: 영상 창에서 q 키
"""

# =================== 표준 라이브러리 ===========================
import argparse                                                         # 명령행 인자(--source, --conf 등)를 읽기 위한 모듈
import time                                                             # FPS(초당 처리 프레임 수) 계산용 시간 측정 모듈

# =================== 외부 라이브러리 ===========================
import cv2                                                              # OpenCV: 영상 읽기, 박스/글자 그리기, 화면 표시

# =================== 프로젝트 모듈 =============================
from detector import TrafficLightDetector                               # 같은 폴더의 detector.py 에서 검출기 클래스 가져오기
from selector import TargetSelector, SelectorConfig                     # 같은 폴더의 selector.py 에서 대상 선택기와 기준값 가져오기

# 박스 색상 (OpenCV는 BGR 순서)
COLOR_TARGET    = (0, 255, 0)                                           # 초록: 최종 대상
COLOR_CANDIDATE = (0, 255, 255)                                         # 노랑: 1단계 통과 후보
COLOR_REJECTED  = (128, 128, 128)                                       # 회색: 1단계 탈락

# =============================================================
# 박스 1개와 글자를 프레임 위에 그리기
# =============================================================
def draw_box(frame, d, color, thickness, label):
    cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, thickness)  # 박스 왼쪽 위, 오른쪽 아래 좌표로 사각형
    cv2.putText(frame, label, 
                (d.x1, max(d.y1 - 6, 12)),                              # 박스 바로 위에 표시 (화면 위로 벗어나지 않게 최소 12)
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

# =============================================================
# 선택 결과(탈락/후보/대상)를 색을 나눠 그리기
# =============================================================
def draw_selection(frame, result, cfg):
    h, w = frame.shape[:2]                                              # 프레임 높이, 너비
    y_limit = int(h * cfg.max_center_y)                                 # 세로 위치 기준 선 (박스 중심이 이 선보다 아래면 탈락)
    cv2.line(frame, (0, y_limit), (w, y_limit), COLOR_REJECTED, 1)      # 기준선을 회색 가로선으로 표시

    for d, reason in result.rejected:                                   # 탈락 박스: 회색 + 탈락 이유 + 가로/세로 비율
        draw_box(frame, d, COLOR_REJECTED, 1, f"{reason} a{d.aspect:.1f}")

    for d in result.candidates:
        if d is not result.target:
            draw_box(frame, d, COLOR_CANDIDATE, 2, f"{d.conf:.2f} a{d.aspect:.1f}")

    if result.target is not None:
        d = result.target
        draw_box(frame, d, COLOR_TARGET, 3, f"TARGET {d.conf:.2f} a{d.aspect:.1f}")

    return frame

# =============================================================
# 명령행 인자 정의
# =============================================================
def parse_args():
    p = argparse.ArgumentParser(description="신호등 검출 + 대상 선택 테스트")
    p.add_argument("--source", default="drive.mp4", help="영상 파일 경로 또는 웹캠 번호(0, 1, ...)")
    p.add_argument("--model", default="yolo26n.pt", help="가중치 파일 경로")
    p.add_argument("--conf", type=float, default=0.25, help="신뢰도 임계값 (0~1)")
    p.add_argument("--imgsz", type=int, default=1280, help="추론 입력 크기")
    p.add_argument("--save", default=None, help="박스가 그려진 영상을 저장할 mp4 경로 (없으면 저장 안 함)")

    # 대상 선택 기준값 (기본값은 selector.py 의 SelectorConfig 와 같음)
    p.add_argument("--min-aspect", type=float, default=1.5, help="가로/세로 비율 하한")
    p.add_argument("--max-cy", type=float, default=0.55, help="박스 중심 y 상한 (화면 높이 대비 0~1)")
    p.add_argument("--min-area", type=int, default=300, help="최소 넓이 (px^2)")

    return p.parse_args()                                           # 입력된 인자를 객체로 반환 (args.source 형태로 사용)

# =============================================================
# 저장용 VideoWriter 생성 (원본과 같은 크기/FPS)
# =============================================================
def make_writer(cap, path):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))                      # 원본 영상 가로 크기
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                     # 원본 영상 세로 크기
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0                         # 원본 FPS (웹캠처럼 0으로 읽히면 30으로 대체)

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")                        # mp4 저장용 코덱
    return cv2.VideoWriter(path, fourcc, fps, (w, h))               # 저장 객체 반환 (프레임 크기가 원본과 같아야 저장됨)

# =============================================================
# 영상을 한 프레임씩 읽어 검출/선택하고 화면에 표시
# =============================================================
def main():
    args = parse_args()                                             # 명령행 인자 읽기

    # "0"과 같은 숫자 문자열이면 웹캠 번호(int)로 아니면 파일 경로(str) 그대로 사용
    source = int(args.source) if args.source.isdigit() else args.source

    # 검출기 생성 (이 시점에 모델 로드 / 필요하면 다운로드)
    detector = TrafficLightDetector(model_path=args.model, conf=args.conf, imgsz=args.imgsz)

    # 대상 선택기 생성 (명령행에서 받은 기준값 사용)
    cfg = SelectorConfig(min_aspect=args.min_aspect, max_center_y=args.max_cy, min_area=args.min_area)
    selector = TargetSelector(cfg)

    cap = cv2.VideoCapture(source)                                  # 영상 파일 또는 카메라 읽기
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {args.source}")      # 메시지 출력 후 종료

    writer = make_writer(cap, args.save) if args.save else None     # --save 가 있을 때만 저장 객체 생성

    cv2.namedWindow("detector", cv2.WINDOW_NORMAL)                  # 크기 조절 가능한 창 생성 (1080p 영상이 화면보다 클 때 대비)

    prev = time.perf_counter()                                      # FPS 계산용: 이전 프레임 처리 시각

    while True:                                                     # 영상이 끝나거나 q를 누를 때까지 반복
        ok, frame = cap.read()                                      # 다음 프레임 읽기 (ok: 성공 여부, frame: BGR 이미지)
        if not ok:                                                  # 더 읽을 프레임이 없으면 (영상 끝)
            break                                                   # 반복 종료

        detections = detector.detect(frame)                         # 1. 신호등 검출
        result = selector.select(detections, frame.shape)           # 2. 대상 신호등 선택
        draw_selection(frame, result, cfg)                          # 탈락/후보/대상 박스 그리기

        if writer is not None:                                      # 저장 옵션이 켜져 있으면
            writer.write(frame)                                     # 박스가 그려진 프레임 저장 (FPS 글자는 아직 안 그려진 상태)

        now     = time.perf_counter()                               # 현재 시각
        fps     = 1.0 / max(now - prev, 1e-6)                       # FPS = 1 / 프레임 처리 시간 (0으로 나누기 방지)
        prev    = now                                               # 다음 계산을 위해 현재 시각 저장

        # 화면 왼쪽 위에 FPS와 검출 개수, 후보개수, 대상 유무 표시
        target_txt = "YES" if result.target is not None else "NO"
        cv2.putText(frame, f"FPS {fps:.1f} lights {len(detections)} candidates {len(result.candidates)} target {target_txt}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("detector", frame)                               # 창에 프레임 표시

        if cv2.waitKey(1) & 0xFF == ord('q'):                       # 1ms 동안 키 입력 대기 (이 호출이 있어야 창이 갱신), q를 누르면 종료
            break

    cap.release()                                                   # 영상/카메라 자원 해제
    if writer is not None:                                          # 저장 중이었다면
        writer.release()                                            # 파일을 닫아야 mp4가 정상적으로 완성됨
    cv2.destroyAllWindows()                                         # 열린 창 모두 닫기

# 이 파일을 직접 실행했을 때만 main() 실행 (다른 파일에서 import 할 때는 실행 안됨)
if __name__ == "__main__":
    main()