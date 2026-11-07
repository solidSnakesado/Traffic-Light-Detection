"""
tools/record_states.py - 영상의 프레임별 원시 상태 기록 (4. 안정화 재생 평가용 입력)
- 영상을 한 프레임씩 1. 검출 -> 3. 판별 -> 2. 대상 선택 (detector_view.py 와 같은 처리) 하고
  프레임마다 target_state 와 참고 정보를 csv 한 줄로 저장 (화면 표시 없음)
- 시각은 영상 시각 t = 프레임 번호 / 영상 FPS (처리 속도와 무관 -> 다시 실행해도 같은 시각)
- 이 csv 를 replay_smoother.py 로 다시 읽어 안정화 기준값을 GPU 없이 비교
- csv 열
    frame, t        : 프레임 번호 (0부터), 영상 시각 (초)
    state           : selector 의 target_state (6개 상태 / UNKNOWN = 켜진 후보 없음 / NONE = 후보 없음)
    n_det, n_cand   : 검출 박스 수, selector 1단계 통과 후보 수
    cands           : 후보별 판별 상태 (| 로 구분, 대상 전환 확인용)
    conf, x1~y2     : 대상 박스 신뢰도와 좌표 (대상이 없으면 빈 값)
    rejects         : 1단계 탈락 이유별 개수 (예: low:2 small:1)
    ms              : 이 프레임 처리 시간 (검출 + 판별 + 선택)

실행 (저장소 폴더에서):
  python tools/record_states.py --source drive.mp4 --model runs/detect/det_vehicle_other/weights/best.pt --det-classes 0 \
         --cls-model runs/classify/state_cls_none/weights/best.pt --csv ~/datasets/traffic_light/states_drive_e03.csv
  시험: --limit 300 (앞 300프레임만) 
"""

# =================== 표준 라이브러리 ===========================
import argparse                                                                     # 명령행 인자(--source, --csv)
import csv                                                                          # 프레임별 결과를 csv 로 저장
import sys                                                                          # 모듈 검색 경로에 저장소 폴더 추가
import time                                                                         # 프레임 처리 시간 측정

from collections    import Counter                                                  # 탈락 이유, 상태별 개수 세기
from pathlib        import Path                                                     # 경로 다루기

# =================== 외부 라이브러리 ===========================
import cv2          # 영상 읽기

# =================== 프로젝트 모듈   ===========================
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))                     # 저장소 폴더 (detector.py, selector.py 가 있는 곳)
from detector       import TrafficLightDetector                                     # 1. 신호등 검출기
from selector       import TargetSelector, SelectorConfig                           # 2. 대상 선택기, 기준값
from flow_judges    import HsvJudge, ModelJudge                                     # 같은 tools 폴더: 3. 판별기 (C4 평가, detector_view 와 같은 코드)

FIELDS = ["frame", "t", "state", "n_det", "n_cand", "cands",                        # csv 열 이름 (순서 = 저장 순서)
          "conf", "x1", "y1", "x2", "y2", "rejects", "ms"]

# =============================================================
# 명령행 인자 (검출/판별/선택 인자는 detector_view.py 와 같음)
# =============================================================
def parse_args():
    p = argparse.ArgumentParser(description="영상 프레임별 원시 상태 기록 (안정화 재생 평가용)")
    p.add_argument("--source", required=True, help="영상 파일 경로")
    p.add_argument("--csv", required=True, help="프레임별 결과를 저장할 csv 경로")
    p.add_argument("--model", default="yolo26n.pt", help="검출 가중치 파일 경로")
    p.add_argument("--det-classes", type=int, nargs="*", default=[9], help="남길 검출 클래스 번호 (기본 COCO 9, 학습 모델은 0 = vehicle, 빈 값 = 전부)")
    p.add_argument("--cls-model", default=None, help="상태 분류 모델 best.pt (없으면 HSV 판별)")
    p.add_argument("--cls-imgsz", type=int, default=224, help="분류 모델 입력 크기 (학습 때와 같게)")
    p.add_argument("--conf", type=float, default=0.25, help="검출 신뢰도 임계값 (0~1)")
    p.add_argument("--imgsz", type=int, default=1280, help="검출 입력 크기")
    p.add_argument("--device", default=None, help="0 = 첫 번째 GPU, cpu = CPU (없으면 자동 선택)")
    p.add_argument("--min-aspect", type=float, default=1.5, help="가로/세로 비율 하한 (selector 기본값과 같음)")
    p.add_argument("--max-cy", type=float, default=0.55, help="박스 중심 y 상한 (화면 높이 대비 0~1)")
    p.add_argument("--min-area", type=int, default=300, help="최소 넓이 (px^2)")
    p.add_argument("--limit", type=int, default=0, help="앞에서부터 이 프레임 수만 기록 (0 = 전부, 시험용)")
    return p.parse_args()

# =============================================================
# 프레임 1장의 결과 -> csv 한 줄 (dict)
# =============================================================
def make_row(i, t, dets, res, ms):
    d   = res.target                                                                # 대상 박스 (없으면 None)
    box = [d.x1, d.y1, d.x2, d.y2] if d is not None else ["", "", "", ""]           # 대상 좌표 (없으면 빈 값 4개)
    rej = Counter(reason for _, reason in res.rejected)                             # 탈락 이유별 개수
    return dict(zip(FIELDS, [
        i, round(t, 4), res.target_state,                                           # 프레임 번호, 영상 시각, 원시 상태
        len(dets), len(res.candidates), "|".join(res.states),                       # 검출 수, 후보 수, 후보별 상태
        round(d.conf, 3) if d is not None else "", *box,                            # 대상 신뢰도, 좌표
        " ".join(f"{k}:{v}" for k, v in sorted(rej.items())),                       # 예: "low:2 small:1"
        round(ms, 1)]))                                                             # 처리 시간 (ms)

# =============================================================
# 기록 실행
# =============================================================
def main():
    args        = parse_args()
    cap         = cv2.VideoCapture(str(Path(args.source).expanduser()))             # 영상 파일 읽기
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {args.source}")
    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0                                 # 영상 FPS (0으로 읽히면 30으로 대체)
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))                            # 전체 프레임 수 (진행 표시 용)
    if args.limit:
        total = min(total, args.limit) if total > 0 else args.limit                 # 시험용이면 기록할 프레임 수로

    detector    = TrafficLightDetector(model_path=args.model, conf=args.conf, imgsz=args.imgsz,
                                       target_classes=args.det_classes or None,     # 빈 값이면 모든 클래스
                                       device=args.device)
    cfg         = SelectorConfig(min_aspect=args.min_aspect, max_center_y=args.max_cy, min_area=args.min_area)
    selector    = TargetSelector(cfg)
    judges      = ModelJudge(args.cls_model, args.cls_imgsz,
                             args.device) if args.cls_model else HsvJudge()         # 상태 판별기
    print(f"source {args.source}: {fps:.2f} fps, {total} frames, judge {judges.name}")

    path                = Path(args.csv).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)                                  # 저장 폴더가 없으면 생성
    states              = Counter()                                                 # 원시 상태별 프레임 수
    n_det, ms_sum, i    = 0, 0.0, 0                                                 # 전체 검출 수, 처리 시간 합, 기록한 프레임 수

    with open(path, "w", newline="", encoding="utf-8") as f:                        
        writer          = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()                                                        # 첫 줄 = 열 이름
        while not (args.limit and i >= args.limit):                                 # 시험용이면 정한 프레임 수까지
            ok, frame   = cap.read()                                                # 다음 프레임 읽기
            if not ok:                                                              # 영상 끝
                break
            t0          = time.perf_counter()
            dets        = detector.detect(frame)                                    # 1. 검출
            res         = selector.select(dets, frame.shape, judges.bind(frame))    # 3. 후보마다 판별 + 2. 대상 1개 선택
            ms          = (time.perf_counter() - t0) * 1000                         # 이 프레임 처리 시간 (ms)
            writer.writerow(make_row(i, i / fps, dets, res, ms))                    # 시각 = 프레임 번호 / 영상 FPS
            states[res.target_state] += 1
            n_det   += len(dets)
            ms_sum  += ms
            i       += 1
            if i % 60 == 0:                                                         # 60프레임마다 진행 상황 (같은 줄에 덮어쓰기)
                print(f"\r{i}/{total} frames, {ms_sum / i:.1f} ms/frame", end="", flush=True)
    cap.release()

    # ===== 결과 요약 출력 =====
    print(f"\nframes {i} ({i / fps:.1f} s), {ms_sum / max(i, 1):.1f} ms/frame "
          f"(processing {1000 * i / max(ms_sum, 1e-6):.1f} fps)")
    print("raw state: " + ", ".join(f"{k} {v} ({v / max(i, 1):.1%})" for k, v in states.most_common()))
    print(f"detections {n_det}")
    if n_det == 0:                                                                  # 학습 모델 인데 --det--classes 를 빼면 검출이 전부 걸러짐
        print("WARNING: 검출 0개 -> --model 과 --det-classes 가 맞는 지 확인 (학습 모델은 --det-classes 0)")
    print(f"csv -> {path}")

if __name__ == "__main__":
    main()