"""
tools/eval_cls.py - 학습한 상태 분류 모델(B 구조) 평가 (C3: 정답 박스 기준, eval_hsv.py 와 같은 조건)
- eval_hsv.py 와 같은 이미지, 같은 정답 박스 (클래스 0~5), 같은 지표, 같은 출력 형식 -> HSV 베이스라인(E01)과 직접 비교
- 정답 박스를 make_crops.py 와 같은 방법(정사각형 회색 여백)으로 잘라 모델에 넣음
- 10픽셀 미만 작은 박스도 빼지 않고 평가 (학습 때만 Ultralytics 가 제외, HSV 평가와 박스 수를 같게 유지)
- 모델은 상태 6개 중 하나를 항상 고르므로 UNKNOWN(거부)은 0
- csv 에는 예측 확률(conf)도 저장 (틀린 박스가 확신이 낮았는 지 확인용)

실행 (저장소 폴더에서):
  python tools/eval_cls.py --src <Validation 폴더> --csv ~/datasets/traffic_light/eval_cls_test.csv
"""

# =================== 표준 라이브러리 ===========================
import argparse                                                                             # 명령행 인자(--src, --model 등) 읽기
import csv                                                                                  # 박스별 결과를 csv 로 저장
import time                                                                                 # 박스당 판별 시간 측정

from collections    import Counter                                                          # 탈락 이유별 개수 세기
from pathlib        import Path                                                             # 경로 다루기

# =================== 외부 라이브러리 ===========================
import cv2                                                                                  # 이미지 읽기
from ultralytics    import YOLO                                                             # 학습한 분류 모델 불러오기

# =================== 프로젝트 모듈   ===========================
from eval_hsv       import STATES, load_items, report, report_signal                        # 같은 tools 폴더: 정답 박스 읽기, 결과 표 출력 (HSV 평가와 같은 함수)
from make_crops     import square_crop                                                      # 같은 tools 폴더: 학습 데이터와 같은 방법으로 자르기
from selector       import TargetSelector                                                   # 2. 대상 선택기 (1단계 탈락 기준만 사용, eval_hsv 가 저장소 경로를 추가해둠)

BEST = Path(__file__).resolve().parent.parent / "runs/classify/state_cls/weights/best.pt"   # train.py 기본 결과 위치

# =============================================================
# 명령행 인자
# =============================================================
def parse_args():
    p = argparse.ArgumentParser(description="상태 분류 모델 평가 (정답 박스 기준, eval_hsv.py 와 같은 조건)")
    p.add_argument("--src", required=True, help="Validation 폴더 (02.라벨링데이터 가 있는 곳)")
    p.add_argument("--data", default="~/datasets/traffic_light", help="convert_labels.py 로 만든 데이터셋 폴더")
    p.add_argument("--split", default="test", choices=["train", "val", "test"], help="평가할 분할")
    p.add_argument("--model", default=str(BEST), help="학습한 가중치 (train.py 결과 best.pt)")
    p.add_argument("--imgsz", type=int, default=224, help="모델 입력 크기 (학습 때와 같게)")
    p.add_argument("--device", default=None, help="0 = 첫 번째 GPU, cpu = CPU (없으면 자동 선택)")
    p.add_argument("--csv", default=None, help="박스별 결과를 저장할 csv 경로 (없으면 저장 안 함)")
    return p.parse_args()

# =============================================================
# 평가 실행
# =============================================================
def main():
    args            = parse_args()
    src             = Path(args.src).expanduser().resolve()                                 # ~ 를 홈 폴더로 바꾼 절대 경로
    data            = Path(args.data).expanduser().resolve()
    items, missing  = load_items(src, data, args.split)                                     # eval_hsv.py 와 같은 이미지, 같은 정답 박스
    model           = YOLO(str(Path(args.model).expanduser()))                              # 학습한 분류 모델
    sel             = TargetSelector()                                                      # 기준값은 기본값 그대로 (eval_hsv.py 와 같음)

    rows            = []                                                                    # 박스별 결과
    unread          = 0                                                                     # 읽지 못한 이미지 수
    t_sum           = 0.0                                                                   # 판별에 걸린 시간 합 (초)
    for i, (img, gts) in enumerate(items, 1):
        print(f"\r{args.split} {i}/{len(items)}", end="", flush=True)                       # 진행 상황 (같은 줄에 덮어쓰기)
        if not gts:                                                                         # 평가할 박스가 없는 이미지는 읽지 않음
            continue
        frame = cv2.imread(str(img))                                                        # 링크를 따라가 원본 이미지 읽기
        if frame is None:                                                                   # 링크가 깨졌거나 파일이 손상됐으면
            unread += 1
            continue
        fh, fw  = frame.shape[:2]                                                           # 이미지 높이, 너비
        crops   = [square_crop(frame, box) for _, _, box in gts]                            # 이미지 안 박스들을 한 번에 잘라서
        t0      = time.perf_counter()
        results = model.predict(crops, imgsz=args.imgsz, device=args.device, verbose=False) # 한 번에 예측 (BGR 이미지 그대로)
        t_sum  += time.perf_counter() - t0
        for (cls, sig, box), r in zip(gts, results):
            pred    = model.names[r.probs.top1]                                             # 가장 확률 높은 클래스 이름 (= 상태 이름, 폴더 이름)
            reason  = sel.reject_reason(box, fh, fw)                                        # selector 1단계 탈락 이유 (통과면 None)
            rows.append({"image": img.stem, "gt": STATES[cls], "signal": str(sig), "pred": pred,
                         "conf": round(float(r.probs.top1conf), 4), "aspect": round(box.aspect, 2),
                         "width": box.width, "height": box.height, "selector": reason or "pass"})

    print(f"\nimages {len(items)} (no json {missing}, unreadable {unread}), boxes {len(rows)}, "
          f"{t_sum / max(len(rows), 1) * 1000:.2f} ms/box")
    print("selector:", ", ".join(f"{k} {v}" for k, v in Counter(r["selector"] for r in rows).most_common()))
    report(rows, "all")                                                                     # 정답 박스 전부        
    report([r for r in rows if r["selector"] == "pass"], "selector")                        # selector 1단계 통과 박스만
    report_signal(rows)

    if args.csv and rows:                                                                   # --csv 를 줬으면 박스별 결과 저장 (오류 분석용)
        path = Path(args.csv).expanduser()          
        path.parent.mkdir(parents=True, exist_ok=True)                                      # 저장 폴더가 없으면 생성
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))                     # 첫 줄 = 열 이름 (image, gt, signal, ...)
            writer.writeheader()
            writer.writerows(rows)
        print(f"csv -> {path}")

if __name__ == "__main__":
    main()