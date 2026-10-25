"""
tools/eval_detect.py - C1/C2 검출 평가 (정답 박스와 검출 박스 비교, 상태 판별은 하지 않음)
- test 이미지마다 detector.py 로 신호등을 검출하고, 정답 박스(9개 클래스 전부)와 IoU >= 0.5 로 1:1 매칭
    매칭: 신뢰도가 높은 검출 박스부터, 아직 매칭되지 않은 정답 박스 중 IoU 가 가장 큰 것과 짝지음
- 두 기준으로 Recall/Precision 집계
    C1 all      : 모든 신호등 (9개 클래스). 어떤 신호등이든 찾으면 맞음
    C2 vehicle  : 차량 신호등 상태(0~5)만 맞음, invisible/pedestrian/other 와 짝지어진 검출도 오검출로 셈
- 추가 집계: 클래스별 Recall, 크기별 Recall(작음 < 32^2 / 중간 < 96^2 / 큼, 픽셀 넓이).
            selector 1단계 통과 차량 신호등의 Recall (실제 파이프라인이 쓰는 박스)
- 학습한 검출 모델은 --det-model <best.pt> --classes (빈값 = 모든 클래스) 로 같은 조건 비교 (8.3 대비)

실행 (저장소 폴더에서)
    python tools/eval_detect.py --src <Validation 폴더> --csv ~/datasets/traffic_light/eval_detect_test.csv
    시험: --limit 20 (앞 20장만)
"""

# =================== 표준 라이브러리 ===========================
import argparse                                                                     # 명령행 인자 (--src, --conf 등) 읽기
import csv                                                                          # 정답 박스별 결과를 csv 로 저장
import time                                                                         # 검출 시간 측정

from collections    import Counter                                                  # 클래스.크기별 개수 세기
from pathlib        import Path                                                     # 경로 다루기

# =================== 외부 라이브러리 ===========================
import cv2                                                                          # 이미지 읽기

# =================== 프로젝트 모듈   ===========================
from eval_hsv       import load_items                                               # 같은 tools 폴더: 정답 박스 읽기 (저장소 폴더를 모듈 경로에 추가)
from convert_labels import NAMES                                                    # 같은 tools 폴더: 클래스 이름 9개
from flow_judges    import iou                                                      # 같은 tools 폴더: 두 박스의 IoU
from selector       import TargetSelector                                           # 2. 대상 선택기 (1단계 탈락 기준만 사용)
from detector       import TrafficLightDetector                                     # 1. 신호등 검출기

N_STATE = 6                                                                         # 0~5 = 차량 신호등 상태 (C2 기준의 정답)
SIZES   = (("small", 32 ** 2), ("medium", 96 ** 2), ("large", float("inf")))        # 크기 구간 (픽셀 넓이 상한)

# =============================================================
# 명령행 인자
# =============================================================
def parse_args():
    p = argparse.ArgumentParser(description="C1/C2 검출 평가 (정답 박스 기준 Recall/Precision)")
    p.add_argument("--src", required=True, help="Validation 폴더 (02.라벨링데이터가 있는 곳)")
    p.add_argument("--data", default="~/datasets/traffic_light", help="convert_labels.py 로 만든 데이터셋 폴더")
    p.add_argument("--split", default="test", choices=["train", "val", "test"], help="평가할 분할")
    p.add_argument("--det-model", default="yolo26n.pt", help="검출 모델 (기본: COCO yolo26n)")
    p.add_argument("--classes", type=int, nargs="*", default=[9], help="남길 검출 클래스 번호 (기본 COCO 9, 빈 값 = 전부)")
    p.add_argument("--conf", type=float, default=0.25, help="검출 신뢰도 임계값 (detector.py 기본값)")
    p.add_argument("--imgsz", type=int, default=1280, help="검출 입력 크기 (detector.py 기본값)")
    p.add_argument("--iou", type=float, default=0.5, help="같은 박스로 볼 IoU 기준")
    p.add_argument("--device", default=None, help="0 = 첫번째 GPU, cpu = CPU (없으면 자동 선택)")
    p.add_argument("--limit", type=int, default=0, help="앞에서부터 이 장수만 평가 (0 = 전부, 시험용)")
    p.add_argument("--csv", default=None, help="정답 박스별 결과를 저장할 csv 경로 (없으면 저장 안 함)")
    return p.parse_args()

# =============================================================
# 검출 박스 목록 <-> 정답 박스 목록 1:1 매칭 -> {정답 번호: (검출 번호, IoU)}
# =============================================================
def match(dets, gts, thr):
    pairs = {}                                                                      # 정답 번호 -> (검출 번호, IoU)
    for di in sorted(range(len(dets)), key=lambda i: -dets[i].conf):                # 신뢰도 높은 검출 박스 부터
        best, best_iou = None, thr                                                  # 기준 이상 겹치는 정답만 후보
        for gi, (_, _, b) in enumerate(gts):
            if gi in pairs:                                                         # 이미 다른 검출과 짝지어진 정답은 건너뜀
                continue
            v = iou(dets[di], b)
            if v >= best_iou:                                                       # 더 많이 겹치는 정답이면 교체
                best, best_iou = gi, v
        if best is not None:
            pairs[best] = (di, best_iou)
    return pairs

# =============================================================
# 크기 구간 이름 (정답 박스 넓이 기준)
# =============================================================
def size_of(b):
    return next(name for name, top in SIZES if b.area < top)

# =============================================================
# 비율 문자열 "찾은 수/전체 = 비율"
# =============================================================
def ratio(a, n):
    return f"{a}/{n} = {a / max(n, 1):.1%}"

# =============================================================
# 결과 요약 출력
# =============================================================
def report(rows, n_det, n_tp_all, n_tp_veh):
    veh = [r for r in rows if r["cls_id"] < N_STATE]                                # 차량 신호등 상태 정답 박스
    print(f"\n[C1 all signals] recall {ratio(sum(r['found'] for r in rows), len(rows))}, "
          f"precision {ratio(n_tp_all, n_det)}")
    print(f"[C2 vehicle 0~5] recall {ratio(sum(r['found'] for r in veh), len(veh))}, "
          f"precision {ratio(n_tp_veh, n_det)}")
    sel = [r for r in veh if r["selector"] == "pass"]                               # selector 1단계를 통과하는 정답 박스
    print(f"[vehicle, selector pass] recall {ratio(sum(r['found'] for r in sel), len(sel))}")

    print("\nrecall by class")
    for i, n in enumerate(NAMES):
        rs = [r for r in rows if r["cls_id"] == i]
        print(f"    {i} {n:<11} {ratio(sum(r['found'] for r in rs), len(rs))}")

    print("\nrecall by size (pixel area: small < 32^2, medium < 96^2, large)")
    for scope, rs in (("all", rows), ("vehicle", veh)):
        print(f"    {scope:<8}" + " ".join(
            f"{name} {ratio(sum(r['found'] for r in rs if r['size'] == name), sum(r['size'] == name for r in rs))}"
            for name, _ in SIZES))

# =============================================================
# 평가 실행
# =============================================================
def main():
    args                = parse_args()
    src                 = Path(args.src).expanduser().resolve()                     # ~ 를 홈 폴더로 바꾼 절대 경로
    data                = Path(args.data).expanduser().resolve()                    
    items, missing      = load_items(src, data, args.split, len(NAMES))             # 이미지마다 정답 박스 9개 클래스 전부
    if args.limit:
        items = items[:args.limit]                                                  # 시험용으로 앞쪽만
    det                 = TrafficLightDetector(model_path=args.det_model, conf=args.conf, imgsz=args.imgsz,
                                               target_classes=args.classes or None, # 빈 값이면 모든 클래스 (학습한 검출 모델용)
                                               device=args.device)
    sel                 = TargetSelector()                                          # 기준값은 기본값 그대로 (C3/C4 와 같음)

    rows                = []                                                        # 정답 박스별 결과
    n_det, unread       = 0, 0                                                      # 전체 검출 박스 수, 읽지 못한 이미지 수    
    n_tp_all, n_tp_veh  = 0, 0                                                      # C1 맞은 검출 수, C2 맞은 검출 수
    t_det               = 0.0                                                       # 검출 시간 합 (초)
    for i, (img, gts) in enumerate(items, 1):
        print(f"\r{args.split} {i}/{len(items)}", end="", flush=True)               # 진행 상황 (같은 줄에 덮어쓰기)
        frame       = cv2.imread(str(img))                                          # 링크를 따라가 원본 이미지 읽기
        if frame is None:                                                           # 링크가 깨졌거나 파일이 손상되었으면
            unread += 1
            continue
        fh, fw      = frame.shape[:2]                                               # 이미지 높이, 너비
        t0          = time.perf_counter()
        dets        = det.detect(frame)                                             # 검출 박스 목록
        t_det      += time.perf_counter() - t0
        pairs       = match(dets, gts, args.iou)                                    # 정답 번호 -> (검출 번호, IoU)
        n_det      += len(dets)
        n_tp_all   += len(pairs)                                                    # 어떤 정답과든 짝지어진 검출 = C1 맞음
        n_tp_veh   += sum(gts[g][0] < N_STATE for g in pairs)                       # 차량 신호등 상태와 짝지어진 검출 = C2 맞음
        for gi, (cls, sig, b) in enumerate(gts):
            di, v = pairs.get(gi, (None, ""))                                       # 짝지어진 검출 (없으면 None)
            rows.append({"image": img.stem, "cls_id": cls, "cls": NAMES[cls], "signal": str(sig),
                         "width": b.width, "height": b.height, "area": b.area, "size": size_of(b),
                         "selector": sel.reject_reason(b, fh, fw) or "pass",
                         "found": di is not None, "iou": round(v, 3) if v != "" else "",
                         "conf": round(dets[di].conf, 3) if di is not None else ""})

    n_img = len(items) - unread                                                     # 실제로 평가한 이미지 수
    print(f"\nimages {n_img} (no json {missing}, unreadable {unread}), gt_boxes {len(rows)}, "
          f"detections {n_det}, detect {t_det / max(n_img, 1) * 1000:.1f} ms/image")
    report(rows, n_det, n_tp_all, n_tp_veh)

    if args.csv and rows:                                                           # --csv 를 줬으면 정답 박스별 결과 저장 (오류 분석용)
        path = Path(args.csv).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)                              # 저장 폴더가 없으면 생성
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))             # 첫 줄 = 열 이름 (image, cls_id, cls, ...)
            writer.writeheader()
            writer.writerows(rows)
        print(f"csv -> {path}")

if __name__ == "__main__":
    main()