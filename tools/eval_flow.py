"""
tools/eval_flow.py - C4 전체 흐름 평가 (COCO 검출 -> selector -> 판별기), 판별기끼리 같은 조건으로 비교
- test 이미지마다 detector.py 로 신호등을 1번 검출하고, 같은 검출 결과를 판별기 마다 selector.select(judge) 에 넣음
    판별기: HSV(classifier.py) + --models 로 준 학습 모델들 (기본: E02 state_cls, E03 state_cls_none)
- 정답 대상: 정답 상태 박스(0~5)에 selector 1단계 필터 + pick 을 그대로 적용한 1개 (없으면 "대상 없음" NONE)
- 비교는 상태만 (정답 상태 == target_state). 박스 일치(IoU >= 0.5)는 참고 지표
- 틀린 이유: det_miss(검출 실패) / wrong_box(다른 박스 선택) / wrong_state(상태 오판)
- 정답 대상이 없는 이미지: false_alarm(없는 신호를 만듦) 비율, unknown(판별 불가 대상) 비율
    etc-only: 상태 박스(0~5)는 없지만 selector 1단계를 통과한 etc 차량 신호등(점멸, 화살표 조합 등)이 있는 이미지
    -> 오경보를 전체(false_al)와 etc-only 를 뺀 이미지(fa_clean)로 나눠 출력
- 검출 설정은 detector.py 기본값 그대로 (yolo26n, imgsz 1280, conf 0.25)
- 학습한 검출 모델: --det-model <best.pt> --det-classes 0 (vehicle 만 selector 에 넘김)

실행 (저장소 폴더에서):
  python tools/eval_flow.py --src <Validation 폴더> --csv ~/datasets/traffic_light/eval_flow_test.csv
  시험: --limit 20 (앞 20장만)
"""

# =================== 표준 라이브러리 ===========================
import argparse                                                                                 # 명령행 인자(--src, --models 등) 읽기
import csv                                                                                      # 이미지별 결과를 csv 로 저장
import time                                                                                     # 검출/판별 시간 측정

from collections    import Counter                                                              # 결과 종류별 개수 세기
from pathlib        import Path                                                                 # 경로 다루기

# =================== 외부 라이브러리 ===========================
import cv2                                                                                      # 이미지 읽기

# =================== 프로젝트 모듈   ===========================
from flow_judges    import HsvJudge, ModelJudge, gt_target, outcome, iou, IOU_MATCH             # 같은 tools 폴더: 판별기, 정답 대상, 결과 분류
from eval_hsv       import STATES, load_items, report                                           # 같은 tools 폴더: 정답 박스 읽기, 혼동 행렬 출력
from convert_labels import NAMES                                                                # 같은 tools 폴더: 클래스 이름 9개 (etc 박스까지 읽기용)
from classifier     import UNKNOWN                                                              # 3. 판별 불가 상태 이름 (eval_hsv 가 저장소 경로를 추가해 둠)
from selector       import TargetSelector, NONE                                                 # 2. 대상 선택기, "대상 없음" 상태 이름
from detector       import TrafficLightDetector                                                 # 1. COCO 신호등 검출기

RUNS    = Path(__file__).resolve().parent.parent / "runs/classify"                              # train.py 결과 폴더
MODELS  = [RUNS / "state_cls/weights/best.pt", RUNS / "state_cls_none/weights/best.pt"]         # 기본 비교 모델 (E02, E03  )
ERRORS  = ("det_miss", "wrong_box", "wrong_state")                                              # 틀린 이유 3가지

# =============================================================
# 명령행 인자
# =============================================================
def parse_args():
    p = argparse.ArgumentParser(description="C4 전체 흐름 평가 (검출 + selector + 판별기)")
    p.add_argument("--src", required=True, help="Validation 폴더 (02.라벨링데이터가 있는 곳)")
    p.add_argument("--data", default="~/datasets/traffic_light", help="convert_labels.py 로 만든 데이터셋 폴더")
    p.add_argument("--split", default="test", choices=["train", "val", "test"], help="평가할 분할")
    p.add_argument("--models", nargs="*", default=[str(m) for m in MODELS], help="비교할 분류 모델 best.pt 들 (0개면 HSV만)")
    p.add_argument("--det-classes", type=int, nargs="*", default=[9], help="남길 검출 클래스 번호 (기본 COCO 9, 학습 모델은 0 = vehicle, 빈 값 = 전부)")
    p.add_argument("--no-hsv", action="store_true", help="HSV 판별기 빼기")
    p.add_argument("--det-model", default="yolo26n.pt", help="검출 모델 (기본: COCO yolo26n)")
    p.add_argument("--conf", type=float, default=0.25, help="검출 신뢰도 임계값 (detector.py 기본값)")
    p.add_argument("--imgsz", type=int, default=1280, help="검출 입력 크기 (detector.py 기본값)")
    p.add_argument("--cls-imgsz", type=int, default=224, help="분류 모델 입력 크기 (학습 때와 같게)")
    p.add_argument("--device", default=None, help="0 = 첫번째 GPU, cpu = CPU (없으면 자동 선택)")
    p.add_argument("--limit", type=int, default=0, help="앞에서부터 이 장수만 평가 (0 = 전부, 시험용)")
    p.add_argument("--csv", default=None, help="이미지별 결과를 저장할 csv 경로 (없으면 저장 안 함)")
    return p.parse_args()

# =============================================================
# 판별기 1개의 결과 요약 출력 -> 비교표 한 줄용 값 반환
# =============================================================
def summary(rows, name, sec):
    tg  = [r for r in rows if r["gt"] != NONE]                                                  # 정답 대상이 있는 이미지
    nt  = [r for r in rows if r["gt"] == NONE]                                                  # 정답 대상이 없는 이미지
    cnt = Counter(r["result"] for r in rows)                                                    # 결과 종류별 개수
    box = sum(r["box_iou"] != "" and r["box_iou"] >= IOU_MATCH for r in tg)                     # 대상 박스가 정답 박스와 일치한 수
    n_t = max(len(tg), 1)                                                                       # 0 으로 나누기 방지
    n_n = max(len(nt), 1)
    print(f"\n===== {name} ===== {sec / max(len(rows), 1) * 1000:.2f} ms/image (judge + select)")
    print(f"target images {len(tg)}: acc {cnt['ok'] / n_t:.1%}, box match {box / n_t:.1%}, "
          + ", ".join(f"{e} {cnt[e]} ({cnt[e] / n_t:.1%})" for e in ERRORS))
    print(f"no-target images {len(nt)}: ok {cnt['ok_none'] / n_n:.1%}, "
          f"false_alarm {cnt['false_alarm']} ({cnt['false_alarm'] / n_n:.1%}), "
          f"unknown {cnt['unknown']} ({cnt['unknown'] / n_n:.1%})")
    ne  = [r for r in nt if r["etc"]]                       # etc 차량 신호등만 있는 이미지
    nc  = [r for r in nt if not r["etc"]]                   # 신호등 대상이 정말 없는 이미지
    fa  = sum(r["result"] == "false_alarm" for r in nc)     # etc-only 를 뺀 오경보 수
    print(f"  etc-only {len(ne)}: pred " + ", ".join(f"{k} {v}" for k, v in Counter(r["pred"] for r in ne).most_common()))
    print(f"  clean {len(nc)}: false_alarm {fa} ({fa / max(len(nc), 1):.1%})")
    report(tg, f"{name} target state", STATES + [UNKNOWN, NONE])                                # 정답 대상 상태 x 예측 상태 (NONE = 대상 못 고름)
    ok_all = (cnt["ok"] + cnt["ok_none"]) / max(len(rows), 1)                                   # 대상 없음까지 포함한 전체 정확도
    return [name, cnt["ok"] / n_t, box / n_t] + [cnt[e] / n_t 
                                                 for e in ERRORS] + [cnt["false_alarm"] / n_n, fa / max(len(nc), 1), ok_all]

# =============================================================
# 평가 실행
# =============================================================
def main():
    args            = parse_args()
    src             = Path(args.src).expanduser().resolve()                                     # ~ 를 홈 폴더로 바꾼 절대 경로
    data            = Path(args.data).expanduser().resolve()
    items, missing  = load_items(src, data, args.split, len(NAMES))                             # 이미지마다 정답 박스 9클래스 전부, 박스 없는 이미지도 포함      
    if args.limit:
        items = items[:args.limit]      # 시험용으로 앞쪽만
    det             = TrafficLightDetector(model_path=args.det_model, conf=args.conf,
                                           imgsz=args.imgsz, device=args.device,
                                           target_classes=args.det_classes or None)             # 검출기 (빈 값이면 모든 클래스)
    sel             = TargetSelector()      # 기준값은 기본값 그대로 (E01~E03과 같음)
    judges          = ([] if args.no_hsv else [HsvJudge()]) + [ModelJudge(m, args.cls_imgsz, args.device) 
                                                               for m in args.models]
    print("judges:", ", ".join(j.name for j in judges))

    rows            = []                                                                        # 이미지 x 판별기별 결과
    sec             = Counter()                                                                 # 판별기별 걸린 시간 합 (초)
    t_det, unread   = 0.0, 0                                                                    # 검출 시간 합, 읽지 못한 이미지 수
    found, n_gt     = 0, 0                                                                      # 정답 대상을 검출기가 찾은 수, 정답 대상 수
    for i, (img, boxes) in enumerate(items, 1):
        print(f"\r{args.split} {i}/{len(items)}", end="", flush=True)                           # 진행 상황 (같은 줄에 덮어쓰기)
        frame = cv2.imread(str(img))                                                            # 링크를 따라가 원본 이미지 읽기
        if frame is None:                                                                       # 링크가 깨졌거나 파일이 손상됐으면
            unread += 1
            continue
        fh, fw                  = frame.shape[:2]                                               # 이미지 높이, 너비
        gts                     = [g for g in boxes if g[0] < len(STATES)]                      # 차량 신호 상태(0~5) 박스만 정답 대상 후보
        etc, _                  = sel.filter([b for _, s, b in boxes if s == "etc"], fh, fw)    # selector 1단계를 통과한 etc 차량 신호등
        gt_box, gt_state, sig   = gt_target(sel, gts, fh, fw)                                   # 정답 대상 (없으면 None, NONE)
        t0                      = time.perf_counter()
        dets                    = det.detect(frame)                                             # 검출은 1번만 (모든 판별기가 같은 박스 사용)
        t_det                  += time.perf_counter() - t0
        if gt_box is not None:
            n_gt           += 1
            found          += any(iou(d, gt_box) >= IOU_MATCH for d in dets)                    # 검출기가 정답 대상을 찾았는 가
        for j in judges:
            t0              = time.perf_counter()
            res             = sel.select(dets, frame.shape, j.bind(frame))                      # 1단계 + 판별 + 켜진 후보 중 1개
            sec[j.name]    += time.perf_counter() - t0
            b               = res.target                                                        # 선택된 대상 박스 (없으면 None)
            rows.append({"judge": j.name, "image": img.stem, "gt": gt_state, "signal": sig,
                         "pred": res.target_state,
                         "result": outcome(gt_box, gt_state, dets, b, res.target_state),
                         "box_iou": round(iou(b, gt_box), 3) if b is not None and gt_box is not None else "",
                         "n_det": len(dets), "n_cand": len(res.candidates),
                         "gt_w": gt_box.width if gt_box is not None else "",
                         "gt_h": gt_box.height if gt_box is not None else "",
                         "etc": int(gt_box is None and len(etc) > 0)})                          # 1 = etc-only 이미지

    n_img = len(items) - unread                                                                 # 실제로 평가한 이미지 수
    print(f"\nimages {n_img} (no json {missing}, unreadable {unread}), "
          f"detect {t_det / max(n_img, 1) * 1000:.1f} ms/image")
    print(f"gt target found by detector (IoU >= {IOU_MATCH}): {found}/{n_gt} = {found / max(n_gt, 1):.1%}")
    table = [summary([r for r in rows if r["judge"] == j.name], j.name, sec[j.name]) for j in judges]

    print(f"\n{'judge':<16}{'acc':>8}{'box':>8}{'det_miss':>10}{'wrong_box':>11}"
          f"{'wrong_st':>10}{'false_al':>10}{'fa_clean':>10}{'all':>8}")                        # 판별기 비교표
    for name, *v in table:
        print(f"{name:<16}" + "".join(f"{x:>{w}.1%}" for x, w in zip(v, (8, 8, 10, 11, 10, 10, 10, 8))))

    if args.csv and rows:                                                                       # --csv 를 줬으면 이미지별 결과 저장 (오류 분석용)
        path = Path(args.csv).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)                                          # 저장 폴더가 없으면 생성
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))                         # 첫 줄 = 열 이름 (judge, image, gt, ...)
            writer.writeheader()
            writer.writerows(rows)
        print(f"csv -> {path}")

if __name__ == "__main__":
    main()