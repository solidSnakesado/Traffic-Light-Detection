"""
tools/eval_hsv.py - HSV 상태 판별 베이스라인 측정 (C3: 정답 박스 기준, YOLO 사용 안 함)
- 데이터셋 split(기본 test)의 이미지마다 원본 라벨 JSON을 읽어 차량 신호 상태 박스(클래스0~5)만 꺼냄
  (invisible / pedestrian / other 는 HSV 판별기에 해당하는 상태가 없어 제외)
- 정답 박스를 classifier.py 의 SignalClassifier 에 그대로 넣어 예측 상태를 정답과 비교
- 두 범위로 집계
    all     : 정답 박스 전부
    selector: selector.py 1단계(비율/위치/넓이/가장자리)를 통과한 박스만 -> 실제 파이프라인에서 판별기가 받는 박스
- 지표: 정확도(UNKNOWN = 오답), 거부율(UNKNOWN 비율), 거부 제외 정확도, 클래스별 정확도, 혼동 행렬, 박스당 판별 시간
- 원본 signal 값별 결과도 출력 (YELLOW 로 합친 "red and yellow", "green and yellow" 를 따로 확인)
- 라벨은 YOLO txt 대신 원본 JSON을 읽음 -> 픽셀 좌표 그대로 + 원본 signal 값 확인 가능
- HSV 기준값을 조정할 때는 --split val 로만 확인 (test 는 최종 비교 전용)

실행
  python tools/eval_hsv.py --src <Validation 폴더> --csv ~/datasets/traffic_light/eval_hsv_test.csv
"""

# =================== 표준 라이브러리 ===========================
import argparse                                                         # 명령행 인자 (--src, --split 등) 읽기
import csv                                                              # 박스별 결과를 csv로 저장
import glob                                                             # 라벨 JSON 목록 찾기 (**: 하위 폴더 전체)
import sys                                                              # 모듈 검색 경로에 저장소 폴더 추가
import time                                                             # 박스당 판별 시간 측정

from collections    import Counter                                      # 예측 상태별 개수 세기
from pathlib        import Path                                         # 경로 다루기
from types          import SimpleNamespace                              # 속성(d.x1 등)으로 읽는 간단한 박스 객체

# =================== 외부 라이브러리 ===========================
import cv2                                                              # 이미지 읽기

# =================== 프로젝트 모듈   ===========================
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))         # 저장소 폴더 (classifier.py, selector.py 가 있는 곳)
from classifier     import SignalClassifier, UNKNOWN                    # 3. HSV 상태 판별기, 판별 불가 상태 이름
from selector       import TargetSelector                               # 2. 대상 선택기 (1단계 탈락 기준만 사용)
from convert_labels import NAMES, class_id, read_label                  # 같은 tools 폴더: 클래스 이름, 클래스 매핑, JSON 읽기

STATES  = NAMES[:6]                                                     # 평가할 클래스 0~5 (classifier.py 상태 이름과 같음)
COLS    = STATES + [UNKNOWN]                                            # 혼동 행렬 열 (예측에는 UNKNOWN 이 추가됨)
SHORT   = {"RED_LEFT": "R_LEFT", "GREEN_LEFT": "G_LEFT"}                # 혼동 행렬 열 제목 줄임 (터미널 폭 80자 안에 맞추기)

# =============================================================
# 원본 좌표 -> 박스 객체 (detector.py 의 Detection 과 같은 속성), 너무 작으면 None
# =============================================================
def make_box(x1, y1, x2, y2, w, h):
    x1, x2 = sorted((x1, x2))                                           # 점 순서가 뒤집혀 있어도 왼쪽 < 오른쪽
    y1, y2 = sorted((y1, y2))                                           # 위 < 아래
    x1, y1 = max(round(x1), 0), max(round(y1), 0)                       # 픽셀 정수로 반올림 + 화면 밖으로 나간 부분 자르기
    x2, y2 = min(round(x2), w), min(round(y2), h)   
    bw, bh = x2 - x1, y2 - y1                                           # 박스 너비, 높이
    if bw < 2 or bh < 2:                                                # convert_labels.py 와 같은 기준 (2픽셀 미만 제외)
        return None
    return SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2, width=bw, height=bh, area=bw * bh,
                           aspect=bw / bh, center=((x1 + x2)  // 2, (y1 + y2) // 2))

# =============================================================
# split 이미지 + 원본 JSON -> [(이미지 경로, [(정답 번호, 원본 signal, 박스), ...]), ...]
# =============================================================
def load_items(src, data, split):
    jsons   = {Path(p).stem: p for p in glob.glob(str(src / "02.라벨링데이터/**/*.json"), recursive=True)}
    items   = []                                                        # 반환할 목록
    missing = 0                                                         # 짝이 되는 JSON 이 없는 이미지 수
    for img in sorted((data / "images" / split).glob("*.jpg")):         # split 에 들어간 이미지 (분할은 데이터셋 그대로 사용)
        if img.stem not in jsons:
            missing += 1
            continue
        boxes, w, h = read_label(jsons[img.stem])                       # convert_labels.py 와 같은 방법으로 읽기
        gts         = []
        for kind, sig, x1, y1, x2, y2 in boxes:
            cls = class_id(kind, sig)                                   # 데이터셋과 같은 클래스 매핑
            box = make_box(x1, y1, x2, y2, w, h)
            if cls < len(STATES) and box is not None:                   # 차량 신호 상태(0~5) 박스만
                gts.append((cls, sig, box))
        items.append((img, gts))
    return items, missing

# =============================================================
# 박스별 결과 -> 전체 지표 + 혼동 행렬(클래스별 정확도 포함) 출력
# =============================================================
def report(rows, title):
    n = len(rows)
    if n == 0:                                                          # 집계할 박스가 없으면
        print(f"\n[{title}] boxes 0")
        return
    ok  = sum(r["gt"] == r["pred"] for r in rows)                       # 맞힌 박스 수
    unk = sum(r["pred"] == UNKNOWN for r in rows)                       # 판별 거부(UNKNOWN) 박스 수
    print(f"\n[{title}] boxes {n} acc {ok / n:.1%} reject {unk / n:.1%} "
          f"acc(excl. reject) {ok / max(n - unk, 1):.1%}")
    print(f"{'gt/pred':<11}" + "".join(f"{SHORT.get(c, c):>8}" for c in COLS) + f"{'n':>6}{'acc':>7}")
    for g in STATES:                                                    # 정답 클래스마다 한 줄
        cnt = Counter(r["pred"] for r in rows if r["gt"] == g)          # 이 클래스 박스의 예측 상태별 개수
        m   = sum(cnt.values())                                         # 이 클래스 박스 수
        acc = f"{cnt[g] / m:.1%}" if m else "-"                         # 박스가 없으면 정확도 표시 안 함
        print(f"{g:<11}" + "".join(f"{cnt[c]:>8}" for c in COLS) + f"{m:>6}{acc:>7}") 

# =============================================================
# 원본 signal 값별 결과 (YELLOW 에 합친 전환 프레임 등 확인용)
# =============================================================
def report_signal(rows):
    print("\nby original signal: n / acc / top predictions")
    for sig in sorted({r["signal"] for r in rows}):                     # 원본 signal 값마다
        rs = [r for r in rows if r["signal"] == sig]                    # 이 값의 박스들
        ok = sum(r["gt"] == r["pred"] for r in rs)                      # 맞힌 박스 수
        top = Counter(r["pred"] for r in rs).most_common(3)             # 많이 나온 예측 3개
        print(f"{sig:<22}{len(rs):>6}{ok / len(rs):>8.1%}   " + ", ".join(f"{p} {c}" for p, c in top))

# =============================================================
# 명령행 인자
# =============================================================
def parse_args():
    p = argparse.ArgumentParser(description="HSV 상태 판별 베이스라인 (정답 박스 기준)")
    p.add_argument("--src", required=True, help="Validation 폴더 (02.라벨링데이터가 있는 곳)")
    p.add_argument("--data", default="~/datasets/traffic_light", help="convert_labels.py 로 만든 데이터셋 폴더")
    p.add_argument("--split", default="test", choices=["train", "val", "test"], help="평가할 분할 (기준값 조정은 val)")
    p.add_argument("--csv", default=None, help="박스별 결과를 저장할 csv 경로 (없으면 저장 안 함)")
    return p.parse_args()

# =============================================================
# 평가 실행
# =============================================================
def main():
    args            = parse_args()
    src             = Path(args.src).expanduser().resolve()             # ~ 를 홈 폴더로 바꾼 절대 경로
    data            = Path(args.data).expanduser().resolve()
    items, missing  = load_items(src, data, args.split)
    clf, sel        = SignalClassifier(), TargetSelector()              # 기준값은 각 파일의 기본값 그대로 (베이스라인)

    rows            = []        # 박스별 결과
    unread          = 0         # 읽지 못한 이미지 수
    t_sum           = 0.0       # 판별에 걸린 시간 합 (초)
    for i, (img, gts) in enumerate(items, 1):
        print(f"\r{args.split} {i}/{len(items)}", end="", flush=True)   # 진행 상황 (같은 줄에 덮어 쓰기)
        if not gts:                                                     # 평가할 박스가 없는 이미지는 일기 안흥
            continue                                                    
        frame = cv2.imread(str(img))                                    # 링크를 따라가 원본 이미지 읽기
        if frame is None:                                               # 링크가 깨졌거나 파일이 손상됐으면
            unread += 1
            continue
        fh, fw      = frame.shape[:2]                                   # 이미지 높이, 너비
        for cls, sig, box in gts:
            t0      = time.perf_counter()
            r       = clf.classify(frame, box)                          # 정답 박스 -> 예측 상태
            t_sum  += time.perf_counter() - t0
            reason  = sel.reject_reason(box, fh, fw)                    # selector 1단계 탈락 이유 (통과면 None)
            rows.append({"image": img.stem, "gt": STATES[cls], "signal": str(sig), "pred": r.state,
                         "lit": r.lit, "cells": r.cells, "aspect": round(box.aspect, 2),
                         "width": box.width, "height": box.height, "selector": reason or "pass",
                         "scores": " ".join(f"{k}:{v:.3f}" for k, v in r.scores.items())})
            
    print(f"\nimages {len(items)} (no json {missing}, unreadable {unread}), boxes {len(rows)}, "
          f"{t_sum / max(len(rows), 1) * 1000:.2f} ms/box")
    print("selector:", ", ".join(f"{k} {v}" for k, v in Counter(r["selector"] for r in rows).most_common()))
    report(rows, "all")                                                 # 정답 박스 전부
    report([r for r in rows if r["selector"] == "pass"], "selector")    # selector 1단계 통과 박스만
    report_signal(rows)

    if args.csv and rows:                                               # --csv 를 줬으면 박스별 결과 저장 (오류 분석용)
        path = Path(args.csv).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)                  # 저장 폴더가 없으면 생성
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())) # 첫 줄 = 열 이름 (image, gt, signal, ...)
            writer.writeheader()
            writer.writerows(rows)
        print(f"csv -> {path}")

if __name__ == "__main__":
    main()