"""
tools/convert_labels.py - AI Hub 신호등 라벨(JSON) -> YOLO 학습 데이터셋 변환
- 입력: AI Hub Validation 폴더 (01.원천데이터 = 이미지, 02.라벨링데이터 = JSON)
- 출력: dataset/images/{train,val,test}/*.jpg, dataset/labels/{train,val,test}/*.txt, dataset/data.yaml
- 라벨 형식 2가지를 모두 읽음
    AI Hub 형식: objects[].class_name / data     [[x1,y1],[x2,y2]] / attribute.signal / image_size
    LabelMe 형식: shapes[].label     / points   [[x1,y1],[x2,y2]] / flags.signal     / imageWidth, imageHeight
- train/val/test 는 Clip 폴더(장면) 단위로 나눔 -> 거의 같은 연속 프레임이 양쪽에 섞여 성능이 부풀려지는 것 방지
- test 는 학습·모델 선택에 쓰지 않는 최종 평가 전용 (베이스라인과 학습 모델을 같은 기준으로 비교)
- 다시 실행하면 이전 분할 결과를 지우고 새로 만듦 (분할이 바뀌어도 이전 파일이 남지 않도록)
- 이미지는 기본으로 심볼릭 링크 (원본을 가리키는 바로가기, 디스크 절약), --copy를 주면 실제 복사

실행: python tools/convert_labels.py --src <Validation 폴더 경로> --out dataset
"""

# =================== 표준 라이브러리 ===========================
import argparse                                                         # 명령행 인자(--src, --out 등) 읽기
import glob                                                             # 패턴으로 파일 목록 찾기 (**: 하위 폴더 전체)
import json                                                             # 라벨 JSON 읽기
import os                                                               # 심볼릭 링크 만들기, 기존 파일 지우기
import random                                                           # Clip 폴더를 섞어 train/val/test 로 나누기
import shutil                                                           # --copy 일 때 이미지 복사, 이전 결과 폴더 지우기

from collections import Counter                                         # 클래스별 박스 수 세기
from pathlib import Path                                                # 경로 다루기 (폴더 이름, 확장자 분리)

# ================ 클래스 정의 (번호 = 리스트 순서) ================
NAMES = ["RED", "YELLOW", "GREEN", "LEFT", "RED_LEFT", "GREEN_LEFT",    # 0~5: 차량 신호 상태 (classifier.py 상태 이름과 같음)
         "invisible", "pedestrian", "other"]                            # 6: 주행 방향이 아닌 신호등 / 7. 보행자 신호 / 8. 기타
OTHER = NAMES.index("other")                                            # 기타 클래스 번호 (8)

# 차량 신호등(vehicular_signal)의 signal 값 -> 클래스 번호 (표에 없는 값은 other)
VEHICLE_SIGNAL = {
    "red":                      0,                                      # 적색
    "yellow":                   1,                                      # 황색
    "red and yellow":           1,                                      # 적+황 전환 프레임 -> 황색
    "green and yellow":         1,                                      # 녹+황 전환 프레임 -> 황색
    "green":                    2,                                      # 녹색
    "green arrow":              3,                                      # 좌회전 화살표
    "red and green arrow":      4,                                      # 적색 + 화살표
    "green and green arrow":    5,                                      # 녹색 + 화살표
}                                                                       # etc, 화살표 2개, 아래 화살표 -> other

# 차량 신호등이 아닌 종류(class_name) -> 클래스 번호 (unusual_signal 등 나머지는 other)
OTHER_KIND = {"invisible_signal": 6, "pedestrian_signal": 7}

# =============================================================
# 종류 + 신호 값 -> 클래스 번호
# =============================================================
def class_id(kind, signal):
    if kind == "vehicular_signal":                                      # 차량 신호등이면
        return VEHICLE_SIGNAL.get(signal, OTHER)                        # 신호값으로 상태 클래스 결정
    return OTHER_KIND.get(kind, OTHER)                                  # 아니면 종류로 결정

# =============================================================
# 라벨 JSON 1개 읽기 (두 형식 모두) -> (박스 목록, 이미지 너비, 높이)
# =============================================================
def read_label(path):
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    if "objects" in j:                                                  # AI Hub 형식
        w, h = j["image_size"]                                          # [너비, 높이]
        items = [(o["class_name"], (o.get("attribute") or {}).get("signal"), o["data"]) 
                 for o in j["objects"]]
    else:                                                               # LabelMe 형식
        w, h = j["imageWidth"], j["imageHeight"]
        items = [(s["label"], (s.get("flags") or {}).get("signal"), s["points"])
                 for s in j["shapes"]]
    boxes = [(kind, sig, pts[0][0], pts[0][1], pts[1][0], pts[1][1])    # (종류, 신호, x1, y1, x2, y2)
             for kind, sig, pts in items]
    return boxes, w, h

# =============================================================
# 픽셀 박스 -> YOLO 한 줄 "클래스 x중심 y중심 너비 높이" (0~1 비율)
# =============================================================
def to_yolo(cls, x1, y1, x2, y2, w, h):
    x1, x2 = sorted((x1, x2))                                           # 점 순서가 뒤집혀 있어도 왼쪽 < 오른쪽
    y1, y2 = sorted((y1, y2))                                           # 위 < 아래
    x1, y1 = max(x1, 0), max(y1, 0)                                     # 화면 밖으로 나간 부분 자르기
    x2, y2 = min(x2, w), min(y2, h)
    if x2 - x1 < 2 or y2 - y1 < 2:                                      # 2픽셀 미만이면 박스로 쓸 수 없음
        return None
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h                       # 중심 좌표 (이미지 크기 대비 비율)
    bw, bh = (x2 - x1) / w, (y2 - y1) / h                               # 너비, 높이 (비율)
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

# =============================================================
# 이미지를 데이터셋 폴더에 넣기 (링크 또는 복사)
# =============================================================
def place_image(src, dst, copy):
    if os.path.lexists(dst):                                            # 다시 실행할 때 이전 파일/링크가 있으면
        os.remove(dst)                                                  # 지우고 새로 생성

    if copy:
        shutil.copy2(src, dst)                                          # 실제 복사 (용량 2배)
    else:
        os.symlink(src, dst)                                            # 원본을 가리키는 링크 (용량 거의 0)

# =============================================================
# 명령행 인자
# =============================================================
def parse_args():
    p = argparse.ArgumentParser(description="AI Hub 신호등 라벨 -> YOLO 데이터셋 변환")
    p.add_argument("--src", required=True, help="Validation 폴더 (01.원천데이터, 02.라벨링데이터 가 있는 곳)")
    p.add_argument("--out", default="dataset", help="출력 폴더")
    p.add_argument("--val", type=float, default=0.15, help="val 로 쓸 Clip 폴더 비율 (0~1)")
    p.add_argument("--test", type=float, default=0.15, help="test 로 쓸 Clip 폴더 비율 (0~1)")
    p.add_argument("--seed", type=int, default=0, help="Clip 섞기 난수 시드 (같으면 항상 같은 분할)")
    p.add_argument("--copy", action="store_true", help="링크 대신 이미지 복사")
    return p.parse_args()

# =============================================================
# 변환 실행
# =============================================================
def main():
    args = parse_args()
    src = Path(args.src).resolve()                                      # 절대 경로 (링크가 어디서든 원본을 찾도록)
    out = Path(args.out).resolve()

    # 이미지 이름(확장자 제외) -> 이미지 경로 (카메라 이미지만, Lidar 제외)
    images = {Path(p).stem: p for p in glob.glob(str(src / "01.원천데이터/**/Camera/**/*.jpg"), recursive=True)}
    labels = sorted(glob.glob(str(src / "02.라벨링데이터/**/*.json"), recursive=True))

    # 라벨 경로에서 Clip 폴더 이름 찾기 (예: Clip_0584)
    clip_of = {f: next(x for x in Path(f).parts if x.startswith("Clip_")) for f in labels}

    clips = sorted(set(clip_of.values()))                               # Clip 폴더 목록
    random.Random(args.seed).shuffle(clips)                             # 시드 고정으로 섞기
    n_test = round(len(clips) * args.test)                              # test 로 쓸 Clip 수
    n_val  = round(len(clips) * args.val)                               # val 로 쓸 Clip 수
    split_of = {c: "test" for c in clips[:n_test]}                      # 맨 앞은 test
    split_of.update({c: "val" for c in clips[n_test:n_test + n_val]})   # 그 다음은 val (나머지는 train)

    splits = ("train", "val", "test")
    for split in splits:                                                # 출력 폴더 만들기
        for kind in ("images", "labels"):
            shutil.rmtree(out / kind / split, ignore_errors=True)       # 이전 결과 지우기 (링크만 지워지고 원본은 그대로)
            (out / kind / split).mkdir(parents=True, exist_ok=True)

    count = {s: Counter() for s in splits}                              # split 별 클래스 박스 수
    n_img = Counter()                                                   # split 별 이미지 수
    missing, dropped = 0, 0                                             # 이미지 없는 라벨 수, 너무 작은 박스 수

    for f in labels:
        stem = Path(f).stem         # 예: 512_RD_1999_CF_048
        if stem not in images:      # 짝이 되는 이미지가 없으면
            missing += 1
            continue                # 건너뜀
        split = split_of.get(clip_of[f], "train")                        # Clip 이 속한 분할

        boxes, w, h = read_label(f)
        lines = []
        for kind, sig, x1, y1, x2, y2 in boxes:
            cls     = class_id(kind, sig)
            line    = to_yolo(cls, x1, y1, x2, y2, w, h)
            if line is None:
                dropped += 1
                continue
            lines.append(line)
            count[split][cls] += 1

        # 라벨 txt 저장 (박스가 없어도 빈 파일 -> 배경 이미지로 학습)
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        place_image(images[stem], out / "images" / split / f"{stem}.jpg", args.copy)
        n_img[split] += 1

    # data.yaml: 학습할 때 model.train(data="dataset/data.yaml") 로 넘기는 설정 파일
    yaml = [f"path: {out}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n")

    # 결과 요약 출력
    print(f"clips train {len(clips) - n_val - n_test} / val {n_val} / test {n_test}")
    print(f"images train {n_img['train']} / val {n_img['val']} / test {n_img['test']} (missing image {missing}, dropped box {dropped})")
    for i, n in enumerate(NAMES):
        print(f"{i} {n:<11} train {count['train'][i]:>6} val {count['val'][i]:>6} test {count['test'][i]:>6}")
    print(f"data.yaml -> {out / 'data.yaml'}") 

if __name__ == "__main__":
    main()