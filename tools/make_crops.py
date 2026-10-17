"""
tools/make_crops.py - 정답 박스를 잘라 상태 분류 모델(B 구조: yolo26n-cls)용 데이터셋 만들기
- 입력: convert_labels.py 로 만든 데이터셋의 split 이미지 + 원본 라벨 JSON (eval_hsv.py 의 load_items 그대로 사용)
- 대상: 차량 신호 상태 박스(클래스 0~5)만 -> HSV 베이스라인(eval_hsv.py)과 완전히 같은 박스
- 출력: <out>/{train,val,test}/<상태 이름>/<이미지 이름>_<번호>.jpg
  (Ultralytics 분류 데이터셋 형식: 폴더 이름 = 클래스 이름)
- 잘라낸 박스는 회색 여백을 붙여 정사각형으로 만듦
  (분류 모델은 짧은 변 기준으로 줄인 뒤 가운데를 정사각형으로 자름 -> 가로로 긴 신호등은 양 끝 적색/녹색 램프가 잘림)
- 원본 해상도 그대로 저장 (크기 조정은 학습할 때 모델이 함)
- 다시 실행하면 이전 결과를 지우고 새로 만듦

실행 (저장소 폴더에서):
  python tools/make_crops.py --src <Validation 폴더>
"""

# =================== 표준 라이브러리 ===========================
import argparse                                                         # 명령행 인자(--src, --out 등) 읽기
import shutil                                                           # 이전 결과 폴더 지우기

from collections    import Counter                                      # split 별 클래스 개수 세기
from pathlib        import Path                                         # 경로 다루기

# =================== 외부 라이브러리 ===========================
import cv2                                                              # 이미지 읽기, 여백 붙이기, 저장

# =================== 프로젝트 모듈   ===========================
from eval_hsv       import STATES, load_items                           # 같은 tools 폴더: 상태 이름 6개, 이미지별 정답 박스 읽기

SPLITS  = ("train", "val", "test")                                      # 만들 분할 (test 는 최종 비교 전용, 학습에 쓰지 않음)
PAD     = (114, 114, 114)                                               # 여백 색 (회색, YOLO 기본 여백 색과 같음)

# =============================================================
# 프레임 + 박스 1개 -> 정사각형으로 여백을 붙인 잘라낸 이미지 (eval_cls.py 에서도 같은 함수 사용)
# =============================================================
def square_crop(frame, box):
    crop    = frame[box.y1:box.y2, box.x1:box.x2]                       # 박스 영역만 자르기 (박스는 이미 화면 안으로 잘려 있음)
    h, w    = crop.shape[:2]                                            # 잘라낸 높이, 너비
    s       = max(h, w)                                                 # 정사각형 한 변 = 긴 변
    top     = (s - h) // 2                                              # 위쪽 여백 (나머지는 아래쪽)
    left    = (s - w) // 2                                              # 왼쪽 여백 (나머지는 오른쪽)
    return cv2.copyMakeBorder(crop, top, s - h - top, left, s - w - left,
                              cv2.BORDER_CONSTANT, value=PAD)           # 가운데에 신호등, 남는 곳은 회색

# =============================================================
# 명령행 인자
# =============================================================
def parse_args():
    p = argparse.ArgumentParser(description="정답 박스 -> 상태 분류용 데이터셋")
    p.add_argument("--src", required=True, help="Validation 폴더 (02.라벨링데이터 가 있는 곳)")
    p.add_argument("--data", default="~/datasets/traffic_light", help="convert_label.py 로 만든 데이터셋 폴더")
    p.add_argument("--out", default="~/datasets/traffic_light_cls", help="출력 폴더")
    return p.parse_args()

# =============================================================
# 데이터셋 만들기
# =============================================================
def main():
    args    = parse_args()
    src     = Path(args.src).expanduser().resolve()                     # ~ 를 홈 폴더로 바꾼 절대 경로
    data    = Path(args.data).expanduser().resolve()
    out     = Path(args.out).expanduser().resolve()

    count   = {s: Counter() for s in SPLITS}                            # split 별 상태별 이미지 수
    missing = 0                                                         # 짝이 되는 JSON이 없는 이미지 수
    unread  = 0                                                         # 읽지 못한 이미지수

    for split in SPLITS:
        shutil.rmtree(out / split, ignore_errors=True)                  # 이전 결과 지우기
        for name in STATES:
            (out / split / name).mkdir(parents=True, exist_ok=True)     # 상태마다 폴더 (박스가 0개여도 폴더는 만듦)

        items, miss = load_items(src, data, split)                      # eval_hsv.py 와 같은 방법으로 정답 박스 읽기
        missing    += miss
        for i, (img, gts) in enumerate(items, 1):
            print(f"\r{split} {i}/{len(items)}", end="", flush=True)    # 진행 상황 (같은 줄에 덮어쓰기)
            if not gts:                                                 # 상태 박스가 없는 이미지는 읽지 않음
                continue
            frame = cv2.imread(str(img))                                # 링크를 따라가 원본 이미지 읽기
            if frame is None:                                           # 링크가 깨졌거나 파일이 손상됐으면
                unread += 1
                continue
            for k, (cls, sig, box) in enumerate(gts):                   # 박스마다 1장씩
                name = STATES[cls]                                      # 상태 이름 = 저장 폴더 이름
                path = out / split / name / f"{img.stem}_{k:02d}.jpg"   # 예: 512_ND_0000_CF_003_00.jpg
                cv2.imwrite(str(path), square_crop(frame, box), [cv2.IMWRITE_JPEG_QUALITY, 95])
                count[split][name] += 1
        print()                                                         # 진행 상황 줄 끝내기

    # 결과 요약 출력
    print(f"no json {missing}, unreadable {unread}")
    for name in STATES:
        print(f"{name:<11} train {count['train'][name]:>6} val {count['val'][name]:>6} test {count['test'][name]:>6}")
    print(f"total       train {sum(count['train'].values()):>6} val {sum(count['val'].values()):>6} "
          f"test {sum(count['test'].values()):>6}")
    print(f"dataset -> {out}")

if __name__ == "__main__":
    main()