"""
tools/remap_classes.py - 9클래스 검출 데이터셋 -> 2클래스(vehicle / other) 검출 데이터셋 변환
- 입력: convert_labels.py 로 만든 데이터셋 (<data>/images/{train,val,test}, <data>/labels/{train,val,test}, 9클래스)
- 출력: <out>/images/{train,val,test}/*.jpg (원본 이미지 링크), <out>/labels/{train,val,test}/*.txt, <out>/data.yaml
- 클래스 변환 (분할/이미지/박스 좌표는 그대로)
    0~5 (RED, YELLOW, GREEN, LEFT, RED_LEFT, GREEN_LEFT) -> 0 vehicle (차량 신호등)
    6~8 (invisible, pedestrian, other)                   -> 1 other   (그 외 신호등)
- 박스가 없는 이미지도 빈 txt 로 그대로 둠 (배경 학습용)
- 이미지는 입력 데이터셋의 링크가 가리키는 원본을 다시 가리킴 (용량 거의 0)
        Ultralytics 는 이미지 경로의 /images/ 를 /labels/ 로 바꿔 라벨을 찾으므로 새 폴더 안에 둘 다 만듦

실행 (저장소 폴더에서):
    python tools/remap_classes.py --data ~/datasets/traffic_light --out ~/datasets/traffic_light_vehicle_other
"""

# =================== 표준 라이브러리 ===========================
import argparse                                             # 명령행 인자(--data, --out 등) 읽기
import os                                                   # 링크 만들기, 링크 대상 찾기

from pathlib import Path                                    # 경로 다루기

SPLITS      = ("train", "val", "test")                      # 분할 이름 (입력 데이터셋과 동일)
NAMES       = ("vehicle", "other")                          # 새 클래스 이름 (번호 = 순서)
N_SRC_CLS   = 9                                             # 입력 데이터셋 클래스 수 (0~8)
N_VEHICLE   = 6                                             # 0~5 가 차량 신호등

# =============================================================
# 9클래스 번호 -> 2클래스 번호
# =============================================================
def map_class(cls):
    if 0 <= cls < N_VEHICLE:                                # 0~5: 차량 신호등
        return 0        
    if N_VEHICLE <= cls < N_SRC_CLS:                        # 6~8: 보행자/측면/기타
        return 1
    return None                                             # 범위 밖 번호 (버림, 개수만 셈)

# =============================================================
# 라벨 txt 1개 변환: 줄마다 첫 숫자(클래스)만 바꾸고 좌표는 그대로
# =============================================================
def convert_label(src, counts):
    lines   = []                                            # 변환된 줄 목록
    dropped = 0                                             # 범위 밖 클래스로 버린 줄 수
    if not src.exists():                                    # 라벨 파일이 없으면 박스 없음으로 처리
        return lines, dropped, False
    for line in src.read_text().splitlines():
        parts = line.split()
        if not parts:                                       # 빈 줄은 건너뜀
            continue
        new = map_class(int(parts[0]))                      # 클래스 번호 변환
        if new is None:
            dropped += 1
            continue
        counts[new] += 1                                    # 새 클래스별 박스 수
        lines.append(" ".join([str(new)] + parts[1:]))      # 좌표(cx cy w h)는 그대로
    return lines, dropped, True

# =============================================================
# 이미지 링크 만들기 (다시 실행해도 덮어씀)
# =============================================================
def place_image(src, dst, copy):
    target = os.path.realpath(src)                          # 입력 링크가 가리키는 원본 (링크의 링크 방지)
    if dst.is_symlink() or dst.exists():                    # 이전 실행 결과가 있으면 지움
        dst.unlink()
    if copy:
        dst.write_bytes(Path(target).read_bytes())          # 복사 (용량 필요)
    else:
        os.symlink(target, dst)                             # 원본을 가리키는 링크

# =============================================================
# 명령행 인자
# =============================================================
def parse_args():
    p = argparse.ArgumentParser(description="9클래스 검출 데이터셋 -> 2클래스(vehicle / other) 변환")
    p.add_argument("--data", default="~/datasets/traffic_light", help="convert_labels.py 로 만든 9클래스 데이터셋 폴더")
    p.add_argument("--out", default="~/datasets/traffic_light_vehicle_other", help="출력 폴더")
    p.add_argument("--copy", action="store_true", help="링크 대신 이미지 복사")
    return p.parse_args()

# =============================================================
# 변환 실행
# =============================================================
def main():
    args    = parse_args()
    data    = Path(args.data).expanduser().resolve()        # ~ 를 홈 폴더로 바꾼 절대 경로
    out     = Path(args.out).expanduser().resolve()         
    if data == out:                                         # 입력을 덮어쓰지 않도록 막음
        raise SystemExit("--out 은 --data 와 다른 폴더로 지정되어야 함")

    total   = {s: [0] * len(NAMES) for s in SPLITS}         # 분할별/클래스별 박스 수
    n_img   = {s: 0 for s in SPLITS}                        # 분할별 이미지 수
    n_empty = {s: 0 for s in SPLITS}                        # 분할별 박스 없는 이미지 수
    no_label, dropped = 0, 0                                # 라벨 파일 없는 이미지 수, 버린 줄 수

    for split in SPLITS:
        src_img = data / "images" / split                   # 입력 이미지 폴더
        src_lbl = data / "labels" / split                   # 입력 라벨 폴더
        if not src_img.is_dir():
            raise SystemExit(f"이미지 폴더 없음: {src_img}")
        for kind in ("images", "labels"):                   # 출력 폴더 만들기
            (out / kind / split).mkdir(parents=True, exist_ok=True)  

        for img in sorted(src_img.glob("*.jpg")):
            lines, d, ok    = convert_label(src_lbl / f"{img.stem}.txt", total[split])
            dropped        += d
            no_label       += 0 if ok else 1
            n_empty[split] += 0 if lines else 1
            (out / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            place_image(img, out / "images" / split / img.name, args.copy)
            n_img[split] += 1

        cache = out / "labels" / f"{split}.cache"           # 이전 학습이 만든 라벨 캐시 (내용이 변경되었으므로 지움)
        if cache.exists():
            cache.unlink()

    # data.yaml: 학습할 때 model.train(data="<out>/data.yaml") 로 넘기는 설정 파일
    yaml    = [f"path: {out}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml   += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n")

    # ===== 결과 출력 (기존 집계와 비교: train vehicle 20,108 / other 26,208) =====
    print(f"{'split':<6} {'images':>7} {'empty':>6} {'vehicle':>8} {'other':>7}")
    for s in SPLITS:
        print(f"{s:<6} {n_img[s]:>7} {n_empty[s]:>6} {total[s][0]:>8} {total[s][1]:>7}")
    print(f"no label file {no_label} / dropped line {dropped}")
    print(f"data.yaml -> {out / 'data.yaml'}")

if __name__ == "__main__":
    main()