"""
tools/compare_states.py - 프레임별 원시 상태 csv 2개 비교 (ROS2 없이 실행)
- 기준 (--ref)  : tools/record_states.py 가 만든 csv
- 비교 (--test) : ROS2 기록 노드 raw_recorder 가 만든 csv (또는 다른 장치의 record_states csv)
- 프레임 번호로 맞춰서 비교. 같은 프레임이 여러 줄이면 마지막 줄을 씀
- 비교 열: state, n_det, n_cand, conf (차이 허용값 --conf-tol), x1~y2 (차이 허용값 --box-tol)
- 출력: 프레임 수, 빠진 프레임, state 일치율, 열별 불일치 수, 불일치 프레임 앞부분

실행 (저장소 폴더에서):
  python tools/compare_states.py --ref ~/datasets/traffic_light/states_drive_e03.csv --test ~/datasets/traffic_light/raw_node.csv
  기준 csv 의 앞부분만 비교: --limit 300
"""

# =================== 표준 라이브러리 ===========================
import argparse                                                             # 명령행 인자
import csv                                                                  # csv 읽기

from collections    import Counter                                          # 불일치 종류별 개수
from pathlib        import Path                                             # 경로 다루기

COLS    = ["state", "n_det", "n_cand", "conf", "x1", "y1", "x2", "y2"]      # 비교할 열
BOX     = ["x1", "y1", "x2", "y2"]                                          # 박스 좌표 열

# =============================================================
# 명령행 인자
# =============================================================
def parse_args():
    p = argparse.ArgumentParser(description="프레임별 원시 상태 csv 2개 비교")
    p.add_argument("--ref", required=True, help="기준 csv (record_states.py)")
    p.add_argument("--test", required=True, help="비교 csv (raw_recorder 등)")
    p.add_argument("--limit", type=int, default=0, help="기준 csv 의 앞 프레임 수만 비교 (0 = 전부)")
    p.add_argument("--conf-tol", type=float, default=0.001, help="conf 차이 허용값")
    p.add_argument("--box-tol", type=int, default=0, help="박스 좌표 차이 허용값 (px)")
    p.add_argument("--show", type=int, default=10, help="불일치 프레임을 앞에서부터 몇 개 표시할지")
    return p.parse_args()

# =============================================================
# csv -> {프레임 번호: 행}, 중복 프레임 수
# =============================================================
def load(path, limit=0):
    rows, dup = {}, 0
    with open(Path(path).expanduser(), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            i = int(r["frame"])
            if limit and i >= limit:                                        # 시험용 앞부분만
                continue
            dup += i in rows                                                # 이미 있으면 중복
            rows[i] = r                                                     # 같은 프레임이면 마지막 줄
    return rows, dup

def num(v, kind=float):
    return kind(v) if v not in ("", None) else 0                            # 빈 값 (대상 없음) = 0

# =============================================================
# 행 2개 -> 다른 열 목록, 박스 차이 최대값, conf 차이
# =============================================================
def diff_row(a, b, conf_tol, box_tol):
    bad = ["state"] if a["state"] != b["state"] else []                     # 원시 상태
    for c in ("n_det", "n_cand"):
        if num(a[c], int) != num(b[c], int):
            bad.append(c)
    dc = abs(num(a["conf"]) - num(b["conf"]))
    if dc > conf_tol:
        bad.append("conf")
    db = max(abs(num(a[c], int) - num(b[c], int)) for c in BOX)
    if db > box_tol:
        bad.append("box")
    return bad, db, dc

# =============================================================
# 비교 실행
# =============================================================
def main():
    args        = parse_args()
    ref, dup_r  = load(args.ref, args.limit)
    test, dup_t = load(args.test, args.limit)
    common      = sorted(set(ref) & set(test))                              # 둘 다 있는 프레임
    missing     = sorted(set(ref) - set(test))                              # 비교 csv 에 없는 프레임
    extra       = sorted(set(test) - set(ref))                              # 기준 csv 에 없는 프레임
    kinds, bad_frames = Counter(), []   
    max_box, max_conf, same_state = 0, 0.0, 0
    for i in common:
        bad, db, dc = diff_row(ref[i], test[i], args.conf_tol, args.box_tol)
        max_box, max_conf = max(max_box, db), max(max_conf, dc)
        same_state += "state" not in bad
        kinds.update(bad)
        if bad:
            bad_frames.append((i, bad))

    # ===== 결과 출력 =====
    n = max(len(common), 1)
    print(f"ref {args.ref}: {len(ref)} frames (dup {dup_r})")
    print(f"test {args.test}: {len(test)} frames (dup {dup_t})")
    print(f"common {len(common)}, missing in test {len(missing)}, extra in test {len(extra)}")
    if missing:
        print(f"  missing (first {args.show}): {missing[:args.show]}")
    print(f"state match {same_state}/{len(common)} ({same_state / n:.2%})")
    print(f"all columns match {len(common) - len(bad_frames)}/{len(common)} "
          f"({(len(common) - len(bad_frames)) / n:.2%})")
    print("mismatch by column: " + (", ".join(f"{k} {v}" for k, v in kinds.most_common()) or "none"))
    print(f"max box diff {max_box} px, max conf diff {max_conf:.4f}")
    for i, bad in bad_frames[:args.show]:                                   # 불일치 프레임 앞부분
        r, t = ref[i], test[i]
        print(f"  frame {i}: {', '.join(bad)} | ref {r['state']} n_det {r['n_det']} conf {r['conf']} "
              f"box {[r[c] for c in BOX]} | test {t['state']} n_det {t['n_det']} conf {t['conf']} "
              f"box {[t[c] for c in BOX]}")

if __name__ == "__main__":
    main()