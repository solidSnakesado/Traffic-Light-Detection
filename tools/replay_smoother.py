"""
tools/replay_smoother.py - 4. 안정화 재생 평가 (record_states.py 의 csv 를 다시 읽어 기준값 비교, GPU 없음)
- 프레임별 원시 상태(state 열)를 영상 시각(t 열) 순서대로 smoother.py 에 넣어 확정 상태를 만들고 지표 비교
- 기준값 인자에 값을 여러 개 주면 모든 조합을 비교 (첫 번째 조합 = 기준, 그림/저장은 첫 번째 조합만)
    예: --hold 1.0 1.5 2.0 --red-hold 0 1 -> 3 x 2 = 6 조합
- 지표 (정답 없이)
    changes     : 상태가 바뀐 횟수 (원시 / 확정)
    short       : short 초 미만으로 끝난 구간 수 (흔들림)
    shown       : 신호가 있는(NONE 이 아닌) 프레임 비율
    delay       : 확정 지연 평균 (원시 상태가 새 상태로 처음 바뀐 시각 -> 확정 시각, 초)
- 지표 (--gt 구간 정답 csv 가 있을 때, 열: start, end, state / 정답 없는 시각은 NONE)
    acc         : 프레임 정확도 (원시 / 확정)
    danger      : 정답이 정지 쪽인데 출발 쪽으로 확정한 시간 (초, 가장 위험한 오류)
    false_stop  : 정답이 출발 쪽인데 정지 쪽으로 확정한 시간 (초)
- --plot <png>  : 첫 번째 조합의 타임라인 (위: 원시 / 아래: 확정 / 정답이 있으면 맨 아래)

실행 (저장소 폴더에서, 어느 PC든 가능):
    python tools/replay_smoother.py --csv ~/datasets/traffic_light/states_drive_e03.csv --plot smooth_timeline.png
    python tools/replay_smoother.py --csv <csv> --hold 1.0 1.5 2.0 --min-obs 0.3 0.15 0 --red-hold 0 1
"""

# =================== 표준 라이브러리 ===========================
import argparse                                                                                         # 명령행 인자 읽기
import csv                                                                                              # 원시 상태 csv, 정답 csv 읽기
import itertools                                                                                        # 기준값 조합 만들기
import sys                                                                                              # 모듈 검색 경로에 저장소 폴더 추가

from pathlib    import Path                                                                             # 경로 다루기

# =================== 프로젝트 모듈   ===========================
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))                                         # 저장소 폴더 (smoother.py가 있는 곳)
from smoother   import StateSmoother, SmootherConfig, STOP, GO, NONE                                    # 4. 안정화 모듈, 정지/출발 쪽 상태

KEYS    = ("t_stop", "t_go", "ratio", "min_obs", "hold", "red_hold")                                    # 비교할 기준값 이름 (SmootherConfig 필드)
COLORS  = {"RED": "#d62728", "YELLOW": "#f2b701", "GREEN": "#2ca02c", "LEFT": "#17becf",        
           "RED_LEFT": "#9467bd", "GREEN_LEFT": "#8c564b", "UNKNOWN": "#7f7f7f"}                  # 그림 색 (NONE 은 흰색 = 안 그림)           

# =============================================================
# 명령행 인자 (기준값은 여러 개 가능, 기본값 = smoother.py 시작값)
# =============================================================
def parse_args():
    d = SmootherConfig()
    p = argparse.ArgumentParser(description="안정화 재생 평가 (원시 상태 csv -> 확정 상태 지표)")
    p.add_argument("--csv", required=True, help="record_states.py 가 만든 프레임별 csv")
    p.add_argument("--gt", default=None, help="구간 정답 csv (열: start,end,state), 없으면 정답 지표 생략")
    p.add_argument("--fps", type=float, default=0, help="처리 fps (0 = csv 의 시각 간격으로 계산)")
    p.add_argument("--t-stop", type=float, nargs="+", default=[d.t_stop], help="정지 쪽 확정 시간 (초)")
    p.add_argument("--t-go", type=float, nargs="+", default=[d.t_go], help="출발 쪽 확정 시간 (초)")
    p.add_argument("--ratio", type=float, nargs="+", default=[d.ratio], help="새 상태 비율 하한")
    p.add_argument("--min-obs", type=float, nargs="+", default=[d.min_obs], help="최소 관측 수 비율")
    p.add_argument("--hold", type=float, nargs="+", default=[d.hold], help="중립 복귀 시간 (초)")
    p.add_argument("--red-hold", type=int, nargs="+", default=[0], choices=[0, 1], help="옵션 E (0 = 끔, 1 = 켬)")
    p.add_argument("--short", type=float, default=0.2, help="이 시간(초) 미만 구간을 흔들림으로 셈")
    p.add_argument("--plot", default=None, help="타임라인 그림 저장 경로 (png)")
    p.add_argument("--save", default=None, help="첫 번째 조합의 프레임별 확정 상태 csv 저장 경로")
    return p.parse_args()

# =============================================================
# 상태 목록 -> 구간 목록 [(상태, 시작 번호, 끝 번호), ...]
# =============================================================
def runs(states):
    out = []
    for i, s in enumerate(states):
        if out and out[-1][0] == s:                                                             # 같은 상태가 이어지면
            out[-1][2] = i                                                                      # 끝 번호만 늘림
        else:
            out.append([s, i, i])                                                               # 새 구간 시작
    return out

# =============================================================
# 정답 csv -> 프레임별 정답 상태 (정답 구간 밖 = NONE)
# =============================================================
def load_gt(path, ts):
    segs = [(float(r["start"]), float(r["end"]), r["state"].strip())
            for r in csv.DictReader(open(Path(path).expanduser(), encoding="utf-8"))]
    return [next((s for a, b, s in segs if a <= t <= b), NONE) for t in ts]

# =============================================================
# 조합 1개 실행 -> (확정 상태 목록, 지표 dict)
# =============================================================
def evaluate(cfg, ts, raw, gt, short):
    sm, conf    = StateSmoother(cfg), []
    for t, s in zip(ts, raw):
        conf.append(sm.update(t, s).state)                                                      # 프레임마다 확정 상태
    dt          = 1.0 / cfg.fps                                                                 # 프레임 간격 (초)
    rr, cr      = runs(raw), runs(conf)
    delays      = []                                                                            # 확정 지연 (초)
    for s, b, _ in cr[1:]:                                                                      # 확정 상태가 바뀐 구간마다
        if s == NONE:                                                                           # 중립 복귀는 지연 계산 안 함
            continue
        k = b
        while k > 0 and raw[k - 1] == s:                                                        # 원시 상태가 s 로 바뀐 처음 프레임까지 거슬러 감
            k -= 1
        delays.append((b - k) * dt)
    m = {"changes": f"{len(rr) - 1}/{len(cr) - 1}",
         "short": f"{sum((e - b + 1) * dt < short for _, b, e in rr)}/{sum((e - b + 1) * dt < short for _, b, e in cr)}",
         "shown": f"{sum(s != NONE for s in conf) / len(conf):.1%}",
         "delay": f"{sum(delays) / max(len(delays), 1):.2f}"}
    if gt:
        n = len(gt)
        m["acc"]        = f"{sum(a == g for a, g in zip(raw, gt)) / n:.1%}/{sum(a == g for a, g in zip(conf, gt)) / n:.1%}"
        m["danger"]     = f"{sum(g in STOP and c in GO for c, g in zip(conf, gt)) * dt:.2f}"
        m["false_stop"] = f"{sum(g in GO and c in STOP for c, g in zip(conf, gt)) * dt:.2f}"
    return conf, m

# =============================================================
# 타임라인 그림 (위: 원시 / 아래: 확정 / 정답)
# =============================================================
def plot(path, ts, rows, title, fps):
    import matplotlib                                                                           # 그림을 그릴 때만 필요
    matplotlib.use("Agg")                                                                       # 창 없이 파일로 저장
    import matplotlib.pyplot as plt
    dt      = 1.0 / fps                                                                         # 프레임 간격 (초)
    fig, ax = plt.subplots(figsize=(16, 1.8 + 0.6 * len(rows)))
    for y, (_, states) in enumerate(rows):
        for s, b, e in runs(states):
            if s in COLORS:
                ax.broken_barh([(ts[b], (e - b + 1) * dt)], (len(rows) - y - 0.8, 0.6), color=COLORS[s])
    ax.set_yticks([len(rows) - y - 0.5 for y in range(len(rows))], [n for n, _ in rows])
    ax.set_xlim(ts[0], ts[-1] + dt)
    ax.set_xlabel("video time (s)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()], COLORS.keys(),
              ncol=len(COLORS), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.45))    # 범례는 그림 아래
    ax.set_title(title, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(Path(path).expanduser(), dpi=110)

# =============================================================
# 평가 실행
# =============================================================
def main():
    args    = parse_args()
    rows    = list(csv.DictReader(open(Path(args.csv).expanduser(), encoding="utf-8")))
    ts      = [float(r["t"]) for r in rows]                                                     # 영상 시각
    raw     = [r["state"] for r in rows]                                                        # 원시 상태
    fps     = args.fps or ((len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else 30.0)           # 처리 fps (기록은 모든 프레임 처리 = 영상 fps)
    gt      = load_gt(args.gt, ts) if args.gt else None
    print(f"frames {len(rows)} ({ts[-1]:.1f} s), fps {fps:.1f}, gt {'yes' if gt else 'no'}")

    combos  = list(itertools.product(args.t_stop, args.t_go, args.ratio, args.min_obs, args.hold, args.red_hold))
    head    = ["changes", "short", "shown", "delay"] + (["acc", "danger", "false_stop"] if gt else [])
    print(f"\n{'t_stop':>6}{'t_go':>6}{'ratio':>6}{'min_obs':>8}{'hold':>6}{'E':>3}  "
          + "".join(f"{h:>14}" for h in head) + "   (raw/confirmed)")
    first   = None
    for c in combos:
        cfg     = SmootherConfig(**dict(zip(KEYS, c)), fps=fps)
        conf, m = evaluate(cfg, ts, raw, gt, args.short)
        first   = first or (c, conf)                                                            # 첫 번째 조합 저장 (그림/저장용)
        print(f"{c[0]:>6}{c[1]:>6}{c[2]:>6}{c[3]:>8}{c[4]:>6}{c[5]:>3}  " + "".join(f"{m[h]:>14}" for h in head))

    c, conf = first
    if args.save:                                                                               # 프레임별 확정 상태 저장
        with open(Path(args.save).expanduser(), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "t", "raw", "confirmed"] + (["gt"] if gt else []))
            for i, (t, r, s) in enumerate(zip(ts, raw, conf)):
                w.writerow([i, t, r, s] + ([gt[i]] if gt else []))
        print(f"\nsave -> {args.save}")
    if args.plot:                                                                               # 타임라인 그림
        lines = [("raw", raw), ("confirmed", conf)] + ([("gt", gt)] if gt else [])
        plot(args.plot, ts, lines, "smoother " + " ".join(f"{k}={v}" for k, v in zip(KEYS, c)) + ", white = NONE", fps)
        print(f"plot -> {args.plot}")

if __name__ == "__main__":
    main()