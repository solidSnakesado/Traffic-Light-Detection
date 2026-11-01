"""
smoother.py - 4. 신호 상태 안정화 모듈 (프레임마다 흔들리는 target_state -> 확정 상태)
- 입력: 프레임마다 (영상 시각 t 초, selector 의 target_state)
    유효 관측: RED, YELLOW, GREEN, LEFT, RED_LEFT, GREEN_LEFT
    무관측: UNKNOWN (켜진 후보 없음), NONE (후보 없음) -> 투표에 넣지 않고 확정 상태 유지
- 출력: 확정 상태 (위 6개 또는 NONE = 따라야 할 신호 없음, 중립) 
- 방법: (C: 시간 창 비율 투표 + D: 비대칭 확정)
    새 상태 X의 확정 시간 T(X): 정지 쪽(RED, YELLOW, RED_LEFT) 은 짧게, 출발 쪽 (GREEN, GREEN_LEFT, LEFT) 은 길게
    X 가 T(X) 초 이상 보였고, 최근 T(X) 초 유효 관측 중 X 비율 >= ratio, X 개수 >= 최소 관측 수 이면 확정
    마지막 유효 관측 후 hold 초가 지나면 NONE (중립) 으로 복귀
- 옵션 E (기본 꺼짐): 확정 RED 에서 YELLOW 로는 바꾸지 않음 (적색 다음 황색은 정상 순서가 아님)
- 기준값은 모두 초 단위 -> 처리 속도와 무관, 최소 관측 수만 fps (실제 처리 fps) 로 계산
- cv2, YOLO 를 쓰지 않음 -> 단독 시험 가능, selector/classifier 와 같은 상태 문자열만 사용
"""

# =================== 표준 라이브러리 ===========================
import math     # 최소 관측 수 올림 계산

from collections import deque                                                   # 최근 관측 창 (오래전 것부터 버림)
from dataclasses import dataclass                                               # 설정값/결과를 담을 데이터 클래스

# ======== 상태 이름 (classifier.py, selector.py 와 같음) =======
UNKNOWN, NONE   = "UNKNOWN", "NONE"                                             # 판별 불가 / 대상 없음
STOP            = ("RED", "YELLOW", "RED_LEFT")                                 # 정지 쪽: 적색 또는 황색이 켜짐 -> 빠르게 확정
GO              = ("GREEN", "GREEN_LEFT", "LEFT")                               # 출발 쪽: 녹색 계열만 켜짐 -> 천천히 확정
STATES          = STOP + GO                                                     # 투표에 넣는 유효 상태 6개

# =============================================================
# 안정화 기준값 (2026-11-01 결정한 시작 값)
# =============================================================
@dataclass
class SmootherConfig:
    t_stop:     float   = 0.2                                                   # 정지 쪽 확정 시간 (초)
    t_go:       float   = 0.5                                                   # 출발 쪽 확정 시간 (초)
    ratio:      float   = 0.7                                                   # 창 안 유효 관측 중 새 상태 비율 하한
    min_obs:    float   = 0.3                                                   # 최소 관측 수 = 창 프레임 수(T x fps) x 이 비율
    hold:       float   = 1.5                                                   # 마지막 유효 관측 후 이 시간이 지나면 NONE 으로 복귀 (초)
    fps:        float   = 60.0                                                  # 실제 처리 fps (최소 관측 수 계산용)
    red_hold:   bool    = False                                                 # 옵션 E: 확정 RED -> YELLOW 전환 막기

# =============================================================
# 프레임 1장의 안정화 결과
# =============================================================
@dataclass
class SmoothResult:
    state:      str     = NONE                                                  # 확정 상태 (5. 명령 변환에 넘기는 값)
    raw:        str     = NONE                                                  # 이번 프레임의 원시 상태 (입력 그대로)
    changed:    bool    = False                                                 # 이번 프레임에 확정 상태가 바뀌었는 지
    since:      float   = math.inf                                              # 마지막 유효 관측 후 지난 시간 (초, 없으면 inf)

# =============================================================
# 원시 상태 -> 확정 상태로 바꿔 주는 클래스 (프레임마다 update 호출)
# =============================================================
class StateSmoother:
    def __init__(self, config=None):
        self.cfg = config if config is not None else SmootherConfig()           # 설정에 안 넘기면 기본값 사용
        self.reset()

    def reset(self):
        """처음 상태로 (새 영상 시작 시)"""
        self.state      = NONE                                                  # 확정 상태
        self.window     = deque()                                               # 최근 유효 관측 (시각, 상태)
        self.last_valid = None                                                  # 마지막 유효 관측 시각
        self.first_seen = {}                                                    # 상태별: 끊기지 않고 보이기 시작한 시각
        self.last_seen  = {}                                                    # 상태별: 마지막으로 보인 시각

    def confirm_time(self, s):
        """상태 s 로 바꿀 때 필요한 확정 시간 (초)"""
        return self.cfg.t_stop if s in STOP else self.cfg.t_go

    def ready(self, s, t):
        """새 상태 s 를 지금 확정해도 되는 지 검사"""
        c, T    = self.cfg, self.confirm_time(s)
        if c.red_hold and self.state == "RED" and s == "YELLOW":                # 옵션 E: 적색 다음 황색은 받지 않음
            return False
        if t - self.first_seen[s] < T - 1e-9:                                   # s 가 아직 T초 동안 보이지 않음
            return False
        recent  = [v for ti, v in self.window if t - ti < T - 1e-9]             # 최근 T 초의 유효 관측
        n       = sum(v == s for v in recent)                                   # 그중 s 개수
        need    = max(1, math.ceil(c.min_obs * T * c.fps - 1e-9))               # 최소 관측 수 (0.2초 x 60fps x 0.3 = 4) 
        return n >= need and n >= c.ratio * len(recent)                         # 개수와 비율 모두 만족

    def update(self, t, raw) -> SmoothResult:
        """프레임 1장: 영상 시각 t (초) + 원시 상태 -> SmoothResult"""
        c, prev = self.cfg, self.state
        if raw in STATES:                                                       # 유효 관측이면
            if raw not in self.last_seen or t - self.last_seen[raw] > c.hold:   # 처음 보였거나 오래 끊겼다 다시 보이면
                self.first_seen[raw] = t                                        # 보이기 시작한 시각을 새로 기록
            self.last_seen[raw] = t
            self.last_valid     = t
            self.window.append((t, raw))                                        # 창에 추가

        keep = max(c.t_stop, c.t_go)                                            # 창 길이 = 가장 긴 확정 시간
        while self.window and t - self.window[0][0] >= keep:                    # 오래된 관측 버리기
            self.window.popleft()

        since = t - self.last_valid if self.last_valid is not None else math.inf    
        if since > c.hold:                                                      # 유효 관측이 오래 없으면
            self.state = NONE                                                   # 중립으로 복귀
            self.first_seen.clear()                                             # 다음 신호등은 처음부터 다시 확인
            self.last_seen.clear()
        elif raw in STATES and raw != self.state and self.ready(raw, t):        # 새 상태가 조건을 만족하면
            self.state = raw                                                    # 확정 상태 변경

        return SmoothResult(state=self.state, raw=raw, changed=self.state != prev, since=since)