"""
classifier.py - 3. 신호 상태 판별 모듈 (HSV 색 규칙, 학습 없음)
- 신호등 박스(Detection) 1개와 프레임을 받아 신호 상태 1개를 반환
- 방법: 박스 영역을 잘라 HSV로 변환 -> 가로로 칸을 나눔 -> 칸마다 "그 칸의 색"의 켜진 픽셀 비율 계산
       -> 켜진 칸들의 조합으로 상태 결정 (색 자체 보다 "어느 칸이 켜졌나"가 우선)
- 한국 가로형 차량 신호등 배열 (왼쪽 -> 오른쪽)
    3구: 적 | 황 | 녹
    4구: 적 | 황 | 녹(좌회전) | 녹
- 3구/4구 구분: 박스 가로/세로 비율 (시험 영상 측정 값: 3구 2.0~3.1, 4구 3.0~3.9 -> 기준 3.0)
  가까이서 찍힌 3구가 4구로 잡혀도 틀리지 않도록 녹색은 칸이 아니라 "녹색 덩어리 중심"으로 화살표/녹 을 구분
- 켜진 칸이 없거나 조합이 맞지 않으면 UNKNOWN
  (꺼진 신호등 뒷면/측면, 노란 간판, LED 깜빡임으로 어둡게 찍힌 프레임 -> 4. 안정화 단계에서 진행)
- detector.py 를 import 하지 않음 -> x1, y1, x2, y2, aspect 속성만 있으면 동작 (YOLO 없이 단독 시험 가능) 

이 스크립트는 CNN 학습 전(1단계)에 쓰는 임시 판별 기능
"""

# =================== 표준 라이브러리 ===========================
from dataclasses import dataclass, field                                                    # 설정값/결과를 담을 데이터 클래스 도구

# =================== 외부 라이브러리 ===========================
import cv2                                                                                  # OpenCV: BGR -> HSV 색 공간 변환, 덩어리(연결 요소) 찾기
import numpy as np                                                                          # 픽셀 좌표/개수 계산

# =================== 출력 상태 이름 ===========================
RED, YELLOW, GREEN, LEFT    = "RED", "YELLOW", "GREEN", "LEFT"                              # 적색 / 황색 / 녹색 / 녹색 좌회전 화살표
RED_LEFT, GREEN_LEFT        = "RED_LEFT", "GREEN_LEFT"                                      # 적색 + 화살표 / 녹색 + 화살표 (4구 동시 점등)
UNKNOWN, NONE               = "UNKNOWN", "NONE"                                             # 판별 불가 / 대상 신호등 없음

# 켜진 칸 조합 -> 상태 (칸 이름 R: 적, Y: 황, L: 좌회전 화살표, G: 녹)
STATE_TABLE = {
    frozenset("R"):     RED,
    frozenset("Y"):     YELLOW,
    frozenset("G"):     GREEN,
    frozenset("L"):     LEFT,
    frozenset("RL"):    RED_LEFT,
    frozenset("GL"):    GREEN_LEFT,
    frozenset("YL"):    YELLOW,                                                             # 황색과 화살표가 함께 보이는 전환 프레임 -> 황색(정지 준비) 우선
}                                                                                           # 표에 없는 조합(적 + 녹, 적 + 황 등)은 UNKNOWN

# =============================================================
# 판별 기준값 모음 (OpenCV HSV: H 0~179, S 0~255, V 0~255)
# =============================================================
@dataclass
class ClassifierConfig:
    s_min:          int     = 90                                                            # 적/황 램프 최소 채도: 햇빛 반사로 하얗게 보이는 꺼진 램프 제외
    v_min:          int     = 150                                                           # 적/황 램프 최소 밝기: 꺼진 램프, 검은 하우징 제외
    g_s_min:        int     = 60                                                            # 녹색/화살표 최소 채도 (화살표는 원형 램프보다 흐리게 찍힘)
    g_v_min:        int     = 120                                                           # 녹색/화살표 최소 밝기
    warm_max:       int     = 35                                                            # 적/황 계열 H 범위 1: 0~35 (주황으로 찍히는 적색 포함)
    warm_wrap:      int     = 155                                                           # 적/황 계열 H 범위 2: 155~179 (분홍~빨강, H는 원형이라 179 다음이 0)
    green_lo:       int     = 68                                                            # 녹색 계열 H 하한 (녹색 램프 약 70~95, 순녹색(60) 간판/글자 제외)
    green_hi:       int     = 97                                                            # 녹색 계열 H 상한 (청록 램프 포함, 하늘색 약 100 이상 제외)
    lit_min:        float   = 0.02                                                          # 칸 넓이 대비 켜진 픽셀 비율 하한: 이 이상이어야 "켜진 칸"
    rel_min:        float   = 0.3                                                           # 적/황 칸끼리 비교: 더 밝은 칸 대비 이 비율 미만이면 번진 빛으로 봄
    dark_v:         int     = 90                                                            # 이 밝기 미만 픽셀은 어두운 하우징으로 봄
    dark_min:       float   = 0.25                                                          # 박스 안 어두운 픽셀 비율 하한: 이보다 적으면 간판/표지판으로 봄
    four_aspect:    float   = 3.0                                                           # 가로/세로 비율이 이 값 이상이면 4구 신호등
    arrow_end:      float   = 0.82                                                          # 4구 화살표 덩어리의 오른쪽 끝 상한 (넘으면 화살표가 아니라 녹색 원형등)
    margin_y:       float   = 0.1                                                           # 박스 위/아래 가장자리 제외 비율 (차양, 배경)

# =============================================================
# 판별 결과 1개를 담는 구조체
# =============================================================
@dataclass
class ClassifyResult:
    state:  str     = UNKNOWN                                                               # 최종 상태 (RED, GREEN, ... UNKNOWN)
    lit:    str     = ""                                                                    # 켜진 칸 이름 (예: "R", "RL") -> 화면 확인용
    scores: dict    = field(default_factory=dict)                                           # 칸별 켜진 픽셀 비율 (예: {"R": 0.31, "Y": 0.0, ...}) -> 기준값 조정용
    cells:  int     = 3                                                                     # 3구 / 4구

# =============================================================
# 신호등 박스 -> 신호 상태로 바꿔 주는 클래스
# =============================================================
class SignalClassifier:
    def __init__(self, config=None):
        self.cfg = config if config is not None else ClassifierConfig()                     # 설정을 안 넘기면 기본값 사용

    def layout(self, aspect):
        """박스 비율로 3구/4구를 정해 (칸 이름, 칸 경계 비율) 반환"""
        if aspect >= self.cfg.four_aspect:                                                  # 가로로 더 길면 4구
            return "RYLG", (0.0, 0.25, 0.5, 0.75, 1.0)                                      # 4칸: 적 | 황 | 화살표 | 녹
        return "RYG", (0.0, 1 / 3, 2 / 3, 1.0)                                              # 3칸: 적 | 황 | 녹

    def masks(self, crop):
        """잘라낸 박스 이미지 -> (적/황 계열 마스크, 녹색 계열 마스크, 어두운 픽셀 마스크)"""
        c = self.cfg
        h, s, v = cv2.split(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))                          # BGR -> HSV 변환 후 채널 3개로 분리
        warm = (s >= c.s_min) & (v >= c.v_min) & \
            ((h <= c.warm_max) | (h >= c.warm_wrap))                                         # 밝고 진한 빨강~주황~노랑 = 켜진 적/황 램프

        green = (s >= c.g_s_min) & (v >= c.g_v_min) & \
            ((h >= c.green_lo) & (h <= c.green_hi))                                         # 밝은 녹색~청록 = 켜진 녹색 램프/화살표

        dark = v < c.dark_v                                                                 # 검은 하우징, 꺼진 램프

        return warm, green, dark
    
    def green_split(self, green, split):
        """4구용: 녹색 덩어리마다 위치를 보고 화살표/녹 칸을 정함 -> (화살표 쪽 픽셀 수, 녹 쪽 픽셀 수)
        (칸 경계로 자르면 3구 녹색등이 경계에 걸쳐 화살표로 오인되므로 덩어리 단위로 배정)""" 
        w                   = green.shape[1]                                                # 박스 가로 픽셀 수
        merged              = cv2.dilate(green.astype(np.uint8), np.ones((3, 3), np.uint8)) # 점 LED 사이 틈을 메워 한 덩어리로
        n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)      # 덩어리 번호(1~n-1)와 위치
        ys, xs              = np.nonzero(green)                                             # 원래 녹색 픽셀의 좌표
        cnt                 = np.bincount(labels[ys, xs], minlength=n)                      # 덩어리별 녹색 픽셀 수
        sum_x               = np.bincount(labels[ys, xs], weights=xs, minlength=n)          # 덩어리별 x 좌표 합 (중심 계산용)
        arrow               = 0
        right               = 0
        for k in range(1, n):                                                               # 0번은 배경이므로 1번부터
            if cnt[k] == 0:
                continue
            x_end = stats[k, cv2.CC_STAT_LEFT] + stats[k, cv2.CC_STAT_WIDTH]                # 덩어리 오른쪽 끝
            if sum_x[k] / cnt[k] < split * w and x_end < self.cfg.arrow_end * w:            # 중심이 경계 안쪽 + 오른쪽 끝도 넘지 않으면
                arrow += cnt[k]                                                             # 화살표 칸
            else:                                                                           # (중심은 왼쪽이어도 오른쪽 끝까지 퍼져 있으면 3구 녹색등)
                right += cnt[k]                                                             # 녹 칸

        return arrow, right

    def classify_crop(self, crop, aspect) -> ClassifyResult:
        """잘라낸 박스 이미지 + 박스 가로/세로 비율 -> ClassifyResult"""
        c                   = self.cfg
        if crop.shape[0] < 4 or crop.shape[1] < 8:                                          # 너무 작아서 나눌 수 없으면
            return ClassifyResult()                                                         # UNKNOWN

        m                   = int(crop.shape[0] * c.margin_y)                               # 위/아래에서 잘라낸 줄 수
        crop                = crop[m:crop.shape[0] - m]                                     # 차양/배경이 섞이는 위아래 가장자리 제외
        warm, green, dark   = self.masks(crop)          

        names, bounds       = self.layout(aspect)                                           # 3구/4구 칸 구성
        if dark.mean() < c.dark_min:                                                        # 어두운 하우징이 거의 없으면 (노란 간판 등)
            return ClassifyResult(cells=len(names))                                         # UNKNOWN

        h, w                = crop.shape[:2]                                                # 박스 높이, 가로 픽셀 수
        scores              = {}
        for i, name in enumerate(names):                                                    # 칸마다
            x0, x1          = int(bounds[i] * w), int(bounds[i + 1] * w)                    # 칸의 왼쪽/오른쪽 x (픽셀)
            mask            = green if name in "LG" else warm                               # 화살표/녹 칸은 녹색, 적/황 칸은 적/황 계열만 셈
            scores[name]    = float(mask[:, x0:x1].mean()) if x1 > x0 else 0.0              # 칸 넓이 대비 켜진 픽셀 비율

        if "L" in names:                                                                    # 4구면 화살표/녹 칸은 덩어리 단위로 다시 계산
            arrow, right                = self.green_split(green, bounds[3])                # 경계 = 화살표 칸과 녹 칸 사이 (0.75)
            cell                        = h * w * 0.25                                      # 4구 칸 1개 넓이
            scores["L"], scores["G"]    = float(arrow / cell), float(right / cell)

        lit     = ""
        best    = max(scores["R"], scores["Y"])                                             # 적.황 칸 중 더 밝은 쪽 비율
        for k in names:
            sc = scores[k]
            if sc < c.lit_min:                                                              # 켜진 픽셀이 너무 적으면
                continue                                                                    # 꺼진 칸
            if k in "RY" and sc < best * c.rel_min:                                         # 적/황 칸: 옆 칸에서 번진 빛이면
                continue                                                                    # 꺼진 칸
            lit += k                                                                        # 켜진 칸으로 기록

        state = STATE_TABLE.get(frozenset(lit), UNKNOWN)                                    # 켜진 칸 조합 -> 상태 (없는 조합은 UNKNOWN)
        return ClassifyResult(state=state, lit=lit, scores=scores, cells=len(names))

    def classify(self, frame, d) -> ClassifyResult:
        """프레임 + 박스 1개 -> ClassifyResult (외부에서는 이 함수를 호출)"""
        h, w    = frame.shape[:2]                                                           # 프레임 높이, 너비
        x1, y1  = max(d.x1, 0), max(d.y1, 0)                                                # 박스가 화면 밖으로 나가지 않게 자름
        x2, y2  = min(d.x2, w), min(d.y2, h)
        return self.classify_crop(frame[y1:y2, x1:x2], d.aspect)                            # 박스 영역만 잘라서 판별