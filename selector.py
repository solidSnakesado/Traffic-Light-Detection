"""
selector.py - 2. 대상 신호등 선택 모듈
- detector.py 가 찾은 신호등 박스 목록에서 차량용 가로형 신호등만 남기고 (1단계: 후보 거르기)
  남은 후보 중 지금 따라야 할 신호등 1개를 고름                        (2단계: 1개 고르기)
- 기준 값은 SelectorConfig 에 모아 두고, detector_view.py 로 영상을 보며 조정
- 기준은 한국 도로의 가로형 차량 신호등에 맞춘 값
"""

# =================== 표준 라이브러리 ===========================
from dataclasses import dataclass, field                            # 설정값/결과를 담을 데이터 클래스 도구 (field: 리스트 기본값 용)
from typing import Optional                                         # "값이 없을 수도 있음(None)"을 표시하는 타입 힌트

# =============================================================
# 선택 기준값 모음 (1920x1080 영상 기준 초기값)
# =============================================================
@dataclass
class SelectorConfig:
    min_aspect:     float   = 1.5                                       # 가로/세로 비율 하한: 이보다 작으면 세로형(보행자신호, 세로 표지판)으로 보고 버림
    max_center_y:   float   = 0.55                                      # 박스 중심 y의 상한 (화면 높이 대비 비율): 위쪽 55% 안에 있어야 통과
    min_area:       int     = 300                                       # 최소 넓이(px^2): 이보다 작으면 너무 멀거나 노이즈로 보고 버림
    area_tie:       float   = 0.8                                       # 가장 큰 박스 넓이의 80% 이상이면 "크기가 비슷하다"고 보고 중앙 근접도로 비교

# =============================================================
# 선택 결과 1프레임분을 담는 구조체
# =============================================================
@dataclass
class SelectionResult:
    target:     Optional[object]    = None                              # 최종 선택된 신호등 Detection 1개 (없으면 None = "신호 없음")
    candidates: list                = field(default_factory=list)       # 1단계를 통과한 후보 Detection 목록
    rejected:   list                = field(default_factory=list)       # 1단계에서 버린 (Detection, 탈락이유) 목록 -> 화면 확인용

# =============================================================
# 검출 박스 목록 -> 대상 신호등 1개로 바꿔 주는 클래스
# =============================================================
class TargetSelector:
    def __init__(self, config=None):
        self.cfg = config if config is not None else SelectorConfig()   # 설정을 안 넘기면 기본값 사용

    def reject_reason(self, d, frame_h) -> Optional[str]:
        """박스 1개를 검사해 탈락 이유를 반환 (통과하면 None)"""
        if d.aspect < self.cfg.min_aspect:                              # 세로형이거나 정사각형에 가까우면
            return "aspect"                                             # -> 가로형 차량 신호등이 아님
        if d.center[1] > frame_h * self.cfg.max_center_y:               # 박스 중심이 화면 아래쪽에 있으면 (y는 아래로 갈수록 커짐)
            return "low"                                                # -> 낮게 달린 보행자 신호, 간판 등
        if d.area < self.cfg.min_area:                                  # 박스가 너무 작으면
            return "small"                                              # -> 너무 먼 신호등 이거나 노이즈

        return None                                                     # 세 조건을 모두 통과 -> 후보

    def filter(self, detections, frame_h):
        """1단계: 후보 거르기 -> (후보 목록, 탈락 목록) 반환"""
        candidates, rejected = [], []                                   # 통과/탈락 나눠 담을 빈 리스트 2개
        for d in detections:                                            # 검출된 박스마다 반복
            reason = self.reject_reason(d, frame_h)                     # 탈락 이유 검사
            if reason is None:                                          # 통과했으면
                candidates.append(d)                                    # 후보에 추가
            else:                                                       # 탈락했으면
                rejected.append((d, reason))                            # 박스와 이유를 함께 저장 (화면에 이유 표시용)

        return candidates, rejected

    def pick(self, candidates, frame_w):
        """2단계: 후보 중 1개 고르기 -> Detection 1개 또는 None 반환"""
        if not candidates:                                              # 후보가 하나도 없으면
            return None                                                 # "신호 없음"

        max_area = max(d.area for d in candidates)                      # 후보 중 가장 큰 넓이 (가장 가까운 신호등)

        # 가장 큰 박스와 크기가 비슷한 박스들만 남김 (같은 교차로에 신호등 여러 개)
        near_max = [d for d in candidates if d.area >= max_area * self.cfg.area_tie]

        cx = frame_w / 2                                                # 화면 가로 중앙 x 좌표
        return min(near_max, key=lambda d: abs(d.center[0] - cx))       # 그중 화면 중앙에 가장 가까운 박스 선택 (내 차로 앞 신호)

    def select(self, detections, frame_shape) -> SelectionResult:
        """검출 결과 + 프레임 크기를 받아 선택 결과를 반환 (외부에서는 이 함수만 호출)"""
        frame_h, frame_w = frame_shape[:2]                              # frame.shape = (높이, 너비, 채널) -> 앞의 2개만 사용
        candidates, rejected = self.filter(detections, frame_h)         # 1단계: 후보 거르기
        target = self.pick(candidates, frame_w)                         # 2단계: 1개 고르기
        return SelectionResult(target=target, candidates=candidates, rejected=rejected)