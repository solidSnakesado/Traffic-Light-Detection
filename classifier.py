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
from dataclasses import dataclass, field                                    # 설정값/결과를 담을 데이터 클래스 도구

# =================== 외부 라이브러리 ===========================
import cv2                                                                  # OpenCV: BGR -> HSV 색 공간 변환, 덩어리(연결 요소) 찾기
import numpy as np                                                          # 픽셀 좌표/개수 계산

# =================== 출력 상태 이름 ===========================
RED, YELLOW, GREEN, LEFT    = "RED", "YELLOW", "GREEN", "LEFT"              # 적색 / 황색 / 녹색 / 녹색 좌회전 화살표
RED_LEFT, GREEN_LEFT        = "RED_LEFT", "GREEN_LEFT"                      # 적색 + 화살표 / 녹색 + 화살표 (4구 동시 점등)
UNKNOWN, NONE               = "UNKNOWN", "NONE"                             # 판별 불가 / 대상 신호등 없음

# 켜진 칸 조합 -> 상태 (칸 이름 R: 적, Y: 황, L: 좌회전 화살표, G: 녹)
STATE_TABLE = {
    frozenset("R"):     RED,
    frozenset("Y"):     YELLOW,
    frozenset("G"):     GREEN,
    frozenset("L"):     LEFT,
    frozenset("RL"):    RED_LEFT,
    frozenset("GL"):    GREEN_LEFT,
    frozenset("YL"):    YELLOW,                                             # 황색과 화살표가 함께 보이는 전환 프레임 -> 황색(정지 준비) 우선
}                                                                           # 표에 없는 조합(적 + 녹, 적 + 황 등)은 UNKNOWN

# =============================================================
# 판별 기준값 모음 (OpenCV HSV: H 0~179, S 0~255, V 0~255)
# =============================================================
@dataclass
class ClassifierConfig:
    pass

# =============================================================
# 판별 결과 1개를 담는 구조첸
# =============================================================
@dataclass
class ClassifyResult:
    pass

# =============================================================
# 신호등 박스 -> 신호 상태로 바꿔 주는 클래스
# =============================================================
class SignalClassifier:
    def __init__(self, config=None):
        pass

    def layout(self, aspect):
        """박스 비율로 3구/4구를 정해 (칸 이름, 칸 경계 비율) 반환"""
        pass

    def masks(self, crop):
        """잘라낸 박스 이미지 -> (적/황 계열 마스크, 녹색 계열 마스크, 어두운 픽셀 마스크)"""
        pass
    
    def green_split(self, green, split):
        """4구용: 녹색 덩어리마다 위치를 보고 화살표/녹 칸을 정함 -> (화살표 쪽 픽셀 수, 녹 쪽 픽셀 수)
        (칸 경계로 자르면 3구 녹색등이 경계에 걸쳐 화살표로 오인되므로 덩어리 단위로 배정)""" 
        pass

    def classify_crop(self, crop, aspect) -> ClassifyResult:
        """잘라낸 박스 이미지 + 박스 가로/세로 비율 -> ClassifyResult"""
        pass

    def classify(self, frame, d) -> ClassifyResult:
        """프레임 + 박스 1개 -> ClassifyResult (외부에서는 이 함수를 호출)"""
        pass