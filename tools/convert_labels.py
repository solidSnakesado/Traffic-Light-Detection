"""
tools/convert_labels.py - AI Hub 신호등 라벨(JSON) -> YOLO 학습 데이터셋 변환
- 입력: AI Hub Validation 폴더 (01.원천데이터 = 이미지, 02.라벨링데이터 = JSON)
- 출력: dataset/images/{train,val}/*.jpg, dataset/labels/{train,val}/*.txt, dataset/data.yaml
- 라벨 형식 2가지를 모두 읽음
    AI Hub 형식: objects[].class_name / data     [[x1,y1],[x2,y2]] / attribute.signal / image_size
    LabelMe 형식: shapes[].label     / points   [[x1,y1],[x2,y2]] / flags.signal     / imageWidth, imageHeight
- train/val 은 Clip 폴더(장면) 단위로 나눔 -> 거의 같은 연속 프레임이 양쪽에 섞여 성능이 부풀려지는 것 방지
- 이미지는 기본으로 심볼릭 링크 (원본을 가리키는 바로가기, 디스크 절약), --copy를 주면 실제 복사

실행: python tools/convert_labels.py --src <Validation 폴더 경로> --out dataset
"""

# =================== 표준 라이브러리 ===========================
import argparse                                                         # 명령행 인자(--src, --out 등) 읽기
import glob                                                             # 패턴으로 파일 목록 찾기 (**: 하위 폴더 전체)
import json                                                             # 라벨 JSON 읽기
import os                                                               # 심볼릭 링크 만들기, 기존 파일 지우기
import random                                                           # Clip 폴더를 섞어 train/val 로 나누기
import shutil                                                           # --copy 일 때 이미지 복사

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
    pass

# =============================================================
# 라벨 JSON 1개 읽기 (두 형식 모두) -> (박스 목록, 이미지 너비, 높이)
# =============================================================
def read_label(path):
    with open(path, encoding="utf-8") as f:
        pass

# =============================================================
# 픽셀 박스 -> YOLO 한 줄 "클래스 x중심 y중심 너비 높이" (0~1 비율)
# =============================================================
def to_yolo(cls, x1, y1, x2, y2, w, h):
    pass

# =============================================================
# 이미지를 데이터셋 폴더에 넣기 (링크 또는 복사)
# =============================================================
def place_image(src, dst, copy):
    pass

# =============================================================
# 명령행 인자
# =============================================================
def place_args():
    pass

# =============================================================
# 변환 실행
# =============================================================
def main():
    pass