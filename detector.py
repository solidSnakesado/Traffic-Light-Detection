"""
detector.py - 1. 신호등 검출 모듈
- YOLO26n(COCO 사전학습) 모델로 영상 프레임에서 신호등(COCO 클래스 9번)을 찾는다.
- 다른 모듈(main.py 등)에서는 TrafficLightDetector 클래스만 import 해서 사용한다.
- 검출 결과를 영상으로 확인하려면 detector_view.py 를 실행한다.

설치: pip install ultralytics opencv-python
"""
# =================== 표준 라이브러리 ===========================
from dataclasses import dataclass                                       # 검출 결과 1개를 담을 간단한 데이터 클래스를 만드는 도구

# =================== 외부 라이브러리 ===========================
from ultralytics import YOLO                                            # Ultralytics 라이브러리의 YOLO 모델 로더 (모델 자체가 아니라 모델을 다루는 도구)

# COCO 데이터셋의 "traffic light" 클래스 번호
# (0: person, 1: bicycle, 2: car, 3: motorcycle, 4: airplane, 5: bus, 6: train, 7: truck, 8: boat, 9: traffic light)
COCO_TRAFFIC_LIGHT = 9

# =============================================================
# 검출 결과 1개(신호등 박스 1개)를 담는 구조체
# =============================================================
@dataclass                                                              # __init__ 등을 자동 생성해 주는 데코레이터 -> 필드만 선언하면 됨
class Detection:
    x1:         int                                                     # 박스 왼쪽 위 꼭지점의 x 좌표 (픽셀)
    y1:         int                                                     # 박스 왼쪽 위 꼭지점의 y 좌표 (픽셀, 위쪽이 0)
    x2:         int                                                     # 박스 오른쪽 아래 꼭지점의 x 좌표
    y2:         int                                                     # 박스 오른쪽 아래 꼭지점의 y 좌표
    conf:       float                                                   # 신뢰도 0~1 (탐지해야할 타겟을 확신하는 정도)
    cls_id:     int                                                     # 클래스 번호 (COCO 모델이면 9)
    cls_name:   str                                                     # 클래스 이름 (COCO 모델이면 "traffic light")

    @property                                                           # 메서드를 변수처럼 d.width 형태로 읽게 해 주는 데코레이터
    def width(self) -> int:
        return self.x2 - self.x1                                        # 박스 가로 길이 = 오른쪽 x - 왼쪽 x

    @property                          
    def height(self) -> int:
        return self.y2 - self.y1                                        # 박스 세로 길이 = 아래 y - 위 y

    @property
    def area(self) -> int:
        return self.width * self.height                                 # 박스 넓이 대상 선택에서 가장 큰 신호등 고를때 사용, 
                                                                        # 여러개의 신호등이 화면에 있을 때 가장 가까운 신호등을 탐지하기 위함

    @property
    def aspect(self) -> float:
        return self.width / max(self.height, 1)                         # 가로/세로 비율 -> 1보다 크면 가로형 (0으로 나누기 방지용 max)

    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)     # 박스 중심 좌표 (x, y)

# =============================================================
# YOLO 모델을 감싸서 "프레임 -> 신호등 박스 목록" 으로 바꿔 주는 클래스
# =============================================================
class TrafficLightDetector:
    def __init__(self,
                 model_path="yolo26n.pt",                               # 사용할 가중치 파일 (2단계에서는 파인튜닝한 best.pt로 교체)
                 conf=0.25,                                             # 신뢰도 임계값: 이보다 낮은 박스는 버림
                 imgsz=1280,                                            # 추론 입력 크기: 신호등이 작아서 기본값 640보다 크게
                 target_classes=(COCO_TRAFFIC_LIGHT,),                   # 남길 클래스 번호 목록: COCO는 (9,), 파인튜닝 모델은 None(전체)
                 device=None):                                          # 연산 장치: GPU가 있으면 GPU 없으면 CPU
        
        # 모델로드, 로컬에 파일이 없으면 ultralytics 공식 릴리스에서 자동 다운로드(최초 1회)
        # 이후에는 로컬에서 직접 실행(API 호출 아님)
        self.model          = YOLO(model_path)              
        self.conf           = conf                                      # 신뢰도 임계값 저장
        self.imgsz          = imgsz                                     # 입력 크기 저장

        # 튜플 리스트로 변환해 저장 (ultralytics predict의 classes 인자는 리스트로 받음)
        # None이면 필터링 없이 모델이 아는 모든 클래스 반환
        self.target_classes = list(target_classes) if target_classes is not None else None
        self.device         = device                                    # 연산 장치 저장
        self.names          = self.model.names                          # {클래스 번호: 이름} 딕셔너리 (예: {9: "traffic light"})

    def detect(self, frame) -> list:
        """BGR 프레임 1장을 받아 Detection 리스트를 반환"""

        # 모델 추론 실행
        results = self.model.predict(
            source=frame,                                               # OpenCV로 읽은 numpy 배열(BGR 순서)을 그대로 입력 가능
            conf=self.conf,                                             # 신뢰도 임계값 미만 박스 제거
            imgsz=self.imgsz,                                           # 이 크기로 리사이즈 해서 추론 (결과 좌표는 원본 크기로 자동 환산됨)
            classes=self.target_classes,                                # 지정한 클래스만 결과에 남김 (COCO: 신호등만)
            device=self.device,                                         # 연산 장치
            verbose=False,                                              # 매 프레임마다 콘솔에 찍히는 로그 끄기
        )

        # 이미지 1장을 넣었으므로 결과 리스트에도 1개만 들어 있음 -> [0]으로 꺼냄
        boxes       = results[0].boxes

        # 결과를 PyTorch 텐서(GPU가 있다면 GPU 메모리에 있음) -> CPU로 옮긴 뒤 numpy 배열로 변환
        xyxy        = boxes.xyxy.cpu().numpy()                          # 모양 (N, 4):  박스 N개의 [x1, y1, x2, y2]       
        confs       = boxes.conf.cpu().numpy()                          # 모양 (N,):    박스 N개의 신뢰도     
        clss        = boxes.cls.cpu().numpy()                           # 모양 (N,):    박스 N개의 클래스 번호 (실수형들이 들어 있음)

        detections  = []                                                # 반환할 결과 리스트

        # zip으로 좌표/신뢰도/클래스를 박스 단위로 묶어서 하나씩 꺼냄
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            cls_id = int(k)                                             # 클래스 번호를 실수 -> 정수로 변환 (9.0 -> 9)
            detections.append(Detection(
                x1=int(x1), y1=int(y1),                                 # 좌표를 픽셀 정수로 변환 (그리기 함수는 정수 좌표를 받음)
                x2=int(x2), y2=int(y2),
                conf=float(c),                                          # numpy float -> 파이썬 float
                cls_id=cls_id,
                cls_name=self.names[cls_id],                            # 번호로 이름 조회
            ))

        return detections                                               # 신호등이 없으면 빈 리스트 [] 반환