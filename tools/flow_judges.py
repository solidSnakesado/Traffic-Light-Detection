"""
tools/flow_judges.py - C4 전체 흐름 평가용 판별기, 정답 대상, 결과 분류 (eval_flow.py 가 사용)
- 판별기(judge): 박스 1개 -> 상태 문자열. selector.select(judge=...) 에 그대로 넘김
    HSV : classifier.py 의 SignalClassifier (판별 불가면 UNKNOWN)
    모델 : train.py 로 학습한 분류 모델 (E02 6클래스 / E03 7클래스)
          모델의 NONE("신호 아님")은 UNKNOWN 으로 바꿔서 넘김
          -> selector 의 NONE("대상 없음")과 섞이지 않고, "켜진 후보만 고르기" 규칙에서 빠짐
- 정답 대상: 정답 상태 박스(0~5)에 selector 1단계 필터 + pick 을 그대로 적용해 고른 1개 (없으면 NONE)
- 결과 분류: 맞음 / 검출 실패 / 다른 박스 선택 / 상태 오판 (정답 대상이 있는 이미지)
            대상 없음 맞음 / 오경보 / 판별 불가 대상 (정답 대상이 없는 이미지)
"""

# =================== 표준 라이브러리 ===========================
from pathlib        import Path                                             # 경로 다루기
from types          import SimpleNamespace                                  # 속성(b.x1 등)으로 읽는 간단한 박스 객체

# =================== 외부 라이브러리 ===========================
from ultralytics    import YOLO                                             # 학습한 분류 모델 불러오기

# =================== 프로젝트 모듈   ===========================
from eval_hsv       import STATES                                           # 같은 tools 폴더: 상태 이름 6개 (저장소 폴더를 모듈 경로에 추가해 줌)
from make_crops     import NONE as CLS_NONE, square_crop                    # 같은 tools 폴더: 모델의 "신호 아님" 이름, 학습 때와 같은 자르기
from classifier     import SignalClassifier, UNKNOWN                        # 3. HSV 상태 판별기, 판별 불가 상태 이름
from selector       import NONE                                             # 2. selector 의 "대상 없음" 상태 이름

IOU_MATCH = 0.5                                                             # 이 값 이상 겹치면 같은 박스로 봄

# =============================================================
# 두 박스의 IoU (겹친 넓이 / 합친 넓이)
# =============================================================
def iou(a, b):
    iw      = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))                     # 겹친 부분 너비 (안 겹치면 0)
    ih      = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))                     # 겹친 부분 높이
    inter   = iw * ih                                                       # 겹친 넓이
    union   = a.area + b.area - inter                                       # 합친 넓이
    return inter / union if union > 0 else 0.0      

# =============================================================
# HSV 판별기 (베이스라인)
# =============================================================
class HsvJudge:
    def __init__(self):
        self.name   = "HSV"                                                 # 결과 표에 쓸 이름
        self.clf    = SignalClassifier()                                    # 기준값은 기본값 그대로 (E01 과 같음)

    def bind(self, frame):
        """프레임 1장 -> selector 에 넘길 judge 함수"""
        return lambda d: self.clf.classify(frame, d).state                  # 판별 불가면 UNKNOWN

# =============================================================
# 학습한 분류 모델 판별기 (E02, E03)
# =============================================================
class ModelJudge:
    def __init__(self, path, imgsz=224, device=None):
        path        = Path(path).expanduser().resolve()
        self.model  = YOLO(str(path))                                       # 학습한 분류 모델
        self.imgsz  = imgsz                                                 # 모델 입력 크기 (학습 때와 같게)
        self.device = device                                                # 연산 장치 (None 이면 자동)
        run         = path.parent.parent.name                               # runs/classify/<이름>/weights/best.pt -> <이름> 
        self.name   = run if path.parent.name == "weights" else path.stem   # 결과 표에 쓸 이름 (예: state_cls_none)

    def bind(self, frame):
        """프레임 1장 -> selector 에 넘길 judge 함수"""
        fh, fw = frame.shape[:2]                                            # 프레임 높이, 너비

        def judge(d):
            x1, y1 = max(d.x1, 0), max(d.y1, 0)                             # 검출 박스를 화면 안으로 자름
            x2, y2 = min(d.x2, fw), min(d.y2, fh)   
            if x2 - x1 < 1 or y2 - y1 < 1:                                  # 자를 수 없는 박스면
                return UNKNOWN                                              # 판별 불가
            b = SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2)                 # square_crop 은 좌표 4개만 사용
            r = self.model.predict(square_crop(frame, b), imgsz=self.imgsz,
                                   device=self.device, verbose=False)[0]    # 박스 1개 예측
            s = self.model.names[r.probs.top1]                              # 가장 확률 높은 클래스 이름
            return UNKNOWN if s == CLS_NONE else s                          # 모델 NONE(신호 아님) -> UNKNOWN (selector NONE 과 구분)
        return judge

# =============================================================
# 정답 상태 박스 -> 정답 대상 (selector 1단계 + pick 그대로), 없으면 (None, NONE, "")
# =============================================================
def gt_target(sel, gts, fh, fw):
    cands, _    = sel.filter([b for _, _, b in gts], fh, fw)        # 1단계: 후보 거르기 (예측과 같은 기준)
    target      = sel.pick(cands, fw)                               # 2단계: 크기 + 화면 중앙 기준 1개 (정답 박스는 모두 켜진 신호)
    for cls, sig, b in gts:
        if b is target:                                             # 고른 박스의 정답 상태 찾기
            return b, STATES[cls], str(sig)
    return None, NONE, ""                                           # 정답 대상 없음

# =============================================================
# 이미지 1장의 결과 분류
# =============================================================
def outcome(gt_box, gt_state, dets, target, pred):
    if gt_box is None:                                              # 정답 대상이 없는 이미지
        if pred == NONE:
            return "ok_none"                                        # 대상 없음을 맞힘
        return "unknown" if pred == UNKNOWN else "false_alarm"      # 판별 불가 대상 / 없는 신호를 만듦 (오경보)
    if pred == gt_state:
        return "ok"                                                 # 상태를 맞힘 (박스 일치는 box_iou 로 따로 확인)
    if not any(iou(d, gt_box) >= IOU_MATCH for d in dets):
        return "det_miss"                                           # 검출 실패: 정답 대상과 겹치는 검출 박스가 없음
    if target is None or iou(target, gt_box) < IOU_MATCH:
        return "wrong_box"                                          # 다른 박스 선택 (또는 대상을 못 고름)
    return "wrong_state"                                            # 맞는 박스를 골랐지만 상태 오판