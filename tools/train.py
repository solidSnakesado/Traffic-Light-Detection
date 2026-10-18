"""
tools/train.py - 신호 상태 분류 모델 학습 (B 구조: 검출 박스를 잘라 yolo26n-cls 로 상태 6개 분류)
- 입력: make_crops.py 로 만든 분류 데이터셋 (<data>/train, <data>/val, 폴더 이름 = 상태 이름)
- test 는 학습/모델 선택에 쓰지 않음 (eval_cls.py 로 HSV 베이스라인과 같은 조건에서 최종 비교)
- 신호등 상태가 바뀌는 데이터 늘리기(augmentation)는 끔
    좌우 뒤집기: 적색(왼쪽)/녹색(오른쪽) 위치와 좌회전 화살표 방향이 바뀜
    색조(H) 변경: 빨강/노랑/초록 자체가 바뀜
    지우기/자르기: 켜진 램프 칸이 지워지거나 잘려 나갈 수 있음
    RandAugment: 색 반전(solarize) 등으로 램프 색이 바뀔 수 있음
  밝기(V)/채도(S) 변화는 유지 (야간, 역광, 흐린 램프 대비)
- 결과: <저장소>/runs/classify/<name>/weights/best.pt (val 정확도가 가장 높은 가중치), results.csv, 그래프

실행 (저장소 폴더에서, GPU가 있는 PC):
  python tools/train.py --epochs 30
"""

# =================== 표준 라이브러리 ===========================
import argparse                                                         # 명령행 인자(--data, --epochs 등) 읽기

from pathlib import Path                                                # 경로 다루기

# =================== 외부 라이브러리 ===========================
from ultralytics import YOLO                                            # 분류 모델 불러오기 + 학습

RUNS = Path(__file__).resolve().parent.parent / "runs" / "classify"     # 결과 저장 위치 (저장소 폴더/runs/classify, 어느 폴더에서 실행해도 동일)

# =============================================================
# 명령행 인자
# =============================================================
def parse_args():
    p = argparse.ArgumentParser(description="신호 상태 분류 모델 학습 (yolo26n-cls)")
    p.add_argument("--data", default="~/datasets/traffic_light_cls", help="make_crops.py 로 만든 분류 데이터셋 폴더")
    p.add_argument("--model", default="yolo26n-cls.pt", help="시작 가중치 (ImageNet 사전학습, 없으면 자동 다운로드)")
    p.add_argument("--epochs", type=int, default=30, help="전체 학습 데이터를 몇 번 반복할지")
    p.add_argument("--imgsz", type=int, default=224, help="모델 입력 크기 (정사각형 한 변 픽셀)")
    p.add_argument("--batch", type=int, default=64, help="한 번에 학습할 이미지 수 (GPU 메모리 부족 시 줄임)")
    p.add_argument("--fraction", type=float, default=1.0, help="train 데이터 중 사용할 비율 (짧은 시험 학습용, 예: 0.1)")
    p.add_argument("--device", default=None, help="0 = 첫 번째 GPU, cpu = CPU (없으면 자동 선택)")
    p.add_argument("--workers", type=int, default=4, help="이미지를 읽는 프로세스 수")
    p.add_argument("--name", default="state_cls", help="결과 폴더 이름 (같은 이름이 있으면 뒤에 숫자가 붙음)")
    return p.parse_args()

# =============================================================
# 학습 실행
# =============================================================
def main():
    args    = parse_args()
    data    = Path(args.data).expanduser().resolve()                    # ~ 를 홈 폴더로 바꾼 절대 경로
    model   = YOLO(args.model)                                          # 분류 모델 (마지막 층은 데이터셋 클래스에 맞게 자동 교체)

    model.train(
        data            = str(data),                                    # 폴더 경로를 주면 분류 데이터셋으로 인식
        epochs          = args.epochs,
        imgsz           = args.imgsz,
        batch           = args.batch,
        fraction        = args.fraction,
        device          = args.device,
        workers         = args.workers,
        project         = str(RUNS),                                    # 결과 상위 폴더
        name            = args.name,                                    # 결과 폴더 이름
        seed            = 0,                                            # 난수 고정 (같은 설정이면 같은 결과에 가깝게)
        deterministic   = True,                                         # 재현성 우선
        fliplr          = 0.0,                                          # 좌우 뒤집기 끔 (램프 위치, 화살표 방향 보존)
        hsv_h           = 0.0,                                          # 색조 변경 끔 (램프 색 보존)
        erasing         = 0.0,                                          # 무작위 지우기 끔 (켜진 램프 보존)
        scale           = 0.0,                                          # 무작위 확대 자르기 끔 (양 끝 램프 보존)
        auto_augment    = None,                                         # RandAugment 끔 (색 반전 등 방지)
        plots           = True,                                         # 학습 곡선, 혼동 행렬 그림 저장
    )    

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"         # val 정확도가 가장 높았던 가중치
    print(f"best -> {best}")

if __name__ == "__main__":
    main()