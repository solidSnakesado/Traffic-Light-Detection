"""
tools/train_detect.py - 신호등 검출 모델 학습 (2클래스: 0 vehicle / 1 other, yolo26n 전이학습)
- 입력: remap_classes.py 로 만든 검출 데이터셋의 data.yaml (train / val 사용, test 는 eval_detect.py 로 따로 평가)
- 시작 가중치: COCO 사전학습 yolo26n.pt (마지막 층은 데이터셋 클래스 수(2)에 맞게 자동 교체)
- 데이터 늘리기 (augmentation)
    좌우 뒤집기 끔: 4구 신호등의 램프 배열(적색 왼쪽, 녹색 오른쪽)과 좌회전 방향이 바뀜
    그 외(모자이크, 확대/축소, 밝기/채도/색조 소폭 변화)는 기본값 부터 시작
- 결과: <저장소>/runs/detect/<name>/weights/best.pt (val mAP 가 가장 높은 가중치), results.csv, 그래프

실행 (저장소 폴더에서 GPU가 있는 PC):
  python tools/train_detect.py --fraction 0.1 --epochs 1 --name det_try     # 짧은 시험 학습
  python tools/train_detect.py --epochs 50                                  # 본 학습
"""

# =================== 표준 라이브러리 ===========================
import argparse                                                                         # 명령행 인자 (--data, --epochs 등) 읽기
from pathlib import Path                                                                # 경로 다루기

# =================== 외부 라이브러리 ===========================
from ultralytics import YOLO                                                            # 검출 모델 불러오기 + 학습

RUNS = Path(__file__).resolve().parent.parent / "runs" / "detect"                       # 결과 저장 위치 (저장소 폴더/runs/detect, 어느 폴더에서 실행해도 동일)

# =============================================================
# 명령행 인자
# =============================================================
def parse_args():
    p =  argparse.ArgumentParser(description="신호등 검출 모델 학습 (yolo26n, vehicle / other)")
    p.add_argument("--data", default="~/datasets/traffic_light_vehicle_other/data.yaml", help="remap_classes.py 로 만든 data.yaml")
    p.add_argument("--model", default="yolo26n.pt", help="시작 가중치 (COCO 사전학습, 없으면 자동 다운로드)")
    p.add_argument("--epochs", type=int, default=50, help="전체 학습 데이터를 몇 번 반복할지")
    p.add_argument("--patience", type=int, default=15, help="val 성능이 이 epoch 수 동안 오르지 않으면 조기 종료")
    p.add_argument("--imgsz", type=int, default=1280, help="모델 입력 크기 (긴 변 픽셀, 작은 신호등 때문에 1280)")
    p.add_argument("--batch", type=float, default=8, help="한 번에 학습할 이미지 수 (-1 = GPU 메모리 60%% 기준 자동, 0.7 = 70%% 기준 자동)")
    p.add_argument("--fraction", type=float, default=1.0, help="train 데이터 중 사용할 비율 (짧은 시험 학습용, 예: 0.1)")
    p.add_argument("--device", default=None, help="0 = 첫 번째 GPU, cpu = CPU (없으면 자동 선택)")
    p.add_argument("--workers", type=int, default=4, help="이미지를 읽는 프로세스 수")
    p.add_argument("--cache", default=None, choices=["ram", "disk"],help="이미지 캐시 (기본 끔: 메모리/저장 공간 부족)")
    p.add_argument("--name", default="det_vehicle_other", help="결과 폴더 이름 (같은 이름이 있으면 뒤에 숫자가 붙음)")
    return p.parse_args()

# =============================================================
# 학습 실행
# =============================================================
def main():
    args    = parse_args()
    data    = Path(args.data).expanduser().resolve()                                    # ~ 를 홈 폴더로 바꾼 절대 경로
    if not data.is_file():
        raise SystemExit(f"data.yaml 없음: {data}")
    batch   = int(args.batch) if args.batch >= 1 or args.batch == -1 else args.batch    # 8.0 -> 8, -1.0 -> -1, 0.7 은 그대로
    model   = YOLO(args.model)                                                          # COCO 사전 학습 검출 모델

    model.train(
        data            = str(data),                                                    # 데이터셋 설정 파일 (경로, 클래스 이름)
        epochs          = args.epochs,
        patience        = args.patience,
        imgsz           = args.imgsz,
        batch           = batch,
        fraction        = args.fraction,
        device          = args.device,
        workers         = args.workers,
        cache           = args.cache or False,                                          # None -> False (캐시 끔)
        project         = str(RUNS),                                                    # 결과 상위 폴더
        name            = args.name,                                                    # 결과 폴더 이름
        seed            = 0,                                                            # 난수 고정 (같은 설정이면 같은 결과에 가깝게)
        deterministic   = True,                                                         # 재현성 우선
        fliplr          = 0.0,                                                          # 좌우 뒤집기 끔 (신호등 램프 배열, 화살표 방향 보존)
        plots           = True,                                                         # 학습 곡선, PR 곡선, 혼동 행렬 그림 저장               
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"                         # val mAP 가 높았던 가중치
    print(f"best -> {best}")

if __name__ == "__main__":
    main()