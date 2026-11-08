"""
tools/export_engine.py - 학습 가중치(best.pt) -> TensorRT 엔진 변환 (실행 장치에서 1번)
- TensorRT 엔진은 만든 장치(GPU + TensorRT 버전)에서만 동작 -> 엔진을 쓸 장치(젯슨)에서 직접 실행
- 코드는 그대로: detector.py / flow_judges.py 는 .pt 와 .engine 을 같은 방식으로 읽음 (--model 인자만 바꿈)
- 입력 크기는 엔진에 고정됨. 이름에 크기를 넣어 저장: best.pt -> best_<세로>x<가로>.engine (+ .onnx)
    검출 (1920x1080 영상, 16:9): --imgsz 736 1280 (세로 736 x 가로 1280, 여백 최소)
    판별 E03 (정사각형 224): --imgsz 224
- 기본 FP16 (--fp32 로 끌 수 있음). 변환 뒤에는 같은 영상으로 결과가 같은지 반드시 비교
    (예: record_states.py --limit 300 결과를 .pt 결과와 비교)

실행 (저장소 폴더에서, 엔진을 쓸 장치에서):
    python tools/export_engine.py --model runs/detect/det_vehicle_other/weights/best.pt --imgsz 736 1280
    python tools/export_engine.py --model runs/classify/state_cls_none/weights/best.pt --imgsz 224
"""

# =================== 표준 라이브러리 ===========================
import argparse                                                             # 명령행 인자 읽기
import time                                                                 # 변환 시간 측정

from pathlib import Path                                                    # 경로 다루기

# =================== 외부 라이브러리 ===========================
from ultralytics import YOLO                                                # 모델 읽기 + 변환

# =============================================================
# 명령행 인자
# =============================================================
def parse_args():
    p = argparse.ArgumentParser(description="best.pt -> TensorRT 엔진 (이름에 입력 크기 포함)")
    p.add_argument("--model", required=True, help="변환할 가중치 .pt 경로")
    p.add_argument("--imgsz", type=int, nargs="+", required=True, help="입력 크기: 정사각형 1개 (224) 또는 세로 가로 2개 (736 1280)")
    p.add_argument("--fp32", action="store_true", help="FP16 대신 FP32 엔진 (기본 FP16)")
    p.add_argument("--device", default="0", help="GPU 번호 (기본 0)")
    p.add_argument("--force", action="store_true", help="같은 이름 엔진이 있으면 덮어씀")
    return p.parse_args()

# =============================================================
# 변환 실행
# =============================================================
def main():
    args    = parse_args()
    src     = Path(args.model).expanduser()
    if not src.is_file():
        raise SystemExit(f"가중치 파일이 없습니다: {src}")
    if len(args.imgsz) not in (1, 2):
        raise SystemExit("--imgsz 는 숫자 1개 (정사각형) 또는 2개 (세로 가로)")
    h, w    = (args.imgsz * 2)[:2]                                          # 1개면 정사각형 (세로 = 가로)
    stem    = src.with_name(f"{src.stem}_{h}x{w}")                          # 예: .../best_736x1280
    engine  = stem.with_suffix(".engine")                                   # 저장할 엔진 이름
    onnx    = stem.with_suffix(".onnx")                                     # 중간 파일 (ONNX) 이름
    if engine.exists() and not args.force:                                  # 기존 엔진 보호
        raise SystemExit(f"이미 있습니다: {engine} (덮어쓰려면 --force)")
    for tmp in (src.with_suffix(".engine"), src.with_suffix(".onnx")):      # Ultralytics 기본 출력 이름 (best.engine/onnx)
        if tmp.exists():
            raise SystemExit(f"기본 이름 파일이 남아 있습니다: {tmp} -> 이름을 바꾸거나 지운 뒤 다시 실행")

    print(f"export {src} -> {engine.name} (imgsz {h}x{w}, {'FP32' if args.fp32 else 'FP16'}, device {args.device})")
    t0      = time.time()
    out     = YOLO(str(src)).export(format="engine", imgsz=(h, w),          # 세로, 가로 순서
                                    half=not args.fp32, device=args.device)
    out = Path(out)                                                         # Ultralytics 가 만든 엔진 (best.engine)
    out.replace(engine)                                                     # 크기가 드러나는 이름으로
    if src.with_suffix(".onnx").exists():
        src.with_suffix(".onnx").replace(onnx)                              # ONNX 도 같은 규칙으로
    print(f"done {time.time() - t0:.0f} s -> {engine} ({engine.stat().st_size / 1e6:.1f} MB)")

if __name__ == "__main__":
    main()