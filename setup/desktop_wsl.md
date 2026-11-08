# DESKTOP (Windows + WSL) 학습·평가 환경

- 역할: **학습, 평가** (`tools/train*.py`, `tools/eval_*.py`, `tools/replay_smoother.py`). 실행 장치는 젯슨 (`setup/jetson_jp72.md`)
- 확인한 값 (progress 문서 기준, 2026-10-24 이후 메인 PC)

| 항목 | 값 |
|---|---|
| PC | DESKTOP-QB1L04J, RTX 5070 Ti Laptop 12GB |
| Python | 3.12.3 |
| 가상환경 | `~/traffic_venv` (`source ~/traffic_venv/bin/activate`) |
| torch | 2.14.1+cu130 |
| ultralytics | 8.4.175 (젯슨과 같은 버전 유지) |
| 저장소 | `~/project/traffic_light` |
| 데이터 | AI Hub Validation 원본 + `~/datasets/traffic_light*` (젯슨으로 옮기지 않음) |

- 모델은 PyTorch `best.pt` 그대로 사용 (약 100 fps, TensorRT 변환 필요 없음)
- 새로 학습해 `best.pt` 가 바뀌면 젯슨으로 복사 → 젯슨에서 `tools/export_engine.py` 로 엔진 다시 생성
- 새 패키지를 설치할 때도 `-c setup/constraints.txt` 사용 권장 (장치 간 numpy 버전 차이 방지)
