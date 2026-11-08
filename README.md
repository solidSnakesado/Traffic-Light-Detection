# Traffic-Light-Detection

주행 영상에서 신호등을 검출하고 신호 상태를 판별해, **ROS2 (Jazzy)** 로 주행 로봇 **Pinky Pro** 를 제어하는 프로젝트

- 처음 계획은 TCP 통신 + 아두이노였으나 ROS2 + Pinky Pro 로 변경 (2026-11-08)
- `주행 신호 판단 TCP 통신 플로우 차트.png` 는 변경 전 계획 (참고용)

## 처리하는 기능

| 단계 | 파일 | 하는 일 | 방법 |
|---|---|---|---|
| ① 검출 | `detector.py` | 프레임에서 차량 신호등 박스 찾기 | YOLO26n, COCO 사전학습 → AI Hub 로 **파인튜닝** (vehicle / other 2클래스, `--det-classes 0`) |
| ③ 상태 판별 | `tools/flow_judges.py` | 박스마다 신호 상태 판별 | YOLO26n-cls, ImageNet 사전학습 → 잘라낸 박스로 **파인튜닝** (RED, YELLOW, GREEN, LEFT, RED_LEFT, GREEN_LEFT, NONE) |
| ② 대상 선택 | `selector.py` | 여러 신호등 중 따라야 할 1개 고르기 | **규칙** (가로세로비 ≥ 1.5, 중심 높이 ≤ 0.55, 면적 ≥ 300, 가장자리 제외) |
| ④ 안정화 | `smoother.py` | 프레임마다 흔들리는 상태를 확정 상태로 | **규칙** (시간 창 비율 투표, 정지 쪽 0.2초 / 출발 쪽 0.5초 확정, 1.5초 안 보이면 NONE) |
| ⑤ 제어 | ROS2 노드 (예정) | 확정 상태 → 속도 명령 | – |
| ⑥ 로봇 | Pinky Pro | `/cmd_vel` 로 주행 | 원본 `pinky_pro` 그대로 사용, 필요한 기능만 이 저장소에 추가 (예정) |

## 개발 과정 (문제 → 다음 방법)

| 순서 | 방법 | 결과 / 문제 | 다음 |
|---|---|---|---|
| 1 | COCO 사전학습 YOLO (학습 안 함) + HSV 색 규칙 `classifier.py` | 시연 영상 대상 표시율 31.7%, 중거리 신호등 미검출. HSV 상태 정확도 60.4% (판별 거부 34.7%, 4구 화살표 구분 못 함) | 상태 판별 학습 |
| 2 | 상태 분류 학습 E02 (6클래스) | 정답 박스 기준 97.1%. 하지만 신호등이 아닌 박스도 6상태 중 하나로 판단 | NONE 클래스 + 박스 흔들기 |
| 3 | E03 (7클래스, +NONE) | 96.2%, 신호 아닌 박스 98.4% 거름. 전체 흐름에서는 COCO 검출이 대상을 놓침 (정확도 30.8%) | 검출 학습 |
| 4 | 검출 파인튜닝 det_vehicle_other | 차량 신호등 Recall 16.9% → 89.4%, 전체 흐름 정확도 97.7%. 영상에서는 프레임마다 상태가 흔들림 | 안정화 |
| 5 | smoother (규칙) | 상태 변화 129 → 25회, 0.2초 미만 깜박임 94 → 0 | 휴대용 실행 장치 |
| 6 | 젯슨 + TensorRT 엔진 | PyTorch 69.5 → 27.8 ms/프레임, 확정 상태는 DESKTOP 과 15,797 / 15,797 같음 | ROS2 + Pinky Pro |

## 주요 결과

| 항목 | 조건 | 값 |
|---|---|---|
| 상태 판별 (E03) | AI Hub test 정답 박스 4,536개 | 96.2% (HSV 60.4%) |
| 전체 흐름 (검출 + 판별 + 선택) | AI Hub test 1,500장 | 97.7% (COCO + HSV 23.6%) |
| 안정화 | `drive.mp4` 263초 | 변화 129 → 25회, 평균 확정 지연 0.31초 |
| 처리 속도 DESKTOP | RTX 5070 Ti, PyTorch | 11.9 ms/프레임 (84 fps) |
| 처리 속도 젯슨 | Orin Nano Super, TensorRT FP16 736x1280 | 27.8 ms/프레임 (약 36 fps) |

- 남은 문제: 갈림길·오르막에서 정면 신호등이 `selector` 높이 기준(0.55)에 걸려 탈락 (101초, 165초), "황색 + 화살표" 는 판별 클래스 밖

## 장치

| 장치 | 역할 | 환경 문서 |
|---|---|---|
| DESKTOP (RTX 5070 Ti Laptop, WSL Ubuntu 24.04) | 학습, 평가, 추론, 로봇 연동 | `setup/desktop_wsl.md` |
| 젯슨 오린 나노 슈퍼 (JetPack 7.2.1) | 추론, 로봇 연동 (휴대용, TensorRT 엔진) | `setup/jetson_jp72.md` |
| Pinky Pro (Raspberry Pi 5) | 주행 로봇 | 예정 |

- ROS2 Jazzy, `ROS_DOMAIN_ID=20` (Pinky Pro 와 같은 번호)
- 가중치 (`*.pt`, `*.engine`), 영상, 데이터셋은 git 에 올리지 않음 (scp 로 옮김)

## 폴더

```
detector.py  selector.py  smoother.py  classifier.py  detector_view.py   # 처리 모듈, 확인 뷰어
tools/       # 데이터셋 변환, 학습, 평가, 기록 (record_states.py), 재생 평가 (replay_smoother.py), 엔진 변환 (export_engine.py)
ros2/        # ROS2 패키지 (traffic_light_msgs, traffic_light_nodes)
setup/       # 장치별 환경 문서, constraints.txt
```

## ROS2 빌드와 실행

```bash
traffic                                                  # ROS2 + 가상환경 + Domain ID (장치별 별칭, setup 문서)
mkdir -p ~/traffic_ws && ln -sfn <저장소>/ros2 ~/traffic_ws/src
cd ~/traffic_ws && python -m colcon build --symlink-install
source install/setup.bash
ros2 run traffic_light_nodes env_check                   # 환경 확인
```

- `python -m colcon` 으로 빌드해야 노드가 가상환경 Python 으로 실행됨 (`colcon` 만 쓰면 시스템 Python → ultralytics 없음)

## 진행 상태

| 단계 | 내용 | 상태 |
|---|---|---|
| R0 | 장치 환경 (DESKTOP, 젯슨) | 완료 |
| R1 | 메시지 / 노드 패키지, `env_check` | 완료 (두 장치) |
| R2 | 영상 + 인식 노드 | 예정 |
| R3 | 안정화 노드 | 예정 |
| R4 | 제어 + 방향 의도 노드 | 예정 |
| R5 | Pinky Pro 연동 (명령 끊김 감시 포함) | 예정 |
| R6 | launch, 문서 | 예정 |

## 실험 상세 (`docs/`)

| 문서 | 내용 |
|---|---|
| [상태 판별](docs/state_classification.md) | E01 HSV, E02, E03 (조건, 혼동 행렬, 오류 유형) |
| [검출 / 전체 흐름](docs/detection_and_flow.md) | D01 COCO, D02 파인튜닝, C4 전체 흐름, 오경보 분석, C6 시연 영상 |
| [안정화](docs/smoothing.md) | smoother 방식, 기준값, drive.mp4 구간별 확인 |
| [실행 장치](docs/jetson_runtime.md) | 젯슨 TensorRT 변환, 속도, DESKTOP 결과 비교 |