# 젯슨 오린 나노 슈퍼 (JetPack 7.2.1) 실행 환경 구성

- 역할: **실행 장치** (ROS2 노드, 인식 = TensorRT 엔진). 학습·평가는 DESKTOP
- 확인한 날짜: 2026-11-08
- 코드와 가중치(`best.pt`)는 DESKTOP 과 같음. 장치마다 다른 것은 이 문서의 **실행 환경**과 **TensorRT 엔진**뿐

## 0. 장치 확인

```bash
cat /etc/nv_tegra_release; lsb_release -ds; df -h / | tail -1; sudo nvpmodel -q | head -2
```

| 항목 | 확인한 값 |
|---|---|
| L4T / JetPack | R39.2.1 / JetPack 7.2.1 (CUDA 13.2, TensorRT 10.16.2) |
| OS | Ubuntu 24.04.5 LTS |
| 저장 | NVMe |
| 전원 모드 | MAXN_SUPER |

- JetPack 6.x (Ubuntu 22.04) 이면 ROS2 Jazzy 를 쓸 수 없음 (Humble 만 가능, 핑키 Jazzy 와 통신 보장 안 됨)

## 1. ROS2 Jazzy

- `/opt/ros/jazzy` 에 설치되어 있었음 (패키지 287개, `cv_bridge`, `demo_nodes_py`, `rosbag2` 포함)
- 확인: 터미널 2개에서 `ros2 run demo_nodes_py talker` / `ros2 run demo_nodes_py listener`
- `~/.bashrc` 에 `echo ... >> ~/.bashrc` 로 줄을 추가할 때 **마지막 줄에 줄바꿈이 없으면 앞 줄 끝에 붙음** → 추가 후 `tail -2 ~/.bashrc` 로 확인

## 2. 전용 가상환경 (시스템 패키지 사용)

```bash
python3 -m venv --system-site-packages ~/venv/traffic_ros
cp setup/constraints.txt ~/venv/traffic_ros/constraints.txt      # 저장소 폴더에서
source ~/venv/traffic_ros/bin/activate
python -c "import numpy, cv2, tensorrt; print(numpy.__version__, numpy.__file__, cv2.__version__, tensorrt.__version__)"
```

- `--system-site-packages`: 가상환경에서 시스템의 TensorRT 파이썬 모듈, ROS2 용 numpy 1.26 을 그대로 씀
- 기존 `~/venv/ros` (다른 작업용, 시스템 패키지 안 봄) 는 바꾸지 않음

## 3. PyTorch (CUDA 13.2 공식 휠)

```bash
python -m pip install --pre torch torchvision \
  --extra-index-url https://download.pytorch.org/whl/cu132 -c ~/venv/traffic_ros/constraints.txt
```

- 설치된 버전: torch 2.15.0+cu132, torchvision 0.29.2+cu132 (다운로드 약 2~3GB)
- **그냥 `pip install torch` 는 쓰지 않음** (데이터센터 GPU 용 휠 → 젯슨 sm_87 에서 `no kernel image`)
- 확인 (실제 GPU 연산까지):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); a=torch.randn(1024,1024,device='cuda'); (a@a).sum().item(); print('matmul OK')"
```

## 4. Ultralytics + 변환 도구

```bash
python -m pip install "ultralytics==8.4.175" onnx onnxslim -c ~/venv/traffic_ros/constraints.txt
echo 'export YOLO_AUTOINSTALL=false' >> ~/venv/traffic_ros/bin/activate
```

- ultralytics 는 **DESKTOP 과 같은 8.4.175** (학습한 `best.pt` 를 그대로 읽기 위해)
- 같이 설치된 것: opencv-python 4.11.0.86 (numpy 1.x 와 맞는 버전으로 자동 선택), onnx 1.23.2, onnxslim 0.1.99
- `YOLO_AUTOINSTALL=false`: Ultralytics 가 제약 없이 패키지를 자동 설치하는 것을 막음 (numpy 2.x 방지). 필요한 패키지가 없으면 오류로 알려 줌

## 5. 실행 별칭

```bash
echo 'alias traffic="source /opt/ros/jazzy/setup.bash; export ROS_DOMAIN_ID=42; source ~/venv/traffic_ros/bin/activate; echo \"ROS2 Jazzy + traffic_ros activated\""' >> ~/.bashrc
source ~/.bashrc
```

- 확인:

```bash
traffic
python -c "import rclpy, cv_bridge, numpy, cv2, torch, ultralytics; b=cv_bridge.CvBridge(); m=b.cv2_to_imgmsg(numpy.zeros((4,4,3),numpy.uint8),'bgr8'); print('OK', numpy.__version__, cv2.__version__, torch.cuda.is_available())"
```

- `ROS_DOMAIN_ID=42`: 같은 번호끼리만 통신. 핑키도 42 로 맞춰야 함

## 6. 파일 옮기기 (DESKTOP → 젯슨)

```bash
sudo systemctl enable --now ssh                                  # 젯슨 (한 번)
```

```bash
J=snake@<젯슨 IP>                                                # DESKTOP, 저장소 폴더
ssh $J "mkdir -p ~/project/traffic_light/runs/detect/det_vehicle_other/weights ~/project/traffic_light/runs/classify/state_cls_none/weights"
scp runs/detect/det_vehicle_other/weights/best.pt $J:~/project/traffic_light/runs/detect/det_vehicle_other/weights/
scp runs/classify/state_cls_none/weights/best.pt  $J:~/project/traffic_light/runs/classify/state_cls_none/weights/
scp drive.mp4 $J:~/project/traffic_light/
```

- 저장소는 젯슨에서도 `~/project/traffic_light` 에 clone (DESKTOP 과 같은 상대 경로)
- 데이터셋은 옮기지 않음 (학습·평가는 DESKTOP)

## 7. TensorRT 엔진 (젯슨에서 만들고 젯슨에서만 사용)

```bash
python tools/export_engine.py --model runs/detect/det_vehicle_other/weights/best.pt --imgsz 736 1280   # 약 7분
python tools/export_engine.py --model runs/classify/state_cls_none/weights/best.pt --imgsz 224          # 약 5분
```

- 결과: `best_736x1280.engine` (세로 736 x 가로 1280), `best_224x224.engine`
- 검출 엔진은 **16:9 영상에 맞춘 직사각형**. 정사각형(1280x1280)은 여백 때문에 전처리·추론이 느림
- 엔진은 JetPack(TensorRT) 버전이 바뀌거나 `best.pt` 가 바뀌면 다시 만듦
- 실행: 코드는 그대로, 모델 인자만 바꿈

```bash
python tools/record_states.py --source drive.mp4 \
  --model runs/detect/det_vehicle_other/weights/best_736x1280.engine --det-classes 0 \
  --cls-model runs/classify/state_cls_none/weights/best_224x224.engine --csv <csv> --limit 300
```

## 8. 측정 결과 (drive.mp4, 검출 + 판별 + 선택, 첫 프레임 제외 중앙값)

| 구성 | ms/프레임 | fps | 원시 상태 (DESKTOP 대비) |
|---|---|---|---|
| PyTorch `.pt` | 69.5 | 14.4 | 300/300 같음 |
| TensorRT 정사각형 1280x1280 + E03 | 52.1 | 19.2 | 300/300 같음 |
| **TensorRT 직사각형 736x1280 + E03** | **34.7** (300프레임) / **27.8** (전체) | **28.8 / 36** | **전체 15,797 중 15,784 같음 (99.92%)** |

- 다른 13프레임은 원래 흔들리던 구간 (120초 황색 + 좌회전, 248~250초 먼 적색, 경계 프레임). **smoother 확정 상태는 15,797/15,797 같음**
- 검출 시간 내역 (직사각형 엔진): 전처리 18.5 (CPU) / 추론 7.7 / 후처리 3.8 ms. 영상 읽기 약 7 ms 별도
- 남은 병목은 CPU 전처리 → 필요하면 GPU 전처리 (검토 항목)
