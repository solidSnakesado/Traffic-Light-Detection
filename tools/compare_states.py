"""
tools/compare_states.py - 프레임별 원시 상태 csv 2개 비교 (ROS2 없이 실행)
- 기준 (--ref)  : tools/record_states.py 가 만든 csv
- 비교 (--test) : ROS2 기록 노드 raw_recorder 가 만든 csv (또는 다른 장치의 record_states csv)
- 프레임 번호로 맞춰서 비교. 같은 프레임이 여러 줄이면 마지막 줄을 씀
- 비교 열: state, n_det, n_cand, conf (차이 허용값 --conf-tol), x1~y2 (차이 허용값 --box-tol)
- 출력: 프레임 수, 빠진 프레임, state 일치율, 열별 불일치 수, 불일치 프레임 앞부분

실행 (저장소 폴더에서):
  python tools/compare_states.py --ref ~/datasets/traffic_light/states_drive_e03.csv --test ~/datasets/traffic_light/raw_node.csv
  기준 csv 의 앞부분만 비교: --limit 300
"""

# =================== 표준 라이브러리 ===========================
import argparse                                                             # 명령행 인자
import csv                                                                  # csv 읽기

from collections    import Counter                                          # 불일치 종류별 개수
from pathlib        import Path                                             # 경로 다루기

COLS    = ["state", "n_det", "n_cand", "conf", "x1", "y1", "x2", "y2"]      # 비교할 열
BOX     = ["x1", "y1", "x2", "y2"]                                          # 박스 좌표 열

# =============================================================
# 명령행 인자
# =============================================================
def parse_args():
    pass

# =============================================================
# csv -> {프레임 번호: 행}, 중복 프레임 수
# =============================================================
def load(path, limit=0):
    pass

# =============================================================
# 행 2개 -> 다른 열 목록, 박스 차이 최대값, conf 차이
# =============================================================
def diff_row(a, b, conf_tol, box_tol):
    pass

# =============================================================
# 비교 실행
# =============================================================
def main():
    pass

if __name__ == "__main__":
    main()