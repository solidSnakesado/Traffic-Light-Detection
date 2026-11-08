"""
repo_path.py - 노드가 저장소의 기존 모듈(detector.py, selector.py, smoother.py, tools/flow_judges.py)을 불러 올 수 있게 경로 설정
- 저장소 폴더는 노드 파라미터 repo_dir 로 받음 (경로 하드코딩 없음)
- repo_dir 이 비어 있으면 이 파일의 실제 위치에서 찾음
    colcon build --symlink-install 이면 이 파일 = <저장소>/ros2/traffic_light_nodes/traffic_light_nodes/repo_path.py
    -> 위로 3단계 = <저장소>
"""

# =================== 표준 라이브러리 ===========================
import sys                                                                          # 모듈 검색 경로에 저장소 폴더 추가

from pathlib import Path                                                            # 경로 다루기

MARKERS = ("detector.py", "selector.py", "smoother.py", "tools/flow_judges.py")     # 저장소 폴더인지 확인할 파일

# =============================================================
# 저장소 폴더를 찾아 sys.path 에 추가하고 경로를 돌려줌
# =============================================================
def setup_repo_path(repo_dir=""):
    repo    = Path(repo_dir).expanduser() if repo_dir else Path(__file__).resolve().parents[3]
    missing = [m for m in MARKERS if not (repo / m).is_file()]                      # 없는 파일 목록
    if missing:
        raise RuntimeError(f"저장소 폴더가 아닙니다: {repo} (없음: {', '.join(missing)}) "
                           f"-> --symlink-install 로 빌드하거나 -p repo_dir:=<저장소 경로> 지정")
    for p in (repo, repo / "tools"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return repo