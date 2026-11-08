# 신호등 상태 1프레임 (원시 상태 /traffic_light/raw, 확정 상태 /traffic_light/state 공용)
std_msgs/Header header      # 발행 시각
uint32  frame               # 영상 프레임 번호 (0부터)
float64 video_time          # 영상 시각 (초) = 프레임 번호 / 영상 FPS -> smoother 기준 시각
string  state               # RED, YELLOW, GREEN, LEFT, RED_LEFT, GREEN_LEFT / UNKNOWN (켜진 후보 없음) / NONE (후보 없음)
float32 conf                # 대상 박스 신뢰도 (대상 없으면 0)
int32   x1                  # 대상 박스 왼쪽 위 x (원본 영상 픽셀, 대상 없으면 0)
int32   y1                  # 대상 박스 왼쪽 위 y
int32   x2                  # 대상 박스 오른쪽 아래 x
int32   y2                  # 대상 박스 오른쪽 아래 y
uint16  n_det               # 검출 박스 수
uint16  n_cand              # selector 1단계 통과 후보 수
