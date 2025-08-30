from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
import time
import math
import cv2
import mediapipe as mp
import uvicorn

app = FastAPI()

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 ["http://localhost:3000"] 등으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gesture_value = 0
last_action_time = 0.0       # 마지막 제스처 갱신 시각
last_seen_time = 0.0         # 마지막으로 손을 본 시각
cooldown = 1.0               # 제스처 갱신 쿨다운 (초)
no_hand_reset_delay = 0.3    # 손을 못 본 상태가 이 시간 넘으면 None으로 리셋
current_gesture = None
lock = threading.Lock()
running = True

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def gesture_loop():
    global gesture_value, last_action_time, current_gesture, running, last_seen_time
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            return

        while running:
            ret, frame = cap.read()
            if not ret:
                # 프레임 읽기 실패: 바로 다음 루프로
                time.sleep(0.01)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            now = time.time()

            has_hand = bool(results.multi_hand_landmarks)

            if has_hand:
                # 손은 보임: 최근 본 시각 업데이트
                with lock:
                    last_seen_time = now

                # 쿨다운이 지난 경우에만 새 제스처 계산/적용
                if (now - last_action_time) > cooldown:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    dist = distance(thumb_tip, index_tip)

                    with lock:
                        if dist < 0.1:
                            current_gesture = "OK"
                        else:
                            # OpenCV 영상은 좌->우로 x 증가
                            if thumb_tip.x < pinky_tip.x:
                                current_gesture = "LEFT"
                                gesture_value = max(gesture_value - 1, -100)
                            else:
                                current_gesture = "RIGHT"
                                gesture_value = min(gesture_value + 1, 100)
                        last_action_time = now

            else:
                # 손이 안 보임: 일정 시간 이상 못 보면 None으로 리셋
                with lock:
                    if (now - last_seen_time) > no_hand_reset_delay:
                        current_gesture = None

            time.sleep(0.01)

    except Exception as e:
        print(f"gesture_loop에서 오류 발생: {e}")
    finally:
        cap.release()
        # 창을 띄우지 않으므로 굳이 필요 없지만 안전 차원에서 호출
        cv2.destroyAllWindows()

threading.Thread(target=gesture_loop, daemon=True).start()

@app.get("/gesture")
def get_gesture():
    with lock:
        return {"gesture": current_gesture, "value": gesture_value} #체스터 ok, left, right 반환

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
