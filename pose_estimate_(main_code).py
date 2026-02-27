import cv2
import time
import math
import socket
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker
from mediapipe.tasks.python.vision import PoseLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode

# ==============================
# RASPBERRY PI CONNECTION
# ==============================

PI_IP = "192.168.1.50"   # 🔥 CHANGE THIS
PORT = 5005

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((PI_IP, PORT))
print("Connected to Raspberry Pi")

# ==============================
# LOAD MEDIAPIPE MODEL
# ==============================

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=RunningMode.VIDEO
)

pose_landmarker = PoseLandmarker.create_from_options(options)

# ==============================
# ANGLE CALCULATION FUNCTION
# ==============================

def calculate_angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c

    angle = math.degrees(
        math.atan2(cy - by, cx - bx) -
        math.atan2(ay - by, ax - bx)
    )

    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle

    return angle

# ==============================
# START WEBCAM
# ==============================

cap = cv2.VideoCapture(0)
pTime = 0
smooth_angle = 90  # starting servo angle

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp = int(time.time() * 1000)
    result = pose_landmarker.detect_for_video(mp_image, timestamp)

    if result.pose_landmarks:
        for pose_landmarks in result.pose_landmarks:

            h, w, _ = frame.shape

            # RIGHT ARM LANDMARKS
            shoulder = pose_landmarks[12]
            elbow = pose_landmarks[14]
            wrist = pose_landmarks[16]

            shoulder_pt = (int(shoulder.x * w), int(shoulder.y * h))
            elbow_pt = (int(elbow.x * w), int(elbow.y * h))
            wrist_pt = (int(wrist.x * w), int(wrist.y * h))

            # Draw points
            cv2.circle(frame, shoulder_pt, 8, (0,255,0), -1)
            cv2.circle(frame, elbow_pt, 8, (0,255,0), -1)
            cv2.circle(frame, wrist_pt, 8, (0,255,0), -1)

            cv2.line(frame, shoulder_pt, elbow_pt, (255,0,255), 3)
            cv2.line(frame, elbow_pt, wrist_pt, (255,0,255), 3)

            # Calculate elbow angle
            elbow_angle = calculate_angle(shoulder_pt, elbow_pt, wrist_pt)

            # Limit angle for servo
            elbow_angle = max(0, min(180, elbow_angle))

            # Smooth movement (reduces jitter)
            smooth_angle = smooth_angle * 0.7 + elbow_angle * 0.3

            # Send to Raspberry Pi
            try:
                client.send((str(int(smooth_angle)) + "\n").encode())
            except:
                print("Connection lost")
                break

            # Display angle
            cv2.putText(frame, f"Elbow: {int(smooth_angle)}",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

    # FPS Counter
    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime != 0 else 0
    pTime = cTime

    cv2.putText(frame, f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,255), 2)

    cv2.imshow("Right Arm Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
client.close()
cv2.destroyAllWindows()
