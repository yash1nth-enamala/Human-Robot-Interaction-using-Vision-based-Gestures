import cv2
import time
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker
from mediapipe.tasks.python.vision import PoseLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode

# ==========================
# Load Pose Model
# ==========================

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=RunningMode.VIDEO
)

pose_landmarker = PoseLandmarker.create_from_options(options)

# ==========================
# Pose Connections (Full Body - 33 landmarks)
# ==========================

POSE_CONNECTIONS = [
    # Face
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),

    # Torso
    (11,12),(11,23),(12,24),(23,24),

    # Left Arm
    (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),

    # Right Arm
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),

    # Left Leg
    (23,25),(25,27),(27,29),(29,31),

    # Right Leg
    (24,26),(26,28),(28,30),(30,32)
]

# ==========================
# Start Webcam
# ==========================

cap = cv2.VideoCapture(0)
pTime = 0

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

            # Draw Skeleton Lines
            for connection in POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start = pose_landmarks[start_idx]
                end = pose_landmarks[end_idx]

                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)

                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Draw Landmark Dots
            for landmark in pose_landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    # ==========================
    # FPS Counter
    # ==========================

    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime != 0 else 0
    pTime = cTime

    cv2.putText(frame, f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2)

    cv2.imshow("Live Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
