import cv2
import math
import pygame
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- MediaPipe Setup ----------------
MODEL_PATH = "hand_landmarker.task"

def smooth(prev, new, alpha=0.15):
    return prev * (1 - alpha) + new * alpha

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

# ---------------- Pygame Setup ----------------
pygame.init()
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("5DOF Virtual Robotic Arm (Leap Motion Style)")
clock = pygame.time.Clock()

# Arm lengths
L1, L2, L3 = 120, 100, 60
BASE = np.array([450, 480])

# ---------------- Helper Functions ----------------
def finger_angle(a, b, c):
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cosang, -1.0, 1.0)))

def draw_arm(base, a1, a2, a3, grip):
    # Forward kinematics
    x1 = base[0] + L1 * math.cos(a1)
    y1 = base[1] - L1 * math.sin(a1)

    x2 = x1 + L2 * math.cos(a1 + a2)
    y2 = y1 - L2 * math.sin(a1 + a2)

    x3 = x2 + L3 * math.cos(a1 + a2 + a3)
    y3 = y2 - L3 * math.sin(a1 + a2 + a3)

    # Arm
    pygame.draw.line(screen, (255,255,255), base, (x1,y1), 8)
    pygame.draw.line(screen, (200,200,200), (x1,y1), (x2,y2), 6)
    pygame.draw.line(screen, (170,170,170), (x2,y2), (x3,y3), 4)

    # Gripper (two jaws)
    jaw_len = 25
    spread = grip

    angle = a1 + a2 + a3
    perp = angle + math.pi / 2

    j1 = (
        x3 + jaw_len * math.cos(angle) + spread * math.cos(perp),
        y3 - jaw_len * math.sin(angle) - spread * math.sin(perp)
    )
    j2 = (
        x3 + jaw_len * math.cos(angle) - spread * math.cos(perp),
        y3 - jaw_len * math.sin(angle) + spread * math.sin(perp)
    )

    pygame.draw.line(screen, (255,120,120), (x3,y3), j1, 4)
    pygame.draw.line(screen, (255,120,120), (x3,y3), j2, 4)

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0)
frame_id = 0

# Joint states
base_yaw = math.radians(90)
shoulder = math.radians(45)
elbow = math.radians(45)
wrist = 0
grip_open = 12  # pixels

try:
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect_for_video(mp_image, frame_id)
        frame_id += 1

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]

            # Wrist → base & shoulder
            wx, wy = lm[0].x, lm[0].y
            base_yaw = np.interp(wx, [0,1], [math.radians(30), math.radians(150)])
            shoulder = np.interp(wy, [0,1], [math.radians(120), math.radians(30)])

            # Elbow from hand up/down (wrist vs middle MCP)
            hand_height = lm[0].y - lm[9].y  # wrist - palm center
            elbow_target = np.interp(
                hand_height,
                [-0.15, 0.15],
                [math.radians(20), math.radians(120)]
            )
            elbow = smooth(elbow, elbow_target)

            # 🔄 FIXED: Palm orientation → wrist (INVERTED)
            wrist = -np.interp(
                lm[9].x - lm[0].x,
                [-0.2, 0.2],
                [-1.0, 1.0]
            )

            # ✋ Gripper control (thumb–index pinch)
            pinch = math.dist(
                (lm[4].x, lm[4].y),
                (lm[8].x, lm[8].y)
            )
            grip_open = np.interp(pinch, [0.02, 0.15], [4, 18])
            grip_open = np.clip(grip_open, 4, 18)

        screen.fill((20,20,20))
        draw_arm(BASE, base_yaw, elbow, wrist, grip_open)
        pygame.display.flip()
        clock.tick(30)

finally:
    cap.release()
    detector.close()
    pygame.quit()
