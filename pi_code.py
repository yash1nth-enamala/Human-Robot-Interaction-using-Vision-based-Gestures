import socket
import RPi.GPIO as GPIO
import time

# ==============================
# GPIO SETUP
# ==============================

GPIO.setmode(GPIO.BCM)

BASE_PIN = 17
SHOULDER_PIN = 18
ELBOW_PIN = 27
WRIST_PIN = 22

GPIO.setup(BASE_PIN, GPIO.OUT)
GPIO.setup(SHOULDER_PIN, GPIO.OUT)
GPIO.setup(ELBOW_PIN, GPIO.OUT)
GPIO.setup(WRIST_PIN, GPIO.OUT)

base_pwm = GPIO.PWM(BASE_PIN, 50)
shoulder_pwm = GPIO.PWM(SHOULDER_PIN, 50)
elbow_pwm = GPIO.PWM(ELBOW_PIN, 50)
wrist_pwm = GPIO.PWM(WRIST_PIN, 50)

base_pwm.start(0)
shoulder_pwm.start(0)
elbow_pwm.start(0)
wrist_pwm.start(0)

# ==============================
# SERVO CONTROL FUNCTION
# ==============================

def set_servo(pwm, angle):
    angle = max(0, min(180, angle))
    duty = 2 + (angle / 18)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.02)
    pwm.ChangeDutyCycle(0)

# ==============================
# SOCKET SERVER SETUP
# ==============================

HOST = ""     # Listen on all interfaces
PORT = 5005

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

print("🤖 Robot Arm Server Started")
print("Waiting for PC connection...")

conn, addr = server.accept()
print("Connected by:", addr)

# ==============================
# INITIAL POSITIONS
# ==============================

current_base = 90
current_shoulder = 90
current_elbow = 90
current_wrist = 90

set_servo(base_pwm, current_base)
set_servo(shoulder_pwm, current_shoulder)
set_servo(elbow_pwm, current_elbow)
set_servo(wrist_pwm, current_wrist)

# ==============================
# MAIN LOOP
# ==============================

try:
    while True:
        data = conn.recv(1024).decode().strip()

        if data:
            try:
                base, shoulder, elbow, wrist = map(int, data.split(","))

                # Smooth movement
                current_base = current_base * 0.7 + base * 0.3
                current_shoulder = current_shoulder * 0.7 + shoulder * 0.3
                current_elbow = current_elbow * 0.7 + elbow * 0.3
                current_wrist = current_wrist * 0.7 + wrist * 0.3

                set_servo(base_pwm, current_base)
                set_servo(shoulder_pwm, current_shoulder)
                set_servo(elbow_pwm, current_elbow)
                set_servo(wrist_pwm, current_wrist)

            except:
                print("Invalid data received:", data)

except KeyboardInterrupt:
    print("\nShutting down...")

finally:
    base_pwm.stop()
    shoulder_pwm.stop()
    elbow_pwm.stop()
    wrist_pwm.stop()
    GPIO.cleanup()
    conn.close()
    server.close()
