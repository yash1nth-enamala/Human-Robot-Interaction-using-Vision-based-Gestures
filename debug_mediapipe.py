import cv2

for i in range(5):  # try first 5 devices
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    if ret:
        print(f"Device {i} works")
    cap.release()
