import cv2
import time

def list_cameras():
    print("Testing cameras...")
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"Camera {i}: OK ({frame.shape})")
                available.append(i)
            else:
                print(f"Camera {i}: Opened but no frame")
            cap.release()
        else:
            print(f"Camera {i}: Failed to open")
    return available

if __name__ == "__main__":
    cams = list_cameras()
    print(f"Available cameras: {cams}")
