import cv2
import numpy as np
from virtual_sensors import VirtualSensors

# Create dummy frame (black image)
frame = np.zeros((480, 640, 3), dtype=np.uint8)
# No detections
detections = []
# Dummy audio values
vocal_score = 0.0
audio_rms = 0.0
audio_fft = np.array([])

vs = VirtualSensors()
result = vs.get_all_sensors(frame, detections, vocal_score, audio_rms, audio_fft)
print('Sensor payload (first 20 keys):')
for k in list(result.keys())[:20]:
    print(f"{k}: {result[k]}")
print('...')
print('Total sensors:', len(result))
