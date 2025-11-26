"""
SentiGuard Main - Simplified & Stable
======================================
Camera + Sample Video + Upload + All 50 Sensors
"""
import cv2
import asyncio
import base64
import time
import os
import numpy as np
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil

# Import local modules
from vision import VisionDetector
from audio import AudioDetector
from virtual_sensors import VirtualSensors
from fusion import FusionEngine

# Config
DEMO_MODE = False
SITE_ID = 1
DEVICE_ID = 42

# System state
class State:
    source_type = "auto"  # auto, sample, upload
    source_changed = False
    uploaded_path = "uploaded_video.mp4"

state = State()

# FastAPI app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Initialize detectors
try:
    vision = VisionDetector()
    print("✓ Vision detector loaded")
except:
    print("⚠️ Vision detector failed, using dummy")
    class DummyVision:
        def detect(self, frame):
            return frame, 0.0, []
        def draw_detections(self, frame, dets):
            return frame
    vision = DummyVision()

audio = AudioDetector()
virtual_sensors = VirtualSensors()
fusion = FusionEngine()

# API endpoints
@app.post("/upload/video")
async def upload_video(file: UploadFile = File(...)):
    try:
        with open(state.uploaded_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        state.source_type = "upload"
        state.source_changed = True
        return {"status": "success", "message": "Video uploaded"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/config/source")
def set_source(source: str):
    if source not in ["auto", "sample", "upload"]:
        return {"status": "error"}
    state.source_type = source
    state.source_changed = True
    return {"status": "success", "source": source}

@app.post("/simulate/demo")
def toggle_demo():
    global DEMO_MODE
    DEMO_MODE = not DEMO_MODE
    return {"demo_mode": DEMO_MODE}

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📡 WebSocket client connected")
    
    def open_video():
        print(f"📹 Opening video source: {state.source_type}")
        
        # Try upload
        if state.source_type == "upload" and os.path.exists(state.uploaded_path):
            print(f"  → Using uploaded video")
            return cv2.VideoCapture(state.uploaded_path)
        
        # Try sample
        if state.source_type == "sample":
            for name in ["demo_video.mp4", "sample_video.mp4"]:
                if os.path.exists(name):
                    print(f"  → Using {name}")
                    return cv2.VideoCapture(name)
        
        # Try camera
        print("  → Searching for camera...")
        for idx in [0, 1, 2]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"  ✓ Camera found at index {idx}")
                    return cap
                cap.release()
        
        # Fallback
        for name in ["demo_video.mp4", "sample_video.mp4"]:
            if os.path.exists(name):
                print(f"  → Fallback to {name}")
                return cv2.VideoCapture(name)
        
        print("  ❌ No video source found!")
        return None
    
    cap = open_video()
    last_detections = []
    last_motion = 0.0
    frame_count = 0
    fps_start = time.time()
    current_fps = 0
    
    try:
        while True:
            # Handle source switch
            if state.source_changed:
                print("🔄 Switching video source...")
                if cap:
                    cap.release()
                cap = open_video()
                state.source_changed = False
            
            # Check cap
            if not cap or not cap.isOpened():
                await asyncio.sleep(1)
                continue
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                # Loop video
                if os.path.exists(state.uploaded_path) or os.path.exists("sample_video.mp4"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            # FPS calculation
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start
                current_fps = 30 / elapsed if elapsed > 0 else 0
                fps_start = time.time()
            
            # Process frame
            frame = cv2.resize(frame, (640, 480))
            cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Vision (smart inference - skip every other frame for performance)
            if frame_count % 2 == 0:
                annotated, motion, detections = vision.detect(frame)
                last_detections = detections
                last_motion = float(motion)
            else:
                # Use cached detection, just draw
                annotated = frame.copy()
                motion = last_motion
                detections = last_detections
            
            # Audio
            vocal_score = float(audio.get_score())
            audio_rms = audio.get_rms()
            
            # All sensors
            all_sensors = virtual_sensors.get_all_sensors(
                frame=frame,
                detections=detections,
                vocal_score=vocal_score,
                audio_rms=audio_rms,
                audio_fft=None
            )
            
            # Fusion
            result = fusion.compute_threat_score(all_sensors)
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', annotated)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
           
            # Build payload with ALL sensors
            payload = {
                "image": f"data:image/jpeg;base64,{img_b64}",
                "event": {
                    "site_id": SITE_ID,
                    "device_id": DEVICE_ID,
                    "timestamp": time.time(),
                    "threat_score": float(result.get('threat_score', 0.0)),
                    "status": result.get('status', 'OK'),
                    **{k: float(v) if isinstance(v, (int, float)) else v for k, v in all_sensors.items()}
                }
            }
            
            await websocket.send_json(payload)
            await asyncio.sleep(0.033)  # 30 FPS
            
    except WebSocketDisconnect:
        print("📡 Client disconnected")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if cap:
            cap.release()

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🛡️  SentiGuard v2.0 - Starting")
    print("=" * 60)
    print(f"Site ID: {SITE_ID} | Device ID: {DEVICE_ID}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8001)
