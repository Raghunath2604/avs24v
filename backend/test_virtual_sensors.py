"""
Test script for Virtual Sensors
Verifies that all sensors compute correctly without errors.
"""

import numpy as np
import sys
import time

print("=" * 60)
print("SentiGuard Virtual Sensors Test")
print("=" * 60)

# Test 1: Import all modules
print("\n[1/5] Importing modules...")
try:
    from virtual_sensors import VirtualSensors
    from fusion import FusionEngine
    from audio import AudioDetector
    from vision import VisionDetector
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize Virtual Sensors
print("\n[2/5] Initializing Virtual Sensors...")
try:
    vs = VirtualSensors()
    print("✓ VirtualSensors initialized")
    print(f"  - MediaPipe available: {vs.pose_detector is not None}")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    sys.exit(1)

# Test 3: Initialize Fusion Engine
print("\n[3/5] Initializing Fusion Engine...")
try:
    fusion = FusionEngine()
    print("✓ FusionEngine initialized")
    print(f"  - Motion weight: {fusion.motion_weight}")
    print(f"  - Vocal weight: {fusion.vocal_scream_weight}")
    print(f"  - Alarm threshold: {fusion.alarm_threshold}")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    sys.exit(1)

# Test 4: Test sensor computation with dummy data
print("\n[4/5] Testing sensor computation...")
try:
    # Create dummy frame (640x480 black image)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create dummy detections
    dummy_detections = [
        {
            'class': 'person',
            'confidence': 0.85,
            'box': [100, 100, 200, 300]
        }
    ]
    
    # Compute sensors
    sensors = vs.get_all_sensors(
        frame=dummy_frame,
        detections=dummy_detections,
        vocal_score=0.3,
        audio_rms=5.0
    )
    
    print("✓ Sensors computed successfully")
    print(f"  - motion_score: {sensors['motion_score']:.3f}")
    print(f"  - optical_flow_magnitude: {sensors['optical_flow_magnitude']:.3f}")
    print(f"  - loitering_duration: {sensors['loitering_duration']:.1f}s")
    print(f"  - trajectory_erraticity: {sensors['trajectory_erraticity']:.3f}")
    print(f"  - crowd_density: {sensors['crowd_density']:.3f}")
    print(f"  - posture_risk_score: {sensors['posture_risk_score']:.3f}")
    print(f"  - vocal_scream_score: {sensors['vocal_scream_score']:.3f}")
    print(f"  - vocal_event_count: {sensors['vocal_event_count']}")
    print(f"  - ambient_noise_level: {sensors['ambient_noise_level']:.3f}")
    print(f"  - camera_health: {sensors['camera_health']['healthy']}")
    print(f"  - lighting_quality: {sensors['lighting_quality']:.3f}")
    
except Exception as e:
    print(f"✗ Sensor computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test fusion
print("\n[5/5] Testing fusion engine...")
try:
    result = fusion.compute_threat_score(sensors)
    
    print("✓ Fusion computed successfully")
    print(f"  - threat_score: {result['threat_score']:.3f}")
    print(f"  - status: {result['status']}")
    print(f"  - alarm: {result['alarm']}")
    
    # Verify result structure
    assert 'threat_score' in result
    assert 'status' in result
    assert 'alarm' in result
    assert 'sensors_raw' in result
    assert 'sensors_smoothed' in result
    assert 'thresholds' in result
    assert 'weights' in result
    
    print("✓ Result structure validated")
    
except Exception as e:
    print(f"✗ Fusion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test multiple frames (loitering)
print("\n[6/6] Testing temporal features (loitering)...")
try:
    for i in range(5):
        # Same person in same position
        sensors = vs.get_all_sensors(
            frame=dummy_frame,
            detections=dummy_detections,
            vocal_score=0.1,
            audio_rms=2.0
        )
        time.sleep(0.1)
    
    print(f"✓ Loitering duration after 5 frames: {sensors['loitering_duration']:.1f}s")
    assert sensors['loitering_duration'] > 0, "Loitering should increase over time"
    
except Exception as e:
    print(f"✗ Temporal test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# All tests passed
print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED")
print("=" * 60)
print("\nVirtual Sensors system is working correctly!")
print("You can now run: python main.py")
