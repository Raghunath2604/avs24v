"""
Virtual Sensors Module - Enhanced v2.0
=======================================
Computes all 22 virtual sensors from raw camera and audio data.
Includes enterprise-level threat detection capabilities.

Author: SentiGuard Team
Version: 2.0 Enhanced
"""

import cv2
import numpy as np
import time
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional

# Optional: MediaPipe for pose estimation
# Optional: MediaPipe for pose estimation
try:
    import mediapipe as mp
    # FIXME: MediaPipe causing crash on some systems, disabled for stability verification
    MEDIAPIPE_AVAILABLE = False 
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Pose features will be limited.")


class VirtualSensors:
    """
    Manages all 22 virtual sensors for the SentiGuard system.
    Maintains state across frames for temporal analysis.
    
    Sensors 1-12: Core sensors
    Sensors 13-22: Advanced enterprise sensors
    """
    
    def __init__(self):
        # Motion tracking state
        self.prev_gray = None
        self.optical_flow_history = deque(maxlen=10)
        
        # Object tracking (centroid-based)
        self.tracked_objects = {}  # {id: {'centroid', 'first_seen', 'positions', 'velocities'}}
        self.next_object_id = 0
        self.max_centroid_dist = 50
        
        # Audio state
        self.vocal_events = deque(maxlen=60)
        self.last_event_time = 0
        self.event_cooldown = 1.0
        self.audio_history = deque(maxlen=50)  # For glass break / metal impact
        
        # System health
        self.frame_timestamps = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Performance
        self.frame_count = 0
        self.start_time = time.time()
        
        # NEW: Advanced sensor state
        self.prev_brightness = None  # Tampering detection
        self.velocity_history = {}  # {obj_id: [velocities]} for velocity spike
        
        # Geo-zones (configurable)
        self.geo_zones = [
            {
                'name': 'Restricted Area',
                'polygon': [(0.2, 0.2), (0.8, 0.2), (0.8, 0.5), (0.2, 0.5)]
            }
        ]
        
        # Pose estimation
        self.pose_detector = None
        if MEDIAPIPE_AVAILABLE:
            try:
                mp_pose = mp.solutions.pose
                self.pose_detector = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception as e:
                print(f"Warning: Could not initialize pose detector: {e}")
    
    # ==================== CORE SENSORS (1-12) ====================
    
    def compute_motion_score(self, detections: List[Dict]) -> float:
        """Sensor #1: Basic motion from detections"""
        has_weapon = any(d['class'] in ['knife', 'scissors'] for d in detections)
        if has_weapon:
            return 1.0
        
        person_detections = [d for d in detections if d['class'] == 'person']
        if person_detections:
            return 0.0  # Neutral
        
        return 0.0
    
    def compute_optical_flow(self, frame: np.ndarray) -> float:
        """Sensor #2: Optical flow magnitude"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
        
        try:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_magnitude = np.mean(magnitude)
            normalized = min(1.0, avg_magnitude / 10.0)
            
            self.optical_flow_history.append(normalized)
            self.prev_gray = gray
            return float(normalized)
        except Exception as e:
            self.prev_gray = gray
            return 0.0
    
    def update_tracking(self, detections: List[Dict], current_time: float):
        """Update object tracking with velocity tracking"""
        current_centroids = []
        for d in detections:
            if d['class'] == 'person':
                x1, y1, x2, y2 = d['box']
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                current_centroids.append((cx, cy))
        
        if len(self.tracked_objects) == 0:
            for centroid in current_centroids:
                self.tracked_objects[self.next_object_id] = {
                    'centroid': centroid,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'positions': [centroid],
                    'velocities': []
                }
                self.next_object_id += 1
        else:
            used_centroids = set()
            updated_objects = {}
            
            for obj_id, obj_data in self.tracked_objects.items():
                best_dist = float('inf')
                best_centroid = None
                best_idx = -1
                
                for idx, centroid in enumerate(current_centroids):
                    if idx in used_centroids:
                        continue
                    dist = np.sqrt(
                        (centroid[0] - obj_data['centroid'][0])**2 +
                        (centroid[1] - obj_data['centroid'][1])**2
                    )
                    if dist < best_dist and dist < self.max_centroid_dist:
                        best_dist = dist
                        best_centroid = centroid
                        best_idx = idx
                
                if best_centroid is not None:
                    used_centroids.add(best_idx)
                    
                    # Calculate velocity
                    if len(obj_data['positions']) > 0:
                        prev_pos = obj_data['positions'][-1]
                        velocity = np.sqrt(
                            (best_centroid[0] - prev_pos[0])**2 +
                            (best_centroid[1] - prev_pos[1])**2
                        )
                        obj_data['velocities'].append(velocity)
                        if len(obj_data['velocities']) > 20:
                            obj_data['velocities'] = obj_data['velocities'][-20:]
                    
                    obj_data['centroid'] = best_centroid
                    obj_data['last_seen'] = current_time
                    obj_data['positions'].append(best_centroid)
                    
                    if len(obj_data['positions']) > 30:
                        obj_data['positions'] = obj_data['positions'][-30:]
                    
                    updated_objects[obj_id] = obj_data
            
            for idx, centroid in enumerate(current_centroids):
                if idx not in used_centroids:
                    updated_objects[self.next_object_id] = {
                        'centroid': centroid,
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'positions': [centroid],
                        'velocities': []
                    }
                    self.next_object_id += 1
            
            self.tracked_objects = updated_objects
        
        # Remove stale
        stale_threshold = 3.0
        self.tracked_objects = {
            obj_id: obj_data 
            for obj_id, obj_data in self.tracked_objects.items()
            if current_time - obj_data['last_seen'] < stale_threshold
        }
    
    def compute_loitering_duration(self) -> float:
        """Sensor #3: Loitering duration"""
        if not self.tracked_objects:
            return 0.0
        
        current_time = time.time()
        max_duration = 0.0
        
        for obj_data in self.tracked_objects.values():
            duration = current_time - obj_data['first_seen']
            
            if len(obj_data['positions']) >= 2:
                positions = np.array(obj_data['positions'])
                displacement = np.linalg.norm(positions[-1] - positions[0])
                
                if displacement < 100:
                    max_duration = max(max_duration, duration)
        
        return float(max_duration)
    
    def compute_trajectory_erraticity(self) -> float:
        """Sensor #4: Trajectory erraticity"""
        if not self.tracked_objects:
            return 0.0
        
        max_erraticity = 0.0
        
        for obj_data in self.tracked_objects.values():
            positions = obj_data['positions']
            
            if len(positions) < 3:
                continue
            
            velocities = []
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                velocities.append((dx, dy))
            
            if len(velocities) < 2:
                continue
            
            magnitudes = [np.sqrt(vx**2 + vy**2) for vx, vy in velocities]
            
            angles = []
            for i in range(1, len(velocities)):
                v1 = np.array(velocities[i-1])
                v2 = np.array(velocities[i])
                
                mag1 = np.linalg.norm(v1)
                mag2 = np.linalg.norm(v2)
                
                if mag1 > 0.1 and mag2 > 0.1:
                    cos_angle = np.dot(v1, v2) / (mag1 * mag2)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    angles.append(angle)
            
            if len(magnitudes) > 1 and len(angles) > 0:
                vel_variance = np.var(magnitudes)
                angle_variance = np.var(angles)
                erraticity = min(1.0, (vel_variance / 100.0) + (angle_variance / 2.0))
                max_erraticity = max(max_erraticity, erraticity)
        
        return float(max_erraticity)
    
    def compute_crowd_density(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> float:
        """Sensor #5: Crowd density"""
        person_count = sum(1 for d in detections if d['class'] == 'person')
        normalized = min(1.0, person_count / 10.0)
        return float(normalized)
    
    def compute_posture_risk_score(self, frame: np.ndarray) -> float:
        """Sensor #6: Posture risk (raised arms, prone, falling)"""
        if not self.pose_detector:
            return 0.0
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb_frame)
            
            if not results.pose_landmarks:
                return 0.0
            
            landmarks = results.pose_landmarks.landmark
            
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # Raised arms
            left_arm_raised = left_wrist.y < left_shoulder.y - 0.1
            right_arm_raised = right_wrist.y < right_shoulder.y - 0.1
            
            if left_arm_raised and right_arm_raised:
                return 0.8
            
            # Prone/falling
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_y = (left_hip.y + right_hip.y) / 2
            body_vertical_diff = abs(shoulder_y - hip_y)
            
            if body_vertical_diff < 0.2:
                return 0.9
            
            return 0.0
        except Exception:
            return 0.0
    
    def update_vocal_events(self, vocal_score: float):
        """Update vocal event history"""
        current_time = time.time()
        
        if vocal_score > 0.7 and (current_time - self.last_event_time) > self.event_cooldown:
            self.vocal_events.append(current_time)
            self.last_event_time = current_time
    
    def compute_vocal_event_count(self) -> int:
        """Sensor #8: Vocal event count"""
        current_time = time.time()
        cutoff_time = current_time - 60.0
        count = sum(1 for event_time in self.vocal_events if event_time > cutoff_time)
        return int(count)
    
    def compute_ambient_noise_level(self, audio_rms: float) -> float:
        """Sensor #9: Ambient noise level"""
        normalized = min(1.0, audio_rms / 50.0)
        return float(normalized)
    
    def compute_camera_health(self) -> Dict:
        """Sensor #10: Camera health"""
        current_time = time.time()
        self.frame_timestamps.append(current_time)
        self.frame_count += 1
        
        if len(self.frame_timestamps) >= 2:
            time_span = self.frame_timestamps[-1] - self.frame_timestamps[0]
            fps = len(self.frame_timestamps) / time_span if time_span > 0 else 0
        else:
            fps = 0
        
        time_since_last = current_time - self.last_frame_time
        frame_frozen = time_since_last > 2.0
        
        self.last_frame_time = current_time
        healthy = fps > 5.0 and not frame_frozen
        
        return {
            'healthy': bool(healthy),
            'fps': float(fps),
            'frame_frozen': bool(frame_frozen),
            'time_since_last_frame': float(time_since_last)
        }
    
    def compute_lighting_quality(self, frame: np.ndarray) -> float:
        """Sensor #11: Lighting quality"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if 80 <= mean_brightness <= 180:
            quality = 1.0
        elif mean_brightness < 80:
            quality = max(0.0, mean_brightness / 80.0)
        else:
            quality = max(0.0, 1.0 - (mean_brightness - 180) / 75.0)
        
        return float(quality)
    
    # ==================== ADVANCED SENSORS (13-22) ====================
    
    def compute_weapon_score(self, detections: List[Dict]) -> float:
        """Sensor #13: Enhanced weapon detection with confidence scoring"""
        weapon_classes = ['knife', 'scissors', 'gun', 'pistol']
        weapon_detections = [d for d in detections if d['class'] in weapon_classes]
        
        if not weapon_detections:
            return 0.0
        
        max_score = max(d['confidence'] for d in weapon_detections)
        
        # Boost if multiple weapons
        if len(weapon_detections) > 1:
            max_score = min(1.0, max_score * 1.2)
        
        return float(max_score)
    
    def compute_object_sharpness(self, frame: np.ndarray, detections: List[Dict]) -> float:
        """Sensor #14: Object sharpness (sharp edges detection)"""
        max_sharpness = 0.0
        
        for detection in detections:
            try:
                x1, y1, x2, y2 = map(int, detection['box'])
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                edges = cv2.Canny(roi, 50, 150)
                edge_density = np.sum(edges) / (roi.shape[0] * roi.shape[1] * 255.0)
                
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                corners = cv2.goodFeaturesToTrack(gray_roi, 25, 0.01, 10)
                corner_count = len(corners) if corners is not None else 0
                
                sharpness = min(1.0, edge_density * 2.0 + corner_count * 0.02)
                max_sharpness = max(max_sharpness, sharpness)
            except Exception:
                continue
        
        return float(max_sharpness)
    
    def compute_body_velocity_spike(self) -> float:
        """Sensor #15: Sudden acceleration detection"""
        max_spike = 0.0
        
        for obj_data in self.tracked_objects.values():
            velocities = obj_data.get('velocities', [])
            
            if len(velocities) >= 3:
                recent = velocities[-3:]
                accelerations = [recent[i] - recent[i-1] for i in range(1, len(recent))]
                
                if accelerations:
                    max_accel = max(accelerations)
                    spike_score = min(1.0, max_accel / 20.0)
                    max_spike = max(max_spike, spike_score)
        
        return float(max_spike)
    
    def compute_aggressive_pose(self, frame: np.ndarray) -> float:
        """Sensor #16: Aggressive posture (attacking stance)"""
        if not self.pose_detector:
            return 0.0
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb_frame)
            
            if not results.pose_landmarks:
                return 0.0
            
            landmarks = results.pose_landmarks.landmark
            
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            # Punching pose (arm extended forward)
            wrist_forward = left_wrist.z < nose.z - 0.2 or right_wrist.z < nose.z - 0.2
            
            # Raised fist
            arm_raised = left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y
            
            if wrist_forward and arm_raised:
                return 0.9
            elif wrist_forward or arm_raised:
                return 0.5
            
            return 0.0
        except Exception:
            return 0.0
    
    def compute_fight_interaction(self) -> float:
        """Sensor #17: Two+ persons in aggressive interaction"""
        if len(self.tracked_objects) < 2:
            return 0.0
        
        objects = list(self.tracked_objects.values())
        max_interaction = 0.0
        
        for i in range(len(objects)):
            for j in range(i+1, len(objects)):
                pos1 = objects[i]['centroid']
                pos2 = objects[j]['centroid']
                dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                if dist < 50:  # Very close
                    velocities1 = objects[i].get('velocities', [])
                    velocities2 = objects[j].get('velocities', [])
                    
                    vel1 = velocities1[-1] if velocities1 else 0
                    vel2 = velocities2[-1] if velocities2 else 0
                    
                    if vel1 > 5 and vel2 > 5:
                        interaction_score = min(1.0, (50 - dist) / 50 * (vel1 + vel2) / 20)
                        max_interaction = max(max_interaction, interaction_score)
        
        return float(max_interaction)
    
    def compute_panic_crowd_movement(self) -> float:
        """Sensor #18: Irregular fast crowd dispersion"""
        if len(self.tracked_objects) < 3:
            return 0.0
        
        velocities = []
        for obj_data in self.tracked_objects.values():
            obj_vels = obj_data.get('velocities', [])
            if obj_vels:
                velocities.append(obj_vels[-1])
        
        if not velocities:
            return 0.0
        
        avg_vel = np.mean(velocities)
        vel_variance = np.var(velocities)
        
        panic_score = min(1.0, (avg_vel / 10) * (vel_variance / 100))
        
        return float(panic_score)
    
    def compute_glass_break_audio(self, audio_fft: np.ndarray, audio_rms: float) -> float:
        """Sensor #19: Glass breaking detection"""
        if len(audio_fft) < 100:
            return 0.0
        
        # Glass: 4-8 kHz range
        high_freq_start = int(len(audio_fft) * 0.4)
        high_freq_end = int(len(audio_fft) * 0.8)
        
        high_freq_energy = np.sum(np.abs(audio_fft[high_freq_start:high_freq_end]))
        total_energy = np.sum(np.abs(audio_fft))
        
        if total_energy == 0:
            return 0.0
        
        high_freq_ratio = high_freq_energy / total_energy
        
        score = 0.0
        if high_freq_ratio > 0.3 and audio_rms > 5.0:
            score = min(1.0, high_freq_ratio * audio_rms / 20)
        
        return float(score)
    
    def compute_metal_impact_audio(self, audio_fft: np.ndarray, audio_rms: float) -> float:
        """Sensor #20: Metal impact detection"""
        if len(audio_fft) < 100:
            return 0.0
        
        # Metal: 1-4 kHz range
        mid_high_start = int(len(audio_fft) * 0.2)
        mid_high_end = int(len(audio_fft) * 0.4)
        
        mid_high_energy = np.sum(np.abs(audio_fft[mid_high_start:mid_high_end]))
        total_energy = np.sum(np.abs(audio_fft))
        
        if total_energy == 0:
            return 0.0
        
        freq_ratio = mid_high_energy / total_energy
        
        score = 0.0
        if freq_ratio > 0.25 and audio_rms > 8.0:
            score = min(1.0, freq_ratio * audio_rms / 25)
        
        return float(score)
    
    def _point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """Point-in-polygon test (ray casting)"""
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def compute_environment_intrusion(self, frame_shape: Tuple[int, int]) -> float:
        """Sensor #21: Geo-fence violation"""
        max_intrusion = 0.0
        
        h, w = frame_shape[:2]
        
        for obj_data in self.tracked_objects.values():
            centroid = obj_data['centroid']
            norm_x = centroid[0] / w
            norm_y = centroid[1] / h
            
            for zone in self.geo_zones:
                if self._point_in_polygon((norm_x, norm_y), zone['polygon']):
                    max_intrusion = 1.0
                    break
        
        return float(max_intrusion)
    
    def compute_tampering_score(self, frame: np.ndarray) -> float:
        """Sensor #22: Camera tampering detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if self.prev_brightness is None:
            self.prev_brightness = brightness
            return 0.0
        
        brightness_drop = self.prev_brightness - brightness
        tampering = 0.0
        
        # Camera covered (very dark)
        if brightness < 10:
            tampering = 1.0
        # Sudden drop
        elif brightness_drop > 50:
            tampering = 0.8
        # Uniform color (blocked)
        else:
            std_dev = np.std(gray)
            if std_dev < 5:
                tampering = 0.7
        
        self.prev_brightness = brightness
        return float(tampering)
    
    # ==================== NEW SENSORS (23-50) ====================
    
    ### VISION SENSORS (continued) ###
    
    def compute_fall_detect_score(self, frame: np.ndarray) -> float:
        """Sensor #23: Fall detection from pose"""
        if not self.pose_detector:
            return 0.0
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb_frame)
            
            if not results.pose_landmarks:
                return 0.0
            
            landmarks = results.pose_landmarks.landmark
            
            # Get key points
            nose = landmarks[0]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # Compute body angle
            hip_y = (left_hip.y + right_hip.y) / 2
            
            # If nose is close to hip level → horizontal body → fall
            if abs(nose.y - hip_y) < 0.15:
                return 1.0
            
            return 0.0
        except Exception:
            return 0.0
    
    def compute_running_score(self) -> float:
        """Sensor #24: Fast forward motion detection"""
        if not self.tracked_objects:
            return 0.0
        
        max_speed = 0.0
        
        for obj_data in self.tracked_objects.values():
            velocities = obj_data.get('velocities', [])
            
            if len(velocities) >= 3:
                # Average of last 3 velocities
                recent_vel = np.mean(velocities[-3:])
                
                # Running: sustained high velocity (>15 pixels/frame)
                if recent_vel > 15:
                    speed_score = min(1.0, recent_vel / 30.0)
                    max_speed = max(max_speed, speed_score)
        
        return float(max_speed)
    
    def compute_visibility_score(self, frame: np.ndarray) -> float:
        """Sensor #25: Scene clarity (contrast, sharpness)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute contrast (standard deviation)
        contrast = np.std(gray) / 128.0  # Normalize
        
        # Compute sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian) / 1000.0  # Normalize
        
        # Combined visibility score
        visibility = min(1.0, (contrast + sharpness) / 2.0)
        
        return float(visibility)
    
    def compute_interaction_object_score(self, detections: List[Dict]) -> float:
        """Sensor #26: Touching equipment/doors/vehicles"""
        # For now, detect if person is near detected objects
        person_boxes = [d['box'] for d in detections if d['class'] == 'person']
        object_boxes = [d['box'] for d in detections if d['class'] not in ['person']]
        
        if not person_boxes or not object_boxes:
            return 0.0
        
        max_interaction = 0.0
        
        for p_box in person_boxes:
            px1, py1, px2, py2 = p_box
            p_center = ((px1 + px2) / 2, (py1 + py2) / 2)
            
            for o_box in object_boxes:
                ox1, oy1, ox2, oy2 = o_box
                o_center = ((ox1 + ox2) / 2, (oy1 + oy2) / 2)
                
                # Compute distance
                dist = np.sqrt((p_center[0] - o_center[0])**2 + (p_center[1] - o_center[1])**2)
                
                # Close interaction: <100 pixels
                if dist < 100:
                    interaction = 1.0 - (dist / 100.0)
                    max_interaction = max(max_interaction, interaction)
        
        return float(max_interaction)
    
    def compute_object_drop_or_pick_score(self) -> float:
        """Sensor #27: Object interaction (simplified)"""
        # Placeholder: Would need object change detection
        # For now, use sudden changes in tracked object count
        if not hasattr(self, 'prev_object_count'):
            self.prev_object_count = 0
        
        current_count = len(self.tracked_objects)
        count_change = abs(current_count - self.prev_object_count)
        
        self.prev_object_count = current_count
        
        # Sudden change in object count
        score = min(1.0, count_change / 3.0)
        
        return float(score)
    
    ### AUDIO SENSORS (continued) ###
    
    def compute_aggressive_voice_score(self, audio_fft: np.ndarray, audio_rms: float) -> float:
        """Sensor #28: Aggressive tone detection"""
        if len(audio_fft) < 100:
            return 0.0
        
        # Aggressive voice: mid frequencies (500-2000 Hz) + loud
        mid_freq_start = int(len(audio_fft) * 0.1)
        mid_freq_end = int(len(audio_fft) * 0.3)
        
        mid_freq_energy = np.sum(np.abs(audio_fft[mid_freq_start:mid_freq_end]))
        total_energy = np.sum(np.abs(audio_fft))
        
        if total_energy == 0:
            return 0.0
        
        ratio = mid_freq_energy / total_energy
        
        # Loud + concentrated mid frequencies
        if ratio > 0.4 and audio_rms > 10.0:
            score = min(1.0, ratio * audio_rms / 30.0)
            return float(score)
        
        return 0.0
    
    def compute_gunshot_like_score(self, audio_rms: float) -> float:
        """Sensor #29: Gunshot-like impulse detection"""
        if not hasattr(self, 'prev_rms'):
            self.prev_rms = 0.0
        
        # Gunshot: very sudden loud spike
        rms_spike = audio_rms - self.prev_rms
        
        self.prev_rms = audio_rms
        
        # Spike > 50 units
        if rms_spike > 50:
            score = min(1.0, rms_spike / 100.0)
            return float(score)
        
        return 0.0
    
    def compute_high_pitch_distress_score(self, audio_fft: np.ndarray, audio_rms: float) -> float:
        """Sensor #30: High-pitched sustained distress"""
        if len(audio_fft) < 100:
            return 0.0
        
        # Very high frequencies >3kHz sustained
        high_freq_start = int(len(audio_fft) * 0.5)
        high_freq_end = int(len(audio_fft) * 0.9)
        
        high_freq_energy = np.sum(np.abs(audio_fft[high_freq_start:high_freq_end]))
        total_energy = np.sum(np.abs(audio_fft))
        
        if total_energy == 0:
            return 0.0
        
        ratio = high_freq_energy / total_energy
        
        # High pitch + loud + sustained
        if ratio > 0.25 and audio_rms > 4.0:
            score = min(1.0, ratio * audio_rms / 15.0)
            return float(score)
        
        return 0.0
    
    def compute_low_frequency_thud_score(self, audio_fft: np.ndarray, audio_rms: float) -> float:
        """Sensor #31: Low-frequency impact (<500Hz)"""
        if len(audio_fft) < 100:
            return 0.0
        
        # Low frequencies
        low_freq_end = int(len(audio_fft) * 0.05)
        
        low_freq_energy = np.sum(np.abs(audio_fft[:low_freq_end]))
        total_energy = np.sum(np.abs(audio_fft))
        
        if total_energy == 0:
            return 0.0
        
        ratio = low_freq_energy / total_energy
        
        # Low frequency + loud
        if ratio > 0.3 and audio_rms > 6.0:
            score = min(1.0, ratio * audio_rms / 20.0)
            return float(score)
        
        return 0.0
    
    def compute_speech_activity_rate(self, audio_rms: float) -> float:
        """Sensor #32: Voice activity detection rate"""
        # Simple VAD: RMS > threshold
        if not hasattr(self, 'speech_frames'):
            self.speech_frames = deque(maxlen=30)  # Last 30 frames
        
        is_speech = audio_rms > 2.0  # Speech threshold
        self.speech_frames.append(1 if is_speech else 0)
        
        # Rate of speech in last second
        speech_rate = sum(self.speech_frames) / len(self.speech_frames) if self.speech_frames else 0.0
        
        return float(speech_rate)
    
    def compute_silence_anomaly_score(self, audio_rms: float) -> float:
        """Sensor #33: Unusual silence detection"""
        # Silence when there should be ambient noise
        if not hasattr(self, 'baseline_noise'):
            self.baseline_noise = audio_rms
        
        # Update baseline (slow moving average)
        self.baseline_noise = 0.99 * self.baseline_noise + 0.01 * audio_rms
        
        # Anomaly: much quieter than baseline
        if self.baseline_noise > 1.0 and audio_rms < 0.2:
            score = min(1.0, (self.baseline_noise - audio_rms) / self.baseline_noise)
            return float(score)
        
        return 0.0
    
    def compute_wind_noise_score(self, audio_fft: np.ndarray) -> float:
        """Sensor #34: Wind noise detection"""
        if len(audio_fft) < 100:
            return 0.0
        
        # Wind: broadband low-frequency noise
        low_freq_energy = np.sum(np.abs(audio_fft[:int(len(audio_fft) * 0.1)]))
        total_energy = np.sum(np.abs(audio_fft))
        
        if total_energy == 0:
            return 0.0
        
        # High low-frequency content but distributed
        low_ratio = low_freq_energy / total_energy
        
        if low_ratio > 0.6:  # Broadband low freq
            score = min(1.0, (low_ratio - 0.6) / 0.4)
            return float(score)
        
        return 0.0
    
    def compute_audio_spectral_anomaly_score(self, audio_fft: np.ndarray) -> float:
        """Sensor #35: Unusual audio spectrum"""
        if len(audio_fft) < 100:
            return 0.0
 
        # Compute spectral centroid
        freqs = np.arange(len(audio_fft))
        magnitudes = np.abs(audio_fft)
        
        if np.sum(magnitudes) == 0:
            return 0.0
        
        centroid = np.sum(freqs * magnitudes) / np.sum(magnitudes)
        normalized_centroid = centroid / len(audio_fft)
        
        # Anomaly: very high or very low centroid
        if normalized_centroid < 0.1 or normalized_centroid > 0.8:
            score = min(1.0, abs(normalized_centroid - 0.5) * 2)
            return float(score)
        
        return 0.0
    
    ### TRACKING / BEHAVIOR SENSORS ###
    
    def compute_person_entry_count(self) -> int:
        """Sensor #36: People entering zone"""
        if not hasattr(self, 'entry_count'):
            self.entry_count = 0
            self.tracked_ids_seen = set()
        
        # Count new IDs
        current_ids = set(self.tracked_objects.keys())
        new_ids = current_ids - self.tracked_ids_seen
        
        self.entry_count += len(new_ids)
        self.tracked_ids_seen = current_ids
        
        return int(self.entry_count)
    
    def compute_person_exit_count(self) -> int:
        """Sensor #37: People exiting zone"""
        if not hasattr(self, 'exit_count'):
            self.exit_count = 0
            self.last_tracked_ids = set()
        
        current_ids = set(self.tracked_objects.keys())
        exited_ids = self.last_tracked_ids - current_ids
        
        self.exit_count += len(exited_ids)
        self.last_tracked_ids = current_ids
        
        return int(self.exit_count)
    
    def compute_group_formation_score(self) -> float:
        """Sensor #38: Group/cluster detection"""
        if len(self.tracked_objects) < 2:
            return 0.0
        
        # Compute pairwise distances
        centroids = [obj['centroid'] for obj in self.tracked_objects.values()]
        
        close_pairs = 0
        total_pairs = 0
        
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                dist = np.sqrt(
                    (centroids[i][0] - centroids[j][0])**2 +
                    (centroids[i][1] - centroids[j][1])**2
                )
                
                if dist < 150:  # Close proximity
                    close_pairs += 1
                total_pairs += 1
        
        if total_pairs == 0:
            return 0.0
        
        # Ratio of close pairs
        group_score = close_pairs / total_pairs
        
        return float(group_score)
    
    def compute_approach_towards_camera_score(self) -> float:
        """Sensor #39: Approaching camera (size increase)"""
        max_approach = 0.0
        
        for obj_data in self.tracked_objects.values():
            positions = obj_data.get('positions', [])
            
            if len(positions) >= 5:
                # Compute bounding box size change (simplified by position spread)
                early_positions = positions[:2]
                recent_positions = positions[-2:]
                
                early_spread = np.std([p[0] for p in early_positions] + [p[1] for p in early_positions])
                recent_spread = np.std([p[0] for p in recent_positions] + [p[1] for p in recent_positions])
                
                # Approaching: spread increases
                if recent_spread > early_spread:
                    approach = min(1.0, (recent_spread - early_spread) / early_spread if early_spread > 0 else 0)
                    max_approach = max(max_approach, approach)
        
        return float(max_approach)
    
    def compute_following_behavior_score(self) -> float:
        """Sensor #40: One person following another"""
        if len(self.tracked_objects) < 2:
            return 0.0
        
        objects = list(self.tracked_objects.values())
        max_following = 0.0
        
        for i in range(len(objects)):
            for j in range(len(objects)):
                if i == j:
                    continue
                
                pos_i = objects[i].get('positions', [])
                pos_j = objects[j].get('positions', [])
                
                if len(pos_i) >= 3 and len(pos_j) >= 3:
                    # Check if i follows j's path
                    correlation = 0.0
                    for k in range(min(3, len(pos_i), len(pos_j))):
                        dist = np.sqrt(
                            (pos_i[-(k+1)][0] - pos_j[-(k+2)][0])**2 +
                            (pos_i[-(k+1)][1] - pos_j[-(k+2)][1])**2
                        )
                        if dist < 30:  # Following closely
                            correlation += 1
                    
                    following_score = correlation / 3.0
                    max_following = max(max_following, following_score)
        
        return float(max_following)
    
    def compute_vehicle_presence_score(self, detections: List[Dict]) -> float:
        """Sensor #41: Vehicle detection"""
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
        has_vehicle = any(d['class'] in vehicle_classes for d in detections)
        
        if has_vehicle:
            # Return max confidence
            vehicle_confs = [d['confidence'] for d in detections if d['class'] in vehicle_classes]
            return float(max(vehicle_confs))
        
        return 0.0
    
    def compute_vehicle_speed_estimate(self, detections: List[Dict]) -> float:
        """Sensor #42: Vehicle speed estimation"""
        # Track vehicle centroids
        if not hasattr(self, 'vehicle_positions'):
            self.vehicle_positions = {}
        
        vehicle_classes = ['car', 'truck', 'bus']
        vehicle_boxes = [d['box'] for d in detections if d['class'] in vehicle_classes]
        
        if not vehicle_boxes:
            return 0.0
        
        # Simple speed estimation from position change
        current_centroids = []
        for box in vehicle_boxes:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            current_centroids.append((cx, cy))
        
        max_speed = 0.0
        
        if hasattr(self, 'prev_vehicle_centroids') and self.prev_vehicle_centroids:
            for curr in current_centroids:
                for prev in self.prev_vehicle_centroids:
                    dist = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
                    # Normalize speed
                    speed = min(1.0, dist / 50.0)
                    max_speed = max(max_speed, speed)
        
        self.prev_vehicle_centroids = current_centroids
        
        return float(max_speed)
    
    def compute_person_identity_hash(self) -> int:
        """Sensor #43: Anonymous person tracking hash"""
        # Simple hash based on active tracked IDs
        active_ids = sorted(list(self.tracked_objects.keys()))
        
        if not active_ids:
            return 0
        
        # Create hash from IDs
        hash_val = sum(active_ids) % 10000
        
        return int(hash_val)
    
    ### SYSTEM / ENVIRONMENTAL SENSORS ###
    
    def compute_device_cpu_temp(self) -> float:
        """Sensor #44: CPU temperature"""
        try:
            import psutil
            temps = psutil.sensors_temperatures()
            
            if temps:
                # Get first available temperature
                for name, entries in temps.items():
                    if entries:
                        return float(entries[0].current)
        except Exception:
            pass
        
        return 0.0
    
    def compute_device_ram_usage(self) -> float:
        """Sensor #45: RAM usage percentage"""
        try:
            import psutil
            return float(psutil.virtual_memory().percent / 100.0)
        except Exception:
            return 0.0
    
    def compute_device_storage_usage(self) -> float:
        """Sensor #46: Storage usage percentage"""
        try:
            import psutil
            return float(psutil.disk_usage('/').percent / 100.0)
        except Exception:
            return 0.0
    
    def compute_network_quality_score(self) -> float:
        """Sensor #47: Network quality (simplified)"""
        # Placeholder: Would need actual network test
        # For now, assume good network
        return 1.0
    
    def compute_time_of_day_context(self) -> float:
        """Sensor #48: Time context (night weight)"""
        from datetime import datetime
        
        current_hour = datetime.now().hour
        
        # Night hours (22:00 - 06:00) = higher risk
        if 22 <= current_hour or current_hour < 6:
            return 1.0  # Night
        elif 6 <= current_hour < 8:
            return 0.5  # Dawn
        elif 18 <= current_hour < 22:
            return 0.7  # Dusk
        else:
            return 0.3  # Day
    
    # ==================== MAIN SENSOR AGGREGATOR ====================
    
    def get_all_sensors(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        vocal_score: float,
        audio_rms: float = 0.0,
        audio_fft: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute ALL 50 virtual sensors in one call.
        
        Returns dict with all sensor values plus metadata.
        """
        current_time = time.time()
        
        # Update tracking
        self.update_tracking(detections, current_time)
        
        # Update vocal events
        self.update_vocal_events(vocal_score)
        
        # Default empty audio_fft if not provided
        if audio_fft is None:
            audio_fft = np.array([])
        
        # Compute ALL 50 sensors
        sensors = {
            ### VISION SENSORS (1-20) ###
            'motion_score': self.compute_motion_score(detections),
            'optical_flow_magnitude': self.compute_optical_flow(frame),
            'trajectory_erraticity': self.compute_trajectory_erraticity(),
            'loitering_duration': self.compute_loitering_duration(),
            'crowd_density': self.compute_crowd_density(detections, frame.shape[:2]),
            'posture_risk_score': self.compute_posture_risk_score(frame),
            'fall_detect_score': self.compute_fall_detect_score(frame),
            'running_score': self.compute_running_score(),
            'fight_interaction_score': self.compute_fight_interaction(),
            'aggressive_pose_score': self.compute_aggressive_pose(frame),
            'weapon_score': self.compute_weapon_score(detections),
            'object_sharpness_score': self.compute_object_sharpness(frame, detections),
            'tampering_score': self.compute_tampering_score(frame),
            'camera_health': self.compute_camera_health(),
            'lighting_quality': self.compute_lighting_quality(frame),
            'visibility_score': self.compute_visibility_score(frame),
            'environment_intrusion_score': self.compute_environment_intrusion(frame.shape[:2]),
            'interaction_object_score': self.compute_interaction_object_score(detections),
            'panic_crowd_movement_score': self.compute_panic_crowd_movement(),
            'object_drop_or_pick_score': self.compute_object_drop_or_pick_score(),
            
            ### AUDIO SENSORS (21-35) ###
            'vocal_scream_score': float(vocal_score),
            'aggressive_voice_score': self.compute_aggressive_voice_score(audio_fft, audio_rms),
            'metal_impact_audio_score': self.compute_metal_impact_audio(audio_fft, audio_rms),
            'glass_break_audio_score': self.compute_glass_break_audio(audio_fft, audio_rms),
            'gunshot_like_score': self.compute_gunshot_like_score(audio_rms),
            'high_pitch_distress_score': self.compute_high_pitch_distress_score(audio_fft, audio_rms),
            'low_frequency_thud_score': self.compute_low_frequency_thud_score(audio_fft, audio_rms),
            'ambient_noise_level': self.compute_ambient_noise_level(audio_rms),
            'speech_activity_rate': self.compute_speech_activity_rate(audio_rms),
            'vocal_event_count': self.compute_vocal_event_count(),
            'audio_direction_confidence': 0.0,  # Placeholder (requires multi-mic)
            'background_sound_class': 0.0,  # Placeholder (requires classifier)
            'silence_anomaly_score': self.compute_silence_anomaly_score(audio_rms),
            'wind_noise_score': self.compute_wind_noise_score(audio_fft),
            'audio_spectral_anomaly_score': self.compute_audio_spectral_anomaly_score(audio_fft),
            
            ### TRACKING / BEHAVIOR SENSORS (36-43) ###
            'person_entry_count': self.compute_person_entry_count(),
            'person_exit_count': self.compute_person_exit_count(),
            'group_formation_score': self.compute_group_formation_score(),
            'approach_towards_camera_score': self.compute_approach_towards_camera_score(),
            'following_behavior_score': self.compute_following_behavior_score(),
            'body_velocity_spike_score': self.compute_body_velocity_spike(),
            'repeat_intruder_score': 0.0,  # Placeholder (requires re-ID model)
            'vehicle_presence_score': self.compute_vehicle_presence_score(detections),
            'vehicle_speed_estimate': self.compute_vehicle_speed_estimate(detections),
            'person_identity_hash': self.compute_person_identity_hash(),
            
            ### SYSTEM / ENVIRONMENTAL SENSORS (44-48) ###
            'device_cpu_temp': self.compute_device_cpu_temp(),
            'device_ram_usage': self.compute_device_ram_usage(),
            'device_storage_usage': self.compute_device_storage_usage(),
            'network_quality_score': self.compute_network_quality_score(),
            'time_of_day_context': self.compute_time_of_day_context(),
            
            ### METADATA ###
            'temperature': None,
            'environment_meta': {
                'frame_count': self.frame_count,
                'tracked_objects': len(self.tracked_objects),
                'uptime_seconds': current_time - self.start_time,
                'total_sensors': 50
            }
        }
        
        return sensors
    
    def close(self):
        """Cleanup resources"""
        if self.pose_detector:
            self.pose_detector.close()
