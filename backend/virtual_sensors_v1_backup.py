"""
Virtual Sensors Module
======================
Computes all derived sensor signals from raw camera and audio data.
Each sensor produces a real-time numeric/boolean value with configurable update rates.

Author: SentiGuard Team
Version: 1.0
"""

import cv2
import numpy as np
import time
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional

# Optional: MediaPipe for pose estimation
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. posture_risk_score will be disabled.")


class VirtualSensors:
    """
    Manages all virtual sensors for the SentiGuard system.
    Maintains state across frames for temporal analysis.
    """
    
    def __init__(self):
        # Motion tracking state
        self.prev_gray = None
        self.optical_flow_history = deque(maxlen=10)
        
        # Object tracking (simple centroid-based)
        self.tracked_objects = {}  # {id: {'centroid': (x,y), 'first_seen': time, 'positions': []}}
        self.next_object_id = 0
        self.max_centroid_dist = 50  # pixels
        
        # Audio state
        self.vocal_events = deque(maxlen=60)  # Last 60 seconds of events
        self.last_event_time = 0
        self.event_cooldown = 1.0  # seconds between counting events
        
        # System health
        self.frame_timestamps = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Performance metrics
        self.frame_count = 0
        self.start_time = time.time()
        
        # Pose estimation (optional)
        self.pose_detector = None
        if MEDIAPIPE_AVAILABLE:
            try:
                mp_pose = mp.solutions.pose
                self.pose_detector = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,  # Lightweight model
        Args:
            detections: List of detection dicts with 'class' and 'confidence'
            
        Returns:
            motion_score [0..1]
        """
        # Check for weapons (high threat)
        has_weapon = any(d['class'] in ['knife', 'scissors'] for d in detections)
        if has_weapon:
            return 1.0
        
        # Check for persons with high confidence
        person_detections = [d for d in detections if d['class'] == 'person']
        if person_detections:
            max_conf = max(d['confidence'] for d in person_detections)
            # For now, person alone is neutral (0.0), but we keep confidence for future use
            return 0.0
        
        return 0.0
    
    def compute_optical_flow(self, frame: np.ndarray) -> float:
        """
        Compute optical flow magnitude between consecutive frames.
        
        Args:
            frame: Current frame (BGR)
            
        Returns:
            optical_flow_magnitude [0..1] - normalized average magnitude
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
        
        try:
            # Compute dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            
            # Compute magnitude
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_magnitude = np.mean(magnitude)
            
            # Normalize (empirically, values rarely exceed 10)
            normalized = min(1.0, avg_magnitude / 10.0)
            
            # Store in history
            self.optical_flow_history.append(normalized)
            
            self.prev_gray = gray
            return float(normalized)
            
        except Exception as e:
            print(f"Optical flow error: {e}")
            self.prev_gray = gray
            return 0.0
    
    def update_tracking(self, detections: List[Dict], current_time: float):
        """
        Update object tracking for loitering and trajectory analysis.
        Uses simple centroid-based tracking.
        
        Args:
            detections: List of detection dicts with 'box' [x1,y1,x2,y2]
            current_time: Current timestamp
        """
        # Extract centroids from current detections
        current_centroids = []
        for d in detections:
            if d['class'] == 'person':
                x1, y1, x2, y2 = d['box']
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                current_centroids.append((cx, cy))
        
        # Match with existing tracked objects
        if len(self.tracked_objects) == 0:
            # Initialize new tracks
            for centroid in current_centroids:
                self.tracked_objects[self.next_object_id] = {
                    'centroid': centroid,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'positions': [centroid]
                }
                self.next_object_id += 1
        else:
            # Match centroids to existing objects (greedy nearest neighbor)
            used_centroids = set()
            updated_objects = {}
            
            for obj_id, obj_data in self.tracked_objects.items():
                best_dist = float('inf')
                best_centroid = None
                best_idx = -1
                
                for idx, centroid in enumerate(current_centroids):
                    if idx in used_centroids:
                        continue
                    dist = np.sqrt((centroid[0] - obj_data['centroid'][0])**2 + 
                                   (centroid[1] - obj_data['centroid'][1])**2)
                    if dist < best_dist and dist < self.max_centroid_dist:
                        best_dist = dist
                        best_centroid = centroid
                        best_idx = idx
                
                if best_centroid is not None:
                    # Update existing object
                    used_centroids.add(best_idx)
                    obj_data['centroid'] = best_centroid
                    obj_data['last_seen'] = current_time
                    obj_data['positions'].append(best_centroid)
                    # Keep only last 30 positions
                    if len(obj_data['positions']) > 30:
                        obj_data['positions'] = obj_data['positions'][-30:]
                    updated_objects[obj_id] = obj_data
            
            # Add new objects for unmatched centroids
            for idx, centroid in enumerate(current_centroids):
                if idx not in used_centroids:
                    updated_objects[self.next_object_id] = {
                        'centroid': centroid,
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'positions': [centroid]
                    }
                    self.next_object_id += 1
            
            self.tracked_objects = updated_objects
        
        # Remove stale objects (not seen for 3 seconds)
        stale_threshold = 3.0
        self.tracked_objects = {
            obj_id: obj_data 
            for obj_id, obj_data in self.tracked_objects.items()
            if current_time - obj_data['last_seen'] < stale_threshold
        }
    
    def compute_loitering_duration(self) -> float:
        """
        Compute maximum loitering duration across all tracked objects.
        
        Returns:
            loitering_duration (seconds) - max time any person stayed in area
        """
        if not self.tracked_objects:
            return 0.0
        
        current_time = time.time()
        max_duration = 0.0
        
        for obj_data in self.tracked_objects.values():
            duration = current_time - obj_data['first_seen']
            
            # Check if object is relatively stationary
            if len(obj_data['positions']) >= 2:
                positions = np.array(obj_data['positions'])
                displacement = np.linalg.norm(positions[-1] - positions[0])
                
                # Only count as loitering if displacement is small
                if displacement < 100:  # pixels
                    max_duration = max(max_duration, duration)
        
        return float(max_duration)
    
    def compute_trajectory_erraticity(self) -> float:
        """
        Compute trajectory erraticity from tracked object movements.
        Measures variance of velocity and direction changes.
        
        Returns:
            trajectory_erraticity [0..1]
        """
        if not self.tracked_objects:
            return 0.0
        
        max_erraticity = 0.0
        
        for obj_data in self.tracked_objects.values():
            positions = obj_data['positions']
            
            if len(positions) < 3:
                continue
            
            # Compute velocities
            velocities = []
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                velocities.append((dx, dy))
            
            if len(velocities) < 2:
                continue
            
            # Compute velocity magnitudes
            magnitudes = [np.sqrt(vx**2 + vy**2) for vx, vy in velocities]
            
            # Compute angles (direction changes)
            angles = []
            for i in range(1, len(velocities)):
                v1 = np.array(velocities[i-1])
                v2 = np.array(velocities[i])
                
                # Avoid division by zero
                mag1 = np.linalg.norm(v1)
                mag2 = np.linalg.norm(v2)
                
                if mag1 > 0.1 and mag2 > 0.1:
                    cos_angle = np.dot(v1, v2) / (mag1 * mag2)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    angles.append(angle)
            
            # Compute erraticity as combination of velocity variance and angle variance
            if len(magnitudes) > 1 and len(angles) > 0:
                vel_variance = np.var(magnitudes)
                angle_variance = np.var(angles)
                
                # Normalize (empirical values)
                erraticity = min(1.0, (vel_variance / 100.0) + (angle_variance / 2.0))
                max_erraticity = max(max_erraticity, erraticity)
        
        return float(max_erraticity)
    
    def compute_crowd_density(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> float:
        """
        Compute crowd density from person detections.
        
        Args:
            detections: List of detection dicts
            frame_shape: (height, width)
            
        Returns:
            crowd_density [0..1] - normalized person count
        """
        person_count = sum(1 for d in detections if d['class'] == 'person')
        
        # Normalize: assume max 10 persons in frame is "crowded"
        normalized = min(1.0, person_count / 10.0)
        
        return float(normalized)
    
    def compute_posture_risk_score(self, frame: np.ndarray) -> float:
        """
        Compute posture risk score using pose estimation.
        Detects raised arms, prone person, falling.
        
        Args:
            frame: Current frame (BGR)
            
        Returns:
            posture_risk_score [0..1]
        """
        if not self.pose_detector:
            return 0.0
        
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb_frame)
            
            if not results.pose_landmarks:
                return 0.0
            
            landmarks = results.pose_landmarks.landmark
            
            # Get key points
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # Raised arms detection
            left_arm_raised = left_wrist.y < left_shoulder.y - 0.1
            right_arm_raised = right_wrist.y < right_shoulder.y - 0.1
            
            if left_arm_raised and right_arm_raised:
                return 0.8  # Both arms raised - possible surrender or distress
            
            # Prone/falling detection (body horizontal)
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_y = (left_hip.y + right_hip.y) / 2
            body_vertical_diff = abs(shoulder_y - hip_y)
            
            if body_vertical_diff < 0.2:  # Body is horizontal
                return 0.9  # Possible fall or prone
            
            return 0.0
            
        except Exception as e:
            print(f"Pose estimation error: {e}")
            return 0.0
    
    def update_vocal_events(self, vocal_score: float):
        """
        Update vocal event history.
        
        Args:
            vocal_score: Current vocal scream score [0..1]
        """
        current_time = time.time()
        
        # Detect event (threshold and cooldown to avoid double-counting)
        if vocal_score > 0.7 and (current_time - self.last_event_time) > self.event_cooldown:
            self.vocal_events.append(current_time)
            self.last_event_time = current_time
    
    def compute_vocal_event_count(self) -> int:
        """
        Compute number of vocal distress events in last 60 seconds.
        
        Returns:
            vocal_event_count (integer)
        """
        current_time = time.time()
        cutoff_time = current_time - 60.0
        
        # Count events within last minute
        count = sum(1 for event_time in self.vocal_events if event_time > cutoff_time)
        
        return int(count)
    
    def compute_ambient_noise_level(self, audio_rms: float) -> float:
        """
        Compute ambient noise level from audio RMS.
        
        Args:
            audio_rms: RMS energy of audio window
            
        Returns:
            ambient_noise_level [0..1]
        """
        # Normalize RMS (empirical max ~100)
        normalized = min(1.0, audio_rms / 50.0)
        return float(normalized)
    
    def compute_camera_health(self) -> Dict:
        """
        Compute camera health metrics.
        
        Returns:
            dict with 'healthy' (bool), 'fps', 'diagnostics'
        """
        current_time = time.time()
        self.frame_timestamps.append(current_time)
        self.frame_count += 1
        
        # Compute FPS
        if len(self.frame_timestamps) >= 2:
            time_span = self.frame_timestamps[-1] - self.frame_timestamps[0]
            if time_span > 0:
                fps = len(self.frame_timestamps) / time_span
            else:
                fps = 0
        else:
            fps = 0
        
        # Detect frame freeze (time since last frame)
        time_since_last = current_time - self.last_frame_time
        frame_frozen = time_since_last > 2.0  # Consider frozen if > 2s gap
        
        self.last_frame_time = current_time
        
        # Health check
        healthy = fps > 5.0 and not frame_frozen
        
        return {
            'healthy': bool(healthy),
            'fps': float(fps),
            'frame_frozen': bool(frame_frozen),
            'time_since_last_frame': float(time_since_last)
        }
    
    def compute_lighting_quality(self, frame: np.ndarray) -> float:
        """
        Compute lighting quality from frame brightness histogram.
        
        Args:
            frame: Current frame (BGR)
            
        Returns:
            lighting_quality [0..1]
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute mean brightness
        mean_brightness = np.mean(gray)
        
        # Optimal range: 80-180 (out of 255)
        # Too dark (< 50) or too bright (> 200) reduces quality
        
        if 80 <= mean_brightness <= 180:
            quality = 1.0
        elif mean_brightness < 80:
            # Too dark
            quality = max(0.0, mean_brightness / 80.0)
        else:
            # Too bright
            quality = max(0.0, 1.0 - (mean_brightness - 180) / 75.0)
        
        return float(quality)
    
    def get_all_sensors(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        vocal_score: float,
        audio_rms: float = 0.0
    ) -> Dict:
        """
        Compute all virtual sensors in one call.
        
        Args:
            frame: Current frame (BGR)
            detections: Vision detections
            vocal_score: Audio vocal scream score
            audio_rms: Audio RMS energy (optional)
            
        Returns:
            dict with all sensor values
        """
        current_time = time.time()
        
        # Update tracking
        self.update_tracking(detections, current_time)
        
        # Update vocal events
        self.update_vocal_events(vocal_score)
        
        # Compute all sensors
        sensors = {
            # Motion sensors
            'motion_score': self.compute_motion_score(detections),
            'optical_flow_magnitude': self.compute_optical_flow(frame),
            'loitering_duration': self.compute_loitering_duration(),
            'trajectory_erraticity': self.compute_trajectory_erraticity(),
            
            # Crowd & pose
            'crowd_density': self.compute_crowd_density(detections, frame.shape[:2]),
            'posture_risk_score': self.compute_posture_risk_score(frame),
            
            # Audio sensors
            'vocal_scream_score': float(vocal_score),
            'vocal_event_count': self.compute_vocal_event_count(),
            'ambient_noise_level': self.compute_ambient_noise_level(audio_rms),
            
            # System health
            'camera_health': self.compute_camera_health(),
            'lighting_quality': self.compute_lighting_quality(frame),
            
            # Optional metadata
            'temperature': None,  # Could be filled from external API
            'environment_meta': {
                'frame_count': self.frame_count,
                'tracked_objects': len(self.tracked_objects),
                'uptime_seconds': current_time - self.start_time
            }
        }
        
        return sensors
    
    def close(self):
        """Cleanup resources."""
        if self.pose_detector:
            self.pose_detector.close()
