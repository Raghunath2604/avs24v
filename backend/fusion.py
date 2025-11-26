"""
Fusion Engine with EMA Smoothing and Hysteresis
================================================
Implements multi-sensor fusion with temporal smoothing and alarm hysteresis.

Formula: threat = Σ_i (w_i * s_i_smoothed) where Σ_i w_i = 1
Smoothing: Exponential Moving Average (EMA) with configurable window
Hysteresis: Requires M consecutive windows above threshold to trigger alarm

Author: SentiGuard Team
Version: 2.0
"""

from collections import deque
from typing import Dict, List
import time


class FusionEngine:
    """
    Enhanced fusion engine that combines multiple virtual sensors
    with temporal smoothing and hysteresis for robust threat detection.
    """
    
    def __init__(self):
        # Core sensor weights (original 8)
        self.motion_weight = 0.20  # Reduced to make room for weapon_score
        self.vocal_scream_weight = 0.20  # Reduced
        self.optical_flow_weight = 0.03
        self.loitering_weight = 0.03
        self.posture_weight = 0.03
        self.crowd_weight = 0.01
        self.trajectory_weight = 0.01
        self.ambient_noise_weight = 0.01  # Used for reduction
        
        # NEW: Advanced sensor weights (10 sensors = 42% total weight)
        self.weapon_score_weight = 0.15  # HIGH priority
        self.tampering_weight = 0.10  # Camera blocking
        self.fight_interaction_weight = 0.08  # Physical altercation
        self.aggressive_pose_weight = 0.05  # Attack stance
        self.body_velocity_spike_weight = 0.03  # Sudden movement
        self.glass_break_weight = 0.03  # Breaking sound
        self.metal_impact_weight = 0.03  # Weapon sound
        self.intrusion_weight = 0.05  # Geo-fence violation
        self.panic_crowd_weight = 0.02  # Crowd panic
        self.object_sharpness_weight = 0.02  # Sharp objects
        
        # Total: 1.00 (100%)
        
        # Thresholds
        self.soft_threshold = 0.60
        self.alarm_threshold = 0.75
        
        # Configuration
        self.low_bandwidth = False
        
        # EMA smoothing parameters
        self.smoothing_window_seconds = 1.5
        self.ema_alpha = 0.3  # Weight for new value
        
        # Sensor history for smoothing (EMA)
        self.sensor_ema = {
            'motion_score': 0.0,
            'vocal_scream_score': 0.0,
            'optical_flow_magnitude': 0.0,
            'loitering_duration': 0.0,
            'posture_risk_score': 0.0,
            'crowd_density': 0.0,
            'trajectory_erraticity': 0.0,
            'ambient_noise_level': 0.0,
            'weapon_score': 0.0,
            'object_sharpness_score': 0.0,
            'body_velocity_spike_score': 0.0,
            'aggressive_pose_score': 0.0,
            'fight_interaction_score': 0.0,
            'panic_crowd_movement_score': 0.0,
            'glass_break_audio_score': 0.0,
            'metal_impact_audio_score': 0.0,
            'environment_intrusion_score': 0.0,
            'tampering_score': 0.0
        }
        
        # Threat score history for hysteresis
        self.threat_history = deque(maxlen=10)
        self.threat_timestamps = deque(maxlen=10)
        
        # Hysteresis parameters
        self.hysteresis_consecutive_windows = 2
        self.hysteresis_duration_seconds = 2.0
        
        # Alarm state
        self.alarm_active = False
        self.alarm_start_time = None
        
        # History for debugging/analysis
        self.history = []
        self.history_limit = 100
        
    def update_config(
        self,
        motion_weight=None,
        vocal_weight=None,
        soft_threshold=None,
        alarm_threshold=None,
        low_bandwidth=None,
        ema_alpha=None,
        hysteresis_windows=None
    ):
        """Update fusion configuration parameters."""
        if motion_weight is not None:
            self.motion_weight = motion_weight
        if vocal_weight is not None:
            self.vocal_scream_weight = vocal_weight
        if soft_threshold is not None:
            self.soft_threshold = soft_threshold
        if alarm_threshold is not None:
            self.alarm_threshold = alarm_threshold
        if low_bandwidth is not None:
            self.low_bandwidth = low_bandwidth
        if ema_alpha is not None:
            self.ema_alpha = ema_alpha
        if hysteresis_windows is not None:
            self.hysteresis_consecutive_windows = hysteresis_windows
    
    def _normalize_sensor(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize sensor value to [0, 1] range."""
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val) if max_val > min_val else value))
    
    def _apply_ema_smoothing(self, sensor_name: str, new_value: float) -> float:
        """
        Apply Exponential Moving Average smoothing.
        
        EMA formula: EMA_t = α * value_t + (1 - α) * EMA_{t-1}
        where α (alpha) is the smoothing factor.
        """
        old_ema = self.sensor_ema.get(sensor_name, new_value)
        new_ema = self.ema_alpha * new_value + (1 - self.ema_alpha) * old_ema
        self.sensor_ema[sensor_name] = new_ema
        return new_ema
    
    def compute_threat_score(self, sensors: Dict) -> Dict:
        """
        Compute fused threat score from all virtual sensors.
        
        Args:
            sensors: Dict containing all sensor values
            
        Returns:
            dict with threat_score, status, sensor details, etc.
        """
        current_time = time.time()
        
        # Extract and normalize sensor values - CORE SENSORS
        motion_score = self._normalize_sensor(sensors.get('motion_score', 0.0))
        vocal_scream = self._normalize_sensor(sensors.get('vocal_scream_score', 0.0))
        optical_flow = self._normalize_sensor(sensors.get('optical_flow_magnitude', 0.0))
        
        # Loitering: normalize to [0,1], assume 30s = 1.0
        loitering_raw = sensors.get('loitering_duration', 0.0)
        loitering = self._normalize_sensor(loitering_raw, 0.0, 30.0)
        
        posture = self._normalize_sensor(sensors.get('posture_risk_score', 0.0))
        crowd = self._normalize_sensor(sensors.get('crowd_density', 0.0))
        trajectory = self._normalize_sensor(sensors.get('trajectory_erraticity', 0.0))
        ambient_noise = self._normalize_sensor(sensors.get('ambient_noise_level', 0.0))
        
        # Extract and normalize - ADVANCED SENSORS (13-22)
        weapon_score = self._normalize_sensor(sensors.get('weapon_score', 0.0))
        object_sharpness = self._normalize_sensor(sensors.get('object_sharpness_score', 0.0))
        body_velocity_spike = self._normalize_sensor(sensors.get('body_velocity_spike_score', 0.0))
        aggressive_pose = self._normalize_sensor(sensors.get('aggressive_pose_score', 0.0))
        fight_interaction = self._normalize_sensor(sensors.get('fight_interaction_score', 0.0))
        panic_crowd = self._normalize_sensor(sensors.get('panic_crowd_movement_score', 0.0))
        glass_break = self._normalize_sensor(sensors.get('glass_break_audio_score', 0.0))
        metal_impact = self._normalize_sensor(sensors.get('metal_impact_audio_score', 0.0))
        intrusion = self._normalize_sensor(sensors.get('environment_intrusion_score', 0.0))
        tampering = self._normalize_sensor(sensors.get('tampering_score', 0.0))
        
        # Apply EMA smoothing to CORE sensors
        motion_smoothed = self._apply_ema_smoothing('motion_score', motion_score)
        vocal_smoothed = self._apply_ema_smoothing('vocal_scream_score', vocal_scream)
        optical_smoothed = self._apply_ema_smoothing('optical_flow_magnitude', optical_flow)
        loitering_smoothed = self._apply_ema_smoothing('loitering_duration', loitering)
        posture_smoothed = self._apply_ema_smoothing('posture_risk_score', posture)
        crowd_smoothed = self._apply_ema_smoothing('crowd_density', crowd)
        trajectory_smoothed = self._apply_ema_smoothing('trajectory_erraticity', trajectory)
        ambient_smoothed = self._apply_ema_smoothing('ambient_noise_level', ambient_noise)
        
        # Apply EMA smoothing to ADVANCED sensors
        weapon_smoothed = self._apply_ema_smoothing('weapon_score', weapon_score)
        object_sharpness_smoothed = self._apply_ema_smoothing('object_sharpness_score', object_sharpness)
        body_velocity_smoothed = self._apply_ema_smoothing('body_velocity_spike_score', body_velocity_spike)
        aggressive_pose_smoothed = self._apply_ema_smoothing('aggressive_pose_score', aggressive_pose)
        fight_interaction_smoothed = self._apply_ema_smoothing('fight_interaction_score', fight_interaction)
        panic_crowd_smoothed = self._apply_ema_smoothing('panic_crowd_movement_score', panic_crowd)
        glass_break_smoothed = self._apply_ema_smoothing('glass_break_audio_score', glass_break)
        metal_impact_smoothed = self._apply_ema_smoothing('metal_impact_audio_score', metal_impact)
        intrusion_smoothed = self._apply_ema_smoothing('environment_intrusion_score', intrusion)
        tampering_smoothed = self._apply_ema_smoothing('tampering_score', tampering)
        
        # Compute fused threat score - ALL 22 SENSORS
        threat_score = (
            # Core sensors (58% total)
            self.motion_weight * motion_smoothed +
            self.vocal_scream_weight * vocal_smoothed +
            self.optical_flow_weight * optical_smoothed +
            self.loitering_weight * loitering_smoothed +
            self.posture_weight * posture_smoothed +
            self.crowd_weight * crowd_smoothed +
            self.trajectory_weight * trajectory_smoothed +
            # Advanced sensors (42% total)
            self.weapon_score_weight * weapon_smoothed +
            self.object_sharpness_weight * object_sharpness_smoothed +
            self.body_velocity_spike_weight * body_velocity_smoothed +
            self.aggressive_pose_weight * aggressive_pose_smoothed +
            self.fight_interaction_weight * fight_interaction_smoothed +
            self.panic_crowd_weight * panic_crowd_smoothed +
            self.glass_break_weight * glass_break_smoothed +
            self.metal_impact_weight * metal_impact_smoothed +
            self.intrusion_weight * intrusion_smoothed +
            self.tampering_weight * tampering_smoothed
        )
        
        # Apply ambient noise reduction (high ambient noise reduces confidence)
        # If ambient noise is very high (>0.8), reduce threat score slightly
        if ambient_smoothed > 0.8:
            noise_penalty = (ambient_smoothed - 0.8) * 0.2  # Max 4% reduction
            threat_score = max(0.0, threat_score - noise_penalty)
        
        # Store in history
        self.threat_history.append(threat_score)
        self.threat_timestamps.append(current_time)
        
        # Apply hysteresis for alarm decision
        alarm = self._check_alarm_hysteresis(threat_score, current_time)
        
        # Determine status
        status = "SAFE"
        if alarm:
            status = "ALARM"
        elif threat_score >= self.soft_threshold:
            status = "WARNING"
        else:
            # Check if any single sensor is elevated
            if motion_smoothed > self.soft_threshold or vocal_smoothed > self.soft_threshold:
                status = "WARNING"
        
        # Prepare result
        result = {
            "threat_score": float(threat_score),
            "status": status,
            "alarm": alarm,
            
            # Raw sensor values
            "sensors_raw": {
                "motion_score": float(motion_score),
                "vocal_scream_score": float(vocal_scream),
                "optical_flow_magnitude": float(optical_flow),
                "loitering_duration": float(loitering_raw),
                "posture_risk_score": float(posture),
                "crowd_density": float(crowd),
                "trajectory_erraticity": float(trajectory),
                "ambient_noise_level": float(ambient_noise),
                "vocal_event_count": int(sensors.get('vocal_event_count', 0)),
            },
            
            # Smoothed sensor values
            "sensors_smoothed": {
                "motion_score": float(motion_smoothed),
                "vocal_scream_score": float(vocal_smoothed),
                "optical_flow_magnitude": float(optical_smoothed),
                "loitering_duration": float(loitering_smoothed),
                "posture_risk_score": float(posture_smoothed),
                "crowd_density": float(crowd_smoothed),
                "trajectory_erraticity": float(trajectory_smoothed),
                "ambient_noise_level": float(ambient_smoothed),
            },
            
            # Configuration
            "thresholds": {
                "soft": float(self.soft_threshold),
                "alarm": float(self.alarm_threshold)
            },
            
            "weights": {
                "motion": float(self.motion_weight),
                "vocal_scream": float(self.vocal_scream_weight),
                "optical_flow": float(self.optical_flow_weight),
                "loitering": float(self.loitering_weight),
                "posture": float(self.posture_weight),
                "crowd": float(self.crowd_weight),
                "trajectory": float(self.trajectory_weight),
            },
            
            # System health
            "camera_health": sensors.get('camera_health', {'healthy': True}),
            "lighting_quality": float(sensors.get('lighting_quality', 1.0)),
            
            # Metadata
            "timestamp": current_time,
            "alarm_duration": self._get_alarm_duration(current_time) if alarm else 0.0,
        }
        
        # Add to history
        self.history.append(result)
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]
        
        return result
    
    def _check_alarm_hysteresis(self, threat_score: float, current_time: float) -> bool:
        """
        Check if alarm should trigger based on hysteresis rules.
        
        Rules:
        1. Threat score >= alarm_threshold for M consecutive windows, OR
        2. Threat score >= alarm_threshold sustained for D seconds
        
        Returns:
            bool - True if alarm should be active
        """
        # Check consecutive windows
        if len(self.threat_history) >= self.hysteresis_consecutive_windows:
            recent_threats = list(self.threat_history)[-self.hysteresis_consecutive_windows:]
            all_above_threshold = all(t >= self.alarm_threshold for t in recent_threats)
            
            if all_above_threshold:
                if not self.alarm_active:
                    self.alarm_active = True
                    self.alarm_start_time = current_time
                return True
        
        # Check sustained duration
        if len(self.threat_history) >= 2 and len(self.threat_timestamps) >= 2:
            # Find first time threat went above threshold
            first_high_idx = None
            for i in range(len(self.threat_history) - 1, -1, -1):
                if self.threat_history[i] >= self.alarm_threshold:
                    first_high_idx = i
                else:
                    break
            
            if first_high_idx is not None:
                duration = current_time - self.threat_timestamps[first_high_idx]
                if duration >= self.hysteresis_duration_seconds:
                    if not self.alarm_active:
                        self.alarm_active = True
                        self.alarm_start_time = current_time
                    return True
        
        # Check if alarm should deactivate
        if self.alarm_active and threat_score < self.alarm_threshold * 0.8:  # 20% below threshold
            self.alarm_active = False
            self.alarm_start_time = None
        
        return self.alarm_active
    
    def _get_alarm_duration(self, current_time: float) -> float:
        """Get duration of current alarm in seconds."""
        if self.alarm_start_time is not None:
            return current_time - self.alarm_start_time
        return 0.0
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get recent fusion history."""
        return self.history[-limit:]
    
    def reset(self):
        """Reset all state (useful for testing)."""
        self.sensor_ema = {k: 0.0 for k in self.sensor_ema.keys()}
        self.threat_history.clear()
        self.threat_timestamps.clear()
        self.alarm_active = False
        self.alarm_start_time = None
        self.history.clear()
