# SentiGuard - Advanced Features Implementation Guide

## Current Status ✅

### Implemented Features (v2.0)

#### ✅ Core Virtual Sensors (12)
1. motion_score
2. optical_flow_magnitude
3. loitering_duration  
4. trajectory_erraticity
5. crowd_density
6. posture_risk_score
7. vocal_scream_score
8. vocal_event_count
9. ambient_noise_level
10. camera_health
11. lighting_quality
12. environment_meta

#### ✅ System Features
- **Alarm Sound System**: Browser-generated siren with alternating frequencies
- **Voice Alerts**: Text-to-speech notifications
- **Visual Flash Alerts**: Screen flashing during alarms
- **Location Tracking**: GPS coordinates, site name, address in every event
- **Event History**: Last 10 events with timestamps
- **Real-time WebSocket**: Live streaming of all sensors
- **Comprehensive Event Payload**: Full audit trail with model versions

---

## Planned Enhancements (v3.0)

### 10 NEW Virtual Sensors (Enterprise-Level)

#### 21. weapon_score [0..1]
**Purpose**: Dedicated ONNX weapon detection model  
**Implementation**:
```python
def compute_weapon_score(self, detections):
    # Enhanced weapon detection with confidence scoring
    weapon_classes = ['knife', 'scissors', 'gun', 'pistol', 'bat', 'crowbar']
    weapon_detections = [d for d in detections if d['class'] in weapon_classes]
    
    if not weapon_detections:
        return 0.0
    
    # Maximum confidence among weapons
    max_score = max(d['confidence'] for d in weapon_detections)
    
    # Boost threat if multiple weapons
    if len(weapon_detections) > 1:
        max_score = min(1.0, max_score * 1.2)
    
    return float(max_score)
```

**Integration**:
- If `weapon_score > 0.60`: `threat_score += 0.3 * weapon_score`
- Auto-trigger: Alert sound + snapshot + operator escalation

#### 22. object_sharpness_score [0..1]
**Purpose**: Detect sharp-edged objects via contour analysis  
**Implementation**:
```python
def compute_object_sharpness(self, frame, detections):
    max_sharpness = 0.0
    
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['box'])
        roi = frame[y1:y2, x1:x2]
        
        # Edge detection
        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.sum(edges) / (roi.shape[0] * roi.shape[1])
        
        # Detect sharp corners
        corners = cv2.goodFeaturesToTrack(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 25, 0.01, 10)
        sharpness = min(1.0, edge_density * 0.5 + (len(corners) if corners is not None else 0) * 0.02)
        
        max_sharpness = max(max_sharpness, sharpness)
    
    return float(max_sharpness)
```

#### 23. body_velocity_spike_score [0..1]
**Purpose**: Detect sudden acceleration (attacks/fights)  
**Implementation**:
```python
def compute_body_velocity_spike(self):
    max_spike = 0.0
    
    for obj_id, obj_data in self.tracked_objects.items():
        positions = obj_data['positions']
        
        if len(positions) < 3:
            continue
        
        # Compute velocities
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            vel = np.sqrt(dx**2 + dy**2)
            velocities.append(vel)
        
        # Detect spikes (sudden acceleration)
        if len(velocities) >= 2:
            accelerations = [velocities[i] - velocities[i-1] for i in range(1, len(velocities))]
            max_accel = max(accelerations) if accelerations else 0
            spike_score = min(1.0, max_accel / 20.0)  # Normalize
            max_spike = max(max_spike, spike_score)
    
    return float(max_spike)
```

#### 24. aggressive_pose_score [0..1]
**Purpose**: Detect attacking posture (raised arm, lunge forward)  
**Implementation**:
```python
def compute_aggressive_pose(self, frame):
    if not self.pose_detector:
        return 0.0
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = self.pose_detector.process(rgb_frame)
    
    if not results.pose_landmarks:
        return 0.0
    
    landmarks = results.pose_landmarks.landmark
    
    # Detect raised arm + forward lean
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]
    nose = landmarks[0]
    left_shoulder = landmarks[11]
    
    # Check for punching pose (one arm extended forward)
    wrist_forward = left_wrist.z < nose.z - 0.2 or right_wrist.z < nose.z - 0.2
    
    # Check for raised fist
    arm_raised = left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y
    
    if wrist_forward and arm_raised:
        return 0.9  # High aggression
    elif wrist_forward or arm_raised:
        return 0.5  # Medium aggression
    
    return 0.0
```

#### 25. fight_interaction_score [0..1]
**Purpose**: Detect two+ persons in aggressive close-range interaction  
**Implementation**:
```python
def compute_fight_interaction(self):
    if len(self.tracked_objects) < 2:
        return 0.0
    
    objects = list(self.tracked_objects.values())
    max_interaction = 0.0
    
    for i in range(len(objects)):
        for j in range(i+1, len(objects)):
            # Compute distance between persons
            pos1 = objects[i]['centroid']
            pos2 = objects[j]['centroid']
            dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            
            # Check if very close (< 50 pixels)
            if dist < 50:
                # Check if both moving erratically
                vel1 = self._get_velocity(objects[i])
                vel2 = self._get_velocity(objects[j])
                
                if vel1 > 5 and vel2 > 5:  # Both moving fast
                    interaction_score = min(1.0, (50 - dist) / 50 * (vel1 + vel2) / 20)
                    max_interaction = max(max_interaction, interaction_score)
    
    return float(max_interaction)
```

#### 26. panic_crowd_movement_score [0..1]
**Purpose**: Detect irregular, fast crowd dispersion  
**Implementation**:
```python
def compute_panic_crowd_movement(self):
    if len(self.tracked_objects) < 3:  # Need at least 3 people
        return 0.0
    
    # Compute average velocity
    velocities = []
    for obj_data in self.tracked_objects.values():
        if len(obj_data['positions']) >= 2:
            p1, p2 = obj_data['positions'][-2:]
            vel = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            velocities.append(vel)
    
    if not velocities:
        return 0.0
    
    avg_vel = np.mean(velocities)
    vel_variance = np.var(velocities)
    
    # High variance + high average velocity = panic
    panic_score = min(1.0, (avg_vel / 10) * (vel_variance / 100))
    
    return float(panic_score)
```

#### 27. glass_break_audio_score [0..1]
**Purpose**: Detect glass breaking spectral signature  
**Implementation**:
```python
def compute_glass_break_audio(self, audio_fft, audio_rms):
    # Glass break has distinct high-frequency signature (4-8 kHz)
    if len(audio_fft) < 100:
        return 0.0
    
    # Analyze high-frequency content
    high_freq_start = int(len(audio_fft) * 0.4)  # ~4kHz assuming 44.1kHz sample rate
    high_freq_end = int(len(audio_fft) * 0.8)    # ~8kHz
    
    high_freq_energy = np.sum(np.abs(audio_fft[high_freq_start:high_freq_end]))
    total_energy = np.sum(np.abs(audio_fft))
    
    if total_energy == 0:
        return 0.0
    
    # Glass break: high frequency ratio + loud RMS
    high_freq_ratio = high_freq_energy / total_energy
    
    score = 0.0
    if high_freq_ratio > 0.3 and audio_rms > 5.0:
        score = min(1.0, high_freq_ratio * audio_rms / 20)
    
    return float(score)
```

#### 28. metal_impact_audio_score [0..1]
**Purpose**: Detect metal-to-metal sharp sounds  
**Implementation**:
```python
def compute_metal_impact_audio(self, audio_fft, audio_rms):
    # Metal impact: sharp transient + mid-high frequencies (1-4 kHz)
    if len(audio_fft) < 100:
        return 0.0
    
    mid_high_start = int(len(audio_fft) * 0.2)  # ~1kHz
    mid_high_end = int(len(audio_fft) * 0.4)    # ~4kHz
    
    mid_high_energy = np.sum(np.abs(audio_fft[mid_high_start:mid_high_end]))
    total_energy = np.sum(np.abs(audio_fft))
    
    if total_energy == 0:
        return 0.0
    
    # Metal: concentrated mid-high frequency + loud + sharp onset
    freq_ratio = mid_high_energy / total_energy
    
    score = 0.0
    if freq_ratio > 0.25 and audio_rms > 8.0:
        score = min(1.0, freq_ratio * audio_rms / 25)
    
    return float(score)
```

#### 29. environment_intrusion_score [0..1]
**Purpose**: Detect person crossing restricted boundary  
**Implementation**:
```python
def compute_environment_intrusion(self):
    max_intrusion = 0.0
    
    for obj_data in self.tracked_objects.values():
        centroid = obj_data['centroid']
        
        # Normalize to [0,1]
        norm_x = centroid[0] / 640  # Assumes 640px width
        norm_y = centroid[1] / 480  # Assumes 480px height
        
        # Check against geo-zones
        for zone in self.geo_zones:
            if self._point_in_polygon((norm_x, norm_y), zone['polygon']):
                max_intrusion = 1.0  # Boolean: inside restricted zone
                break
    
    return float(max_intrusion)

def _point_in_polygon(self, point, polygon):
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
```

#### 30. tampering_score [0..1]
**Purpose**: Detect camera blocking or movement  
**Implementation**:
```python
def compute_tampering_score(self, frame):
    # Compute frame brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    # Initialize
    if self.prev_brightness is None:
        self.prev_brightness = brightness
        return 0.0
    
    # Detect sudden brightness drop (covering camera)
    brightness_drop = self.prev_brightness - brightness
    
    tampering = 0.0
    
    # Camera covered (very dark)
    if brightness < 10:
        tampering = 1.0
    # Sudden drop
    elif brightness_drop > 50:
        tampering = 0.8
    # Uniform color (blocked by object)
    else:
        std_dev = np.std(gray)
        if std_dev < 5:  # Very low variance = uniform
            tampering = 0.7
    
    self.prev_brightness = brightness
    return float(tampering)
```

---

## 10 NEW System Features

### Feature 1: Multi-Camera Weapon Correlation
**Status**: Planned  
**Description**: Cross-camera threat correlation  
**Implementation**:
- Backend aggregates detections from multiple cameras
- If weapon detected on Camera A and person tracked to Camera B → boost threat
- Requires: DeepSORT ID tracking across cameras

### Feature 2: Anonymous Face Blur
**Status**: Planned  
**Description**: Privacy-preserving snapshots  
**Implementation**:
```python
import cv2

def blur_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
        frame[y:y+h, x:x+w] = blurred
    
    return frame
```

### Feature 3: Auto-Incident Timeline Generation
**Status**: Partially Implemented (Event History)  
**Enhancement**:
```python
class IncidentTimeline:
    def __init__(self):
        self.incidents = []
        self.current_incident = None
    
    def process_event(self, event):
        if event['status'] in ['ALARM', 'WARNING']:
            if self.current_incident is None:
                # Start new incident
                self.current_incident = {
                    'start_time': event['timestamp'],
                    'peak_threat': event['threat_score'],
                    'events': [event],
                    'snapshots': []
                }
            else:
                # Continue incident
                self.current_incident['events'].append(event)
                self.current_incident['peak_threat'] = max(
                    self.current_incident['peak_threat'],
                    event['threat_score']
                )
        elif self.current_incident and event['status'] == 'SAFE':
            # End incident
            self.current_incident['end_time'] = event['timestamp']
            self.current_incident['duration'] = (
                self.current_incident['end_time'] - 
                self.current_incident['start_time']
            )
            self.incidents.append(self.current_incident)
            self.current_incident = None

    def generate_summary(self, incident):
        start = datetime.fromtimestamp(incident['start_time'])
        duration = incident['duration']
        peak = incident['peak_threat']
        
        summary = f"""
        Incident Report
        ===============
        Start Time: {start.strftime('%Y-%m-%d %H:%M:%S')}
        Duration: {duration:.1f} seconds
        Peak Threat: {peak:.2f}
        Events: {len(incident['events'])}
        
        Timeline:
        """
        
        for e in incident['events']:
            t = datetime.fromtimestamp(e['timestamp'])
            summary += f"\n  {t.strOpt:%H:%M:%S} - {e['status']} ({e['threat_score']:.2f})"
        
        return summary
```

### Feature 4: Regional Alarm Policies
**Status**: Configurable via fusion weights  
**Enhancement**: Zone-based thresholds
```python
ZONE_POLICIES = {
    'gate': {'alarm_threshold': 0.65, 'sensitivity': 'HIGH'},
    'storage': {'alarm_threshold': 0.75, 'sensitivity': 'MEDIUM'},
    'public': {'alarm_threshold': 0.85, 'sensitivity': 'LOW'}
}

def get_zone_policy(location):
    # Determine zone from location coordinates
    zone = determine_zone(location)
    return ZONE_POLICIES.get(zone, {'alarm_threshold': 0.75})
```

### Feature 5: Geo-Fence Violation Alerts
**Status**: Implemented in sensor #29  
**UI Enhancement**: Map-based zone editor
```javascript
// Frontend: Draw zones on camera feed overlay
function drawZones(canvas, zones) {
    const ctx = canvas.getContext('2d');
    zones.forEach(zone => {
        ctx.beginPath();
        zone.polygon.forEach((point, i) => {
            if (i === 0) ctx.moveTo(point[0] * canvas.width, point[1] * canvas.height);
            else ctx.lineTo(point[0] * canvas.width, point[1] * canvas.height);
        });
        ctx.closePath();
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.stroke();
    });
}
```

### Feature 6: Offline Panic Button
**Status**: Planned  
**Implementation**:
```python
class OfflinePanicButton:
    def __init__(self):
        self.offline_queue = deque(maxlen=100)
        self.is_online = True
    
    def trigger_panic(self):
        event = {
            'type': 'PANIC_BUTTON',
            'timestamp': time.time(),
            'priority': 'HIGHEST'
        }
        
        # Local alarm
        play_local_siren()
        
        # Try to send
        try:
            send_to_backend(event)
        except ConnectionError:
            self.offline_queue.append(event)
            save_local(event)
    
    def sync_offline_events(self):
        while self.offline_queue:
            event = self.offensive_queue.popleft()
            try:
                send_to_backend(event)
            except ConnectionError:
                self.offline_queue.appendleft(event)
                break
```

### Feature 7: Thermal + Visual Fusion (Virtual)
**Status**: Planned  
**Implementation**: AI night enhancement
```python
def enhance_low_light(frame):
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge and convert back
    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced
```

### Feature 8: Two-Stage Alert Escalation
**Status**: Implemented (SAFE/WARNING/ALARM)  
**Enhancement**: Operator confirmation
```python
class TwoStageAlert:
    def __init__(self):
        self.soft_alerts = []
        self.hard_alerts = []
    
    def process_threat(self, threat_score):
        if 0.60 <= threat_score < 0.75:
            # Soft alert → operator
            self.soft_alerts.append({
                'time': time.time(),
                'score': threat_score,
                'action': 'OPERATOR_REVIEW'
            })
            notify_operator_dashboard()
        
        elif threat_score >= 0.75:
            # Hard alert → automatic
            self.hard_alerts.append({
                'time': time.time(),
                'score': threat_score,
                'action': 'AUTO_SIREN_SMS'
            })
            trigger_siren()
            send_sms_gateway()
```

### Feature 9: Device Health Dashboard
**Status**: Partially Implemented  
**Enhancement**: Full metrics
```python
import psutil

def get_device_health():
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'cpu_temp': get_cpu_temperature(),  # Platform-specific
        'network_latency': ping_backend(),
        'camera_fps': virtual_sensors.compute_camera_health()['fps'],
        'uptime': time.time() - virtual_sensors.start_time
    }
```

### Feature 10: Behavior Replay (AI Summary)
**Status**: Planned  
**Implementation**: GPT-based summarization
```python
def generate_incident_summary(incident):
    events = incident['events']
    
    # Extract key facts
    location = events[0]['site_name']
    start_time = datetime.fromtimestamp(events[0]['timestamp'])
    duration = incident['duration']
    peak_threat = incident['peak_threat']
    has_weapon = any(e.get('meta', {}).get('has_weapon') for e in events)
    loitering = max(e.get('loitering_duration', 0) for e in events)
    
    # Generate natural language summary
    summary = f"""
    Incident Report - {location}
    
    Summary:
    At {start_time.strftime('%I:%M %p')}, suspicious activity was detected at {location}.
    """
    
    if has_weapon:
        summary += "A weapon-like object was identified. "
    
    if loitering > 10:
        summary += f"The individual loitered for {loitering:.0f} seconds. "
    
    summary += f"Threat level escalated to {'HIGH' if peak_threat > 0.8 else 'MEDIUM'}."
    summary += f"\n\nDuration: {duration:.0f} seconds"
    summary += f"\nPeak Threat Score: {peak_threat:.2f}"
    
    return summary
```

---

## Implementation Priority

### Phase 1 (Immediate - Week 1)
- ✅ Alarm sound system
- ✅ Location tracking
- ✅ Event history
- ✅ Enhanced UI

### Phase 2 (Short-term - Week 2-3)
- [ ] Weapon score enhancement
- [ ] Body velocity spike
- [ ] Tampering detection
- [ ] Glass break / metal impact audio
- [ ] Two-stage escalation

### Phase 3 (Medium-term - Month 1)
- [ ] Aggressive pose
- [ ] Fight interaction
- [ ] Panic crowd movement
- [ ] Geo-fence violations
- [ ] Incident timeline

### Phase 4 (Long-term - Month 2-3)
- [ ] Multi-camera correlation
- [ ] Face anonymization
- [ ] Behavior replay
- [ ] Full device health dashboard
- [ ] Offline resilience

---

## Configuration

All new sensors can be configured:

```python
# In fusion.py - Add weights for new sensors
self.weapon_weight = 0.15  # Dedic ated weapon detection
self.body_velocity_spike_weight = 0.05
self.aggressive_pose_weight = 0.08
self.fight_interaction_weight = 0.10
self.panic_crowd_weight = 0.05
self.glass_break_weight = 0.07
self.metal_impact_weight = 0.07
self.intrusion_weight = 0.10
self.tampering_weight = 0.15
self.object_sharpness_weight = 0.05
```

---

## Testing Procedures

Each new sensor should be tested:

1. **Unit Test**: Isolated sensor computation
2. **Integration Test**: Sensor in fusion pipeline
3. **Live Test**: Real-world scenarios
4. **Performance Test**: FPS impact measurement

---

**Document Version**: 3.0-DRAFT  
**Last Updated**: 2025-11-26  
**Status**: Design Complete, Implementation in Progress
