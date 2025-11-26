# SentiGuard - 50 Virtual Sensors Master System
## Complete Implementation Specification

**Status**: Implementation in Progress  
**Total Sensors**: 50 (20 Vision + 15 Audio + 10 Tracking + 5 System)  
**Current**: 22 implemented  
**Remaining**: 28 to implement

---

## SENSOR CATEGORIES

### 1. Vision-based Sensors (20 total)

| # | Sensor | Status | Implementation |
|---|--------|--------|----------------|
| 1 | motion_score | ✅ | YOLO detection |
| 2 | optical_flow_magnitude | ✅ | Farneback |
| 3 | trajectory_erraticity | ✅ | Tracking variance |
| 4 | loitering_duration | ✅ | Centroid tracking |
| 5 | crowd_density | ✅ | Person count |
| 6 | posture_risk_score | ✅ | MediaPipe pose |
| 7 | **fall_detect_score** | 🔨 NEW | Pose angle analysis |
| 8 | **running_score** | 🔨 NEW | Fast forward motion |
| 9 | fight_interaction_score | ✅ | Multi-person proximity |
| 10 | aggressive_pose_score | ✅ | Raised arm detection |
| 11 | weapon_score | ✅ | YOLO weapon model |
| 12 | object_sharpness_score | ✅ | Edge detection |
| 13 | tampering_score | ✅ | Brightness/obstruction |
| 14 | camera_health_score | ✅ | FPS monitoring |
| 15 | lighting_quality | ✅ | Brightness histogram |
| 16 | **visibility_score** | 🔨 NEW | Scene clarity |
| 17 | environment_intrusion_score | ✅ | Geo-zones |
| 18 | **interaction_object_score** | 🔨 NEW | Touching equipment |
| 19 | panic_crowd_movement_score | ✅ | Rapid dispersion |
| 20 | **object_drop_or_pick_score** | 🔨 NEW | Object interaction |

### 2. Audio-based Sensors (15 total)

| # | Sensor | Status | Implementation |
|---|--------|--------|----------------|
| 21 | vocal_scream_score | ✅ | FFT + RMS |
| 22 | **aggressive_voice_score** | 🔨 NEW | Tone analysis |
| 23 | metal_impact_audio_score | ✅ | 1-4kHz detection |
| 24 | glass_break_audio_score | ✅ | 4-8kHz detection |
| 25 | **gunshot_like_score** | 🔨 NEW | Impulse detection |
| 26 | **high_pitch_distress_score** | 🔨 NEW | >3kHz sustained |
| 27 | **low_frequency_thud_score** | 🔨 NEW | <500Hz impact |
| 28 | ambient_noise_level | ✅ | RMS normalization |
| 29 | **speech_activity_rate** | 🔨 NEW | Voice detection rate |
| 30 | vocal_event_count | ✅ | Event counting |
| 31 | **audio_direction_confidence** | 🔨 NEW | Multi-mic (optional) |
| 32 | **background_sound_class** | 🔨 NEW | Vehicle/machinery |
| 33 | **silence_anomaly_score** | 🔨 NEW | Too quiet |
| 34 | **wind_noise_score** | 🔨 NEW | Wind detection |
| 35 | **audio_spectral_anomaly_score** | 🔨 NEW | Unusual spectrum |

### 3. Tracking / Behavior Sensors (10 total)

| # | Sensor | Status | Implementation |
|---|--------|--------|----------------|
| 36 | **person_entry_count** | 🔨 NEW | Zone crossing |
| 37 | **person_exit_count** | 🔨 NEW | Zone crossing |
| 38 | **group_formation_score** | 🔨 NEW | Cluster detection |
| 39 | **approach_towards_camera_score** | 🔨 NEW | Z-axis movement |
| 40 | **following_behavior_score** | 🔨 NEW | Trailing detection |
| 41 | body_velocity_spike_score | ✅ | Acceleration |
| 42 | **repeat_intruder_score** | 🔨 NEW | Re-ID fingerprint |
| 43 | **vehicle_presence_score** | 🔨 NEW | Car/truck detection |
| 44 | **vehicle_speed_estimate** | 🔨 NEW | Motion estimation |
| 45 | **person_identity_hash** | 🔨 NEW | Anonymous tracking |

### 4. System / Environmental Sensors (5 total)

| # | Sensor | Status | Implementation |
|---|--------|--------|----------------|
| 46 | **device_cpu_temp** | 🔨 NEW | psutil |
| 47 | **device_ram_usage** | 🔨 NEW | psutil |
| 48 | **device_storage_usage** | 🔨 NEW | psutil |
| 49 | **network_quality_score** | 🔨 NEW | Latency test |
| 50 | **time_of_day_context** | 🔨 NEW | Day/night weight |

---

## ENHANCED FUSION FORMULA

### Clustering Approach

```python
# Behavior cluster
behavior_cluster_score = mean([
    trajectory_erraticity,
    loitering_duration_norm,  # normalized to [0,1]
    aggressive_pose_score,
    fight_interaction_score
])

# Environment cluster
environment_risk_score = mean([
    1.0 - lighting_quality,  # penalty
    crowd_density,
    ambient_noise_level
])

# Base fusion (before weapon boost)
base_threat = (
    0.35 * smooth(motion_score) +
    0.35 * smooth(vocal_scream_score) +
    0.20 * behavior_cluster_score +
    0.10 * environment_risk_score
)

# Weapon boost (non-linear)
weapon_smooth = smooth(weapon_score)
if weapon_smooth > 0.60:
    boosted = min(1.0, base_threat + (0.4 * weapon_smooth))
else:
    boosted = base_threat

# Final threat with higher alpha for responsiveness
final_threat = EMA(boosted, alpha=0.6)

# Decision
if final_threat >= 0.75:
    alarm = True
elif final_threat >= 0.60:
    soft_alert = True
else:
    safe = True
```

---

## DATABASE SCHEMA (PostgreSQL)

```python
# models.py
from sqlalchemy import Column, Integer, Float, Boolean, String, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Event(Base):
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True, index=True)
    site_id = Column(Integer, ForeignKey("sites.id"), index=True)
    device_id = Column(Integer, ForeignKey("devices.id"), index=True)
    
    # Timestamp
    timestamp = Column(Float, index=True)
    
    # Threat
    threat_score = Column(Float, index=True)
    alarm = Column(Boolean, default=False, index=True)
    soft_alert = Column(Boolean, default=False)
    
    # Location
    camera_zone = Column(String(50), nullable=True, index=True)
    x_norm = Column(Float, default=0.0)
    y_norm = Column(Float, default=0.0)
    
    # Files
    snapshot_path = Column(String(255), nullable=True)
    clip_path = Column(String(255), nullable=True)
    
    # ALL 50 SENSORS stored as JSON
    sensors = Column(JSON, nullable=False)
    
    # Clusters (pre-computed for fast queries)
    behavior_cluster = Column(Float)
    environment_cluster = Column(Float)
    weapon_boost_applied = Column(Boolean, default=False)

class Site(Base):
    __tablename__ = "sites"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    location_lat = Column(Float)
    location_lon = Column(Float)
    timezone = Column(String(50))

class Device(Base):
    __tablename__ = "devices"
    
    id = Column(Integer, primary_key=True)
    site_id = Column(Integer, ForeignKey("sites.id"))
    device_name = Column(String(100))
    camera_zone = Column(String(50))
    status = Column(String(20))  # active, offline, maintenance
```

### Indexes

```sql
CREATE INDEX idx_event_siteid ON events(site_id);
CREATE INDEX idx_event_deviceid ON events(device_id);
CREATE INDEX idx_event_timestamp ON events(timestamp);
CREATE INDEX idx_event_threat ON events(threat_score);
CREATE INDEX idx_event_alarm ON events(alarm);
CREATE INDEX idx_event_zone ON events(camera_zone);

-- JSON GIN index for fast sensor queries
CREATE INDEX idx_event_sensors_gin ON events USING GIN (sensors);
```

---

## API SCHEMA (Pydantic)

```python
# schemas.py
from pydantic import BaseModel
from typing import Optional, Dict

class EventLocation(BaseModel):
    camera_zone: str
    x_norm: float
    y_norm: float

class EventIn(BaseModel):
    site_id: int
    device_id: int
    timestamp: float
    
    threat_score: float
    alarm: bool
    soft_alert: bool
    
    event_location: EventLocation
    virtual_sensors: Dict[str, float]  # All 50 sensors
    
    behavior_cluster: float
    environment_cluster: float
    weapon_boost_applied: bool
    
    snapshot_b64: Optional[str] = None
    clip_b64: Optional[str] = None
    
    model_versions: Dict[str, str]

class EventOut(BaseModel):
    id: int
    site_id: int
    device_id: int
    timestamp: float
    
    threat_score: float
    alarm: bool
    soft_alert: bool
    
    camera_zone: str
    x_norm: float
    y_norm: float
    
    snapshot_url: Optional[str]
    clip_url: Optional[str]
    
    sensors: Dict[str, float]
    behavior_cluster: float
    environment_cluster: float
    
    class Config:
        orm_mode = True
```

---

## COMPLETE API ENDPOINTS

```python
# main.py (FastAPI routes)

@app.post("/api/v1/events/ingest", response_model=dict)
async def ingest_event(event: EventIn, db: Session = Depends(get_db)):
    """Ingest event from edge agent"""
    
    # Save snapshot to MinIO
    snapshot_url = None
    if event.snapshot_b64:
        snapshot_bytes = base64.b64decode(event.snapshot_b64)
        snapshot_url = await save_to_minio(
            snapshot_bytes,
            f"snapshots/site{event.site_id}_device{event.device_id}/{event.timestamp}.jpg"
        )
    
    # Save audio clip
    clip_url = None
    if event.clip_b64:
        clip_bytes = base64.b64decode(event.clip_b64)
        clip_url = await save_to_minio(
            clip_bytes,
            f"audio/site{event.site_id}_device{event.device_id}/{event.timestamp}.wav"
        )
    
    # Create event
    db_event = Event(
        site_id=event.site_id,
        device_id=event.device_id,
        timestamp=event.timestamp,
        threat_score=event.threat_score,
        alarm=event.alarm,
        soft_alert=event.soft_alert,
        camera_zone=event.event_location.camera_zone,
        x_norm=event.event_location.x_norm,
        y_norm=event.event_location.y_norm,
        snapshot_path=snapshot_url,
        clip_path=clip_url,
        sensors=event.virtual_sensors,
        behavior_cluster=event.behavior_cluster,
        environment_cluster=event.environment_cluster,
        weapon_boost_applied=event.weapon_boost_applied
    )
    
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    
    # Broadcast via WebSocket
    await ws_manager.broadcast({
        "type": "event",
        "payload": {
            "id": db_event.id,
            **event.dict(),
            "snapshot_url": snapshot_url,
            "clip_url": clip_url
        }
    })
    
    return {"status": "ok", "event_id": db_event.id}


@app.get("/api/v1/events", response_model=List[EventOut])
def list_events(
    limit: int = 50,
    site_id: Optional[int] = None,
    alarm_only: bool = False,
    db: Session = Depends(get_db)
):
    """List events with filtering"""
    query = db.query(Event)
    
    if site_id:
        query = query.filter(Event.site_id == site_id)
    if alarm_only:
        query = query.filter(Event.alarm == True)
    
    events = query.order_by(Event.timestamp.desc()).limit(limit).all()
    return events


@app.get("/api/v1/events/{event_id}", response_model=EventOut)
def get_event(event_id: int, db: Session = Depends(get_db)):
    """Get single event by ID"""
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@app.get("/api/v1/sites/{site_id}/events", response_model=List[EventOut])
def get_site_events(site_id: int, limit: int = 100, db: Session = Depends(get_db)):
    """Get all events for a site"""
    events = db.query(Event).filter(
        Event.site_id == site_id
    ).order_by(Event.timestamp.desc()).limit(limit).all()
    return events


@app.get("/api/v1/health")
def health_check():
    """System health check"""
    return {
        "backend": "ok",
        "database": check_database_connection(),
        "minio": check_minio_connection(),
        "websocket_clients": ws_manager.client_count()
    }


@app.get("/api/v1/config")
def get_config():
    """Get fusion configuration"""
    return fusion_engine.get_config()


@app.post("/api/v1/config")
def update_config(config: dict):
    """Update fusion configuration"""
    fusion_engine.update_config(**config)
    return {"status": "ok", "config": fusion_engine.get_config()}
```

---

## IMPLEMENTATION PRIORITY

### Phase 1: Critical Sensors (Week 1)
Priority sensors to add first:

1. **fall_detect_score** - Safety critical
2. **running_score** - Escape/chase detection  
3. **gunshot_like_score** - Emergency
4. **speech_activity_rate** - Distress
5. **person_entry/exit_count** - Tracking

### Phase 2: Enhanced Audio (Week 2)
6. **aggressive_voice_score**
7. **high_pitch_distress_score**
8. **low_frequency_thud_score**
9. **silence_anomaly_score**
10. **wind_noise_score**

### Phase 3: Tracking (Week 3)
11. **group_formation_score**
12. **approach_towards_camera_score**
13. **following_behavior_score**
14. **vehicle_presence_score**
15. **vehicle_speed_estimate**

### Phase 4: System & Polish (Week 4)
16. **device_cpu_temp / ram / storage**
17. **network_quality_score**
18. **time_of_day_context**
19. **visibility_score**
20. **interaction_object_score**

---

## FILES TO CREATE/UPDATE

### New Files
- `models.py` - SQLAlchemy models
- `schemas.py` - Pydantic schemas
- `database.py` - Database connection
- `minio_client.py` - MinIO integration
- `websocket_manager.py` - WebSocket handling

### Update Files
- `virtual_sensors.py` - Add 28 new sensors
- `fusion.py` - New clustering formula
- `main.py` - Complete API overhaul
- `requirements.txt` - Add sqlalchemy, psycopg2, minio, psutil

---

**Document Status**: Specification Complete  
**Next**: Implement critical sensors first  
**Target**: Production-grade 50-sensor system
