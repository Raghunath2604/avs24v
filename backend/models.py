"""
Database models for SentiGuard
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON, Text, LargeBinary
from sqlalchemy.sql import func
from database import Base

class AlarmEvent(Base):
    """Alarm event records"""
    __tablename__ = "alarm_events"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    threat_score = Column(Float, nullable=False)
    status = Column(String(20), nullable=False)  # ALARM, WARNING, OK
    snapshot_url = Column(String(500), nullable=True)
    clip_url = Column(String(500), nullable=True)
    sensor_data = Column(JSON, nullable=True)  # All 50 sensor values
    sms_sent = Column(Boolean, default=False)
    message = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<AlarmEvent {self.id} - {self.status} at {self.timestamp}>"

class SensorReading(Base):
    """Time-series sensor data"""
    __tablename__ = "sensor_readings"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    sensor_name = Column(String(100), nullable=False, index=True)
    value = Column(Float, nullable=False)
    raw_value = Column(Float, nullable=True)
    smoothed_value = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<SensorReading {self.sensor_name}={self.value} at {self.timestamp}>"

class VideoUpload(Base):
    """Uploaded video metadata"""
    __tablename__ = "video_uploads"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(500), nullable=False)
    upload_time = Column(DateTime(timezone=True), server_default=func.now())
    file_size = Column(Integer, nullable=True)  # bytes
    duration = Column(Float, nullable=True)  # seconds
    processed = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<VideoUpload {self.filename}>"

class SystemLog(Base):
    """System event logs"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR
    component = Column(String(50), nullable=False)  # camera, alarm, fusion, etc
    message = Column(Text, nullable=False)
    metadata = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<SystemLog {self.level} - {self.component} at {self.timestamp}>"

class VideoFrame(Base):
    """Stored video frames for analysis/replay"""
    __tablename__ = "video_frames"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    frame_number = Column(Integer, nullable=False)
    frame_data = Column(LargeBinary, nullable=False)  # JPEG compressed
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    source = Column(String(50), nullable=True)  # camera, upload, sample
    threat_score = Column(Float, nullable=True)
    detections = Column(JSON, nullable=True)  # YOLO detections
    
    def __repr__(self):        return f"<VideoFrame {self.frame_number} at {self.timestamp}>"

class AudioChunk(Base):
    """Stored audio chunks for analysis"""
    __tablename__ = "audio_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    chunk_number = Column(Integer, nullable=False)
    audio_data = Column(LargeBinary, nullable=False)  # WAV format
    duration_ms = Column(Integer, nullable=True)
    rms_level = Column(Float, nullable=True)
    dominant_frequency = Column(Float, nullable=True)
    vocal_detected = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<AudioChunk {self.chunk_number} at {self.timestamp}>"
