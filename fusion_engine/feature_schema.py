# fusion_engine/feature_schema.py

from pydantic import BaseModel
from typing import Optional, List


class VisionData(BaseModel):
    """Structured output from the Vision AI module."""
    object_count: int
    threat_score: float  # 0 to 1
    bounding_boxes: Optional[List[List[int]]] = None  # [[x1,y1,x2,y2], ...]


class AudioData(BaseModel):
    """Structured output from the Audio AI module."""
    volume_db: float
    anomaly_score: float  # 0 to 1
    keywords_detected: Optional[List[str]] = None


class MotionData(BaseModel):
    """Structured output from Motion / Dynamics subsystem."""
    speed: float
    acceleration: float
    jerk: float  # derivative of acceleration


class SensorPacket(BaseModel):
    """Unified structure containing ALL sensor subsystem outputs.
    This is what the Fusion Engine receives every cycle.
    """
    vision: Optional[VisionData] = None
    audio: Optional[AudioData] = None
    motion: Optional[MotionData] = None

    timestamp: float  # epoch seconds
