"""
Anomaly Alert ORM Model
"""

import uuid
from datetime import datetime
from sqlalchemy import (Column, String, Float,
                         Boolean, DateTime, TEXT)
from backend.app.database import Base


class AnomalyAlert(Base):
    __tablename__ = "anomaly_alerts"

    id          = Column(TEXT, primary_key=True,
                          default=lambda: str(uuid.uuid4()))
    patient_id  = Column(TEXT, nullable=False)
    modality    = Column(String(10), nullable=False)
    alert_type  = Column(String(50), nullable=False)
    confidence  = Column(Float, nullable=False)
    severity    = Column(String(20), default='medium')
    is_resolved = Column(Boolean, default=False)
    detected_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)