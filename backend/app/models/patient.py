"""
Patient ORM Model
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.dialects.sqlite import TEXT
from backend.app.database import Base


class Patient(Base):
    __tablename__ = "patients"

    id         = Column(TEXT, primary_key=True,
                         default=lambda: str(uuid.uuid4()))
    name       = Column(String(100), nullable=False)
    age        = Column(Integer)
    gender     = Column(String(10))
    bed_number = Column(String(20))
    ward       = Column(String(50))
    status     = Column(String(20), default='active')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow,
                         onupdate=datetime.utcnow)