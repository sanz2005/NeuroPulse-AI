"""
Anomaly Alert API Routes for NeuroPulse AI
"""

import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel

from backend.app.database import get_db
from backend.app.models.alert import AnomalyAlert

router = APIRouter()


class AlertCreate(BaseModel):
    patient_id: str
    modality:   str
    alert_type: str
    confidence: float
    severity:   Optional[str] = 'medium'


class AlertResponse(BaseModel):
    id:          str
    patient_id:  str
    modality:    str
    alert_type:  str
    confidence:  float
    severity:    str
    is_resolved: bool
    detected_at: datetime

    class Config:
        from_attributes = True


@router.post("/", response_model=AlertResponse)
async def create_alert(
    alert: AlertCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new anomaly alert."""
    # Determine severity from confidence
    if alert.confidence >= 0.9:
        severity = 'critical'
    elif alert.confidence >= 0.75:
        severity = 'high'
    elif alert.confidence >= 0.6:
        severity = 'medium'
    else:
        severity = 'low'

    new_alert = AnomalyAlert(
        id=str(uuid.uuid4()),
        patient_id=alert.patient_id,
        modality=alert.modality,
        alert_type=alert.alert_type,
        confidence=alert.confidence,
        severity=severity,
        is_resolved=False
    )
    db.add(new_alert)
    await db.commit()
    await db.refresh(new_alert)
    return new_alert


@router.get("/", response_model=List[AlertResponse])
async def get_all_alerts(
    db: AsyncSession = Depends(get_db)
):
    """Get all unresolved alerts."""
    result = await db.execute(
        select(AnomalyAlert)
        .where(AnomalyAlert.is_resolved == False)
        .order_by(AnomalyAlert.detected_at.desc())
    )
    return result.scalars().all()


@router.get("/{patient_id}",
            response_model=List[AlertResponse])
async def get_patient_alerts(
    patient_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get all alerts for a specific patient."""
    result = await db.execute(
        select(AnomalyAlert)
        .where(AnomalyAlert.patient_id == patient_id)
        .order_by(AnomalyAlert.detected_at.desc())
    )
    return result.scalars().all()


@router.put("/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Mark an alert as resolved."""
    await db.execute(
        update(AnomalyAlert)
        .where(AnomalyAlert.id == alert_id)
        .values(is_resolved=True,
                resolved_at=datetime.utcnow())
    )
    await db.commit()
    return {"message": "Alert resolved successfully"}