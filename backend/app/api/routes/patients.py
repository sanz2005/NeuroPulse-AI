"""
Patient API Routes for NeuroPulse AI
CRUD operations for patient management.
"""

import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel

from backend.app.database import get_db
from backend.app.models.patient import Patient

router = APIRouter()


# ── Pydantic Schemas ───────────────────────────────────────────────────────────
class PatientCreate(BaseModel):
    name:       str
    age:        Optional[int] = None
    gender:     Optional[str] = None
    bed_number: Optional[str] = None
    ward:       Optional[str] = None


class PatientResponse(BaseModel):
    id:         str
    name:       str
    age:        Optional[int]
    gender:     Optional[str]
    bed_number: Optional[str]
    ward:       Optional[str]
    status:     str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Routes ─────────────────────────────────────────────────────────────────────
@router.post("/", response_model=PatientResponse)
async def create_patient(
    patient: PatientCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new patient."""
    new_patient = Patient(
        id=str(uuid.uuid4()),
        name=patient.name,
        age=patient.age,
        gender=patient.gender,
        bed_number=patient.bed_number,
        ward=patient.ward,
        status='active'
    )
    db.add(new_patient)
    await db.commit()
    await db.refresh(new_patient)
    return new_patient


@router.get("/", response_model=List[PatientResponse])
async def get_all_patients(
    db: AsyncSession = Depends(get_db)
):
    """Get all active patients."""
    result = await db.execute(
        select(Patient).where(Patient.status == 'active')
    )
    return result.scalars().all()


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific patient by ID."""
    result = await db.execute(
        select(Patient).where(Patient.id == patient_id)
    )
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404,
                            detail="Patient not found")
    return patient


@router.put("/{patient_id}/discharge")
async def discharge_patient(
    patient_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Discharge a patient."""
    await db.execute(
        update(Patient)
        .where(Patient.id == patient_id)
        .values(status='discharged',
                updated_at=datetime.utcnow())
    )
    await db.commit()
    return {"message": "Patient discharged successfully"}


@router.delete("/{patient_id}")
async def delete_patient(
    patient_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a patient record."""
    result = await db.execute(
        select(Patient).where(Patient.id == patient_id)
    )
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404,
                            detail="Patient not found")
    await db.delete(patient)
    await db.commit()
    return {"message": "Patient deleted successfully"}

# ── Patient Cases for Clinical Dashboard ──────────────────────────────────────
PATIENT_CASES = [
    {
        "id": "case-001", "name": "Arjun Mehta",
        "age": 54, "gender": "Male",
        "bed": "ICU-3", "ward": "Cardiac ICU",
        "condition": "Suspected Arrhythmia",
        "ecg_window": 120, "eeg_window": 50, "emg_window": 80,
        "risk_level": "critical",
        "clinical_note": "Patient admitted with chest pain and palpitations."
    },
    {
        "id": "case-002", "name": "Priya Sharma",
        "age": 28, "gender": "Female",
        "bed": "Neuro-2B", "ward": "Neurology Ward",
        "condition": "Epilepsy Monitoring",
        "ecg_window": 10, "eeg_window": 200, "emg_window": 150,
        "risk_level": "high",
        "clinical_note": "Known epileptic patient under continuous EEG monitoring."
    },
    {
        "id": "case-003", "name": "Ravi Kulkarni",
        "age": 67, "gender": "Male",
        "bed": "ICU-7", "ward": "General ICU",
        "condition": "Multi-System Monitoring",
        "ecg_window": 340, "eeg_window": 180, "emg_window": 200,
        "risk_level": "critical",
        "clinical_note": "Post-operative patient with cardiac and neurological concerns."
    },
    {
        "id": "case-004", "name": "Sneha Patil",
        "age": 35, "gender": "Female",
        "bed": "Ward-1A", "ward": "General Ward",
        "condition": "Routine Monitoring",
        "ecg_window": 5, "eeg_window": 20, "emg_window": 30,
        "risk_level": "low",
        "clinical_note": "Post-recovery patient. All vitals stable."
    },
    {
        "id": "case-005", "name": "Deepak Nair",
        "age": 45, "gender": "Male",
        "bed": "Cardiac-4", "ward": "Cardiology",
        "condition": "Cardiac Stress Test",
        "ecg_window": 500, "eeg_window": 30, "emg_window": 400,
        "risk_level": "high",
        "clinical_note": "Stress-induced cardiac monitoring with muscle fatigue."
    },
    {
        "id": "case-006", "name": "Meera Joshi",
        "age": 72, "gender": "Female",
        "bed": "Geriatric-2", "ward": "Geriatrics",
        "condition": "Elderly Multi-Morbidity",
        "ecg_window": 800, "eeg_window": 100, "emg_window": 600,
        "risk_level": "high",
        "clinical_note": "Elderly patient with known cardiac and neurological history."
    },
]


@router.get("/cases")
async def get_patient_cases():
    """Get all predefined patient cases."""
    return PATIENT_CASES


@router.get("/cases/{case_id}")
async def get_patient_case(case_id: str):
    """Get a specific patient case."""
    case = next((c for c in PATIENT_CASES if c["id"] == case_id), None)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case