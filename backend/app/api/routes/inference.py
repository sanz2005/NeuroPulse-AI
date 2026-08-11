"""
SNN Inference API Routes for NeuroPulse AI
Runs trained SNN models on biosignal windows.
"""

import sys
import numpy as np
import torch
from typing import List
from fastapi import APIRouter
from pydantic import BaseModel

sys.path.append('.')

from backend.app.config import settings

router = APIRouter()

# ── Load models once at startup ────────────────────────────────────────────────
_models = {}


def load_models():
    """Load all trained SNN models."""
    global _models
    if _models:
        return _models

    try:
        from ai.models.snn.ecg_snn import ECGSNN
        from ai.models.snn.eeg_snn import EEGSNN
        from ai.models.snn.emg_snn import EMGSNN

        device = settings.MODEL_DEVICE

        # ECG SNN
        ecg_model = ECGSNN(input_size=256, hidden1=256,
                            hidden2=128, hidden3=64,
                            n_classes=2, timesteps=64)
        ckpt = torch.load(
            f'{settings.SAVED_MODELS_DIR}/ecg_snn_best.pth',
            map_location=device
        )
        ecg_model.load_state_dict(ckpt['model_state_dict'])
        ecg_model.eval()
        _models['ecg'] = ecg_model

        # EEG SNN
        eeg_model = EEGSNN(n_channels=18, input_size=64,
                            hidden1=128, hidden2=64,
                            hidden3=32, n_classes=2,
                            timesteps=64)
        ckpt = torch.load(
            f'{settings.SAVED_MODELS_DIR}/eeg_snn_best.pth',
            map_location=device
        )
        eeg_model.load_state_dict(ckpt['model_state_dict'])
        eeg_model.eval()
        _models['eeg'] = eeg_model

        # EMG SNN
        emg_model = EMGSNN(n_channels=12, window_size=256,
                            hidden1=64, hidden2=32,
                            n_classes=2, timesteps=32)
        ckpt = torch.load(
            f'{settings.SAVED_MODELS_DIR}/emg_snn_best.pth',
            map_location=device
        )
        emg_model.load_state_dict(ckpt['model_state_dict'])
        emg_model.eval()
        _models['emg'] = emg_model

        print("All SNN models loaded successfully!")

    except Exception as e:
        print(f"Model loading error: {e}")

    return _models


# ── Schemas ────────────────────────────────────────────────────────────────────
class ECGInferenceRequest(BaseModel):
    signal:     List[float]  # 256 samples
    patient_id: str


class EEGInferenceRequest(BaseModel):
    signal:     List[List[float]]  # [18, 256]
    patient_id: str


class EMGInferenceRequest(BaseModel):
    signal:     List[List[float]]  # [12, 256]
    patient_id: str


class InferenceResponse(BaseModel):
    patient_id:   str
    modality:     str
    prediction:   int
    confidence:   float
    label:        str
    total_spikes: float
    sparsity:     float
    is_anomaly:   bool


# ── Routes ─────────────────────────────────────────────────────────────────────
@router.post("/ecg", response_model=InferenceResponse)
async def ecg_inference(request: ECGInferenceRequest):
    """Run ECG-SNN inference on a signal window."""
    models = load_models()

    if 'ecg' not in models:
        return InferenceResponse(
            patient_id=request.patient_id,
            modality='ecg',
            prediction=0,
            confidence=0.5,
            label='unknown',
            total_spikes=0,
            sparsity=0,
            is_anomaly=False
        )

    signal = np.array(request.signal, dtype=np.float32)
    x      = torch.FloatTensor(signal).unsqueeze(0)

    with torch.no_grad():
        logits, spike_record = models['ecg'](x)

    probs      = torch.softmax(logits, dim=1)
    prediction = probs.argmax(dim=1).item()
    confidence = probs[0, prediction].item()

    return InferenceResponse(
        patient_id=request.patient_id,
        modality='ecg',
        prediction=prediction,
        confidence=round(confidence, 4),
        label='arrhythmia' if prediction == 1 else 'normal',
        total_spikes=spike_record['total_spikes'],
        sparsity=round(spike_record['sparsity'], 4),
        is_anomaly=prediction == 1
    )


@router.post("/eeg", response_model=InferenceResponse)
async def eeg_inference(request: EEGInferenceRequest):
    """Run EEG-SNN inference on a signal window."""
    models = load_models()

    if 'eeg' not in models:
        return InferenceResponse(
            patient_id=request.patient_id,
            modality='eeg',
            prediction=0,
            confidence=0.5,
            label='unknown',
            total_spikes=0,
            sparsity=0,
            is_anomaly=False
        )

    signal = np.array(request.signal, dtype=np.float32)
    x      = torch.FloatTensor(signal).unsqueeze(0)

    with torch.no_grad():
        logits, spike_record = models['eeg'](x)

    probs      = torch.softmax(logits, dim=1)
    prediction = probs.argmax(dim=1).item()
    confidence = probs[0, prediction].item()

    return InferenceResponse(
        patient_id=request.patient_id,
        modality='eeg',
        prediction=prediction,
        confidence=round(confidence, 4),
        label='seizure' if prediction == 1 else 'normal',
        total_spikes=spike_record['total_spikes'],
        sparsity=round(spike_record['sparsity'], 4),
        is_anomaly=prediction == 1
    )


@router.post("/emg", response_model=InferenceResponse)
async def emg_inference(request: EMGInferenceRequest):
    """Run EMG-SNN inference on a signal window."""
    models = load_models()

    if 'emg' not in models:
        return InferenceResponse(
            patient_id=request.patient_id,
            modality='emg',
            prediction=0,
            confidence=0.5,
            label='unknown',
            total_spikes=0,
            sparsity=0,
            is_anomaly=False
        )

    signal = np.array(request.signal, dtype=np.float32)
    x      = torch.FloatTensor(signal).unsqueeze(0)

    with torch.no_grad():
        logits, spike_record = models['emg'](x)

    probs      = torch.softmax(logits, dim=1)
    prediction = probs.argmax(dim=1).item()
    confidence = probs[0, prediction].item()

    return InferenceResponse(
        patient_id=request.patient_id,
        modality='emg',
        prediction=prediction,
        confidence=round(confidence, 4),
        label='anomaly' if prediction == 1 else 'normal',
        total_spikes=spike_record['total_spikes'],
        sparsity=round(spike_record['sparsity'], 4),
        is_anomaly=prediction == 1
    )


@router.get("/models/status")
async def models_status():
    """Check which models are loaded."""
    models  = load_models()
    return {
        "ecg_loaded": 'ecg' in models,
        "eeg_loaded": 'eeg' in models,
        "emg_loaded": 'emg' in models,
        "device":     settings.MODEL_DEVICE
    }