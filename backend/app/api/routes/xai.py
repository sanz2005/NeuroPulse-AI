"""
XAI API Routes for NeuroPulse AI
Serves spike attribution maps.
"""

import sys
import os
import numpy as np
import torch
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

sys.path.append('.')

router  = APIRouter()
_models = {}


def get_snn_model(modality: str):
    """Load SNN model for XAI."""
    global _models
    if modality in _models:
        return _models[modality]

    try:
        if modality == 'ecg':
            from ai.models.snn.ecg_snn import ECGSNN
            model = ECGSNN(input_size=256, hidden1=256,
                            hidden2=128, hidden3=64,
                            n_classes=2, timesteps=64)
            ckpt_path = 'ai/saved_models/ecg_snn_best.pth'

        elif modality == 'eeg':
            from ai.models.snn.eeg_snn import EEGSNN
            model = EEGSNN(n_channels=18, input_size=64,
                            hidden1=128, hidden2=64,
                            hidden3=32, n_classes=2,
                            timesteps=64)
            ckpt_path = 'ai/saved_models/eeg_snn_best.pth'

        elif modality == 'emg':
            from ai.models.snn.emg_snn import EMGSNN
            model = EMGSNN(n_channels=12, window_size=256,
                            hidden1=64, hidden2=32,
                            n_classes=2, timesteps=32)
            ckpt_path = 'ai/saved_models/emg_snn_best.pth'
        else:
            return None

        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        _models[modality] = model
        return model

    except Exception as e:
        print(f"XAI model load error: {e}")
        return None


class XAIRequest(BaseModel):
    signal:    List
    modality:  str
    patient_id: str


class XAIResponse(BaseModel):
    patient_id:  str
    modality:    str
    attribution: List[float]
    timesteps:   int
    max_region:  int
    explanation: str


@router.post("/attribution", response_model=XAIResponse)
async def get_attribution(request: XAIRequest):
    """Get spike attribution map for a signal window."""
    model = get_snn_model(request.modality)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available for {request.modality}"
        )

    signal = np.array(request.signal, dtype=np.float32)
    x      = torch.FloatTensor(signal).unsqueeze(0)

    with torch.no_grad():
        attribution = model.get_spike_attribution(x)

    attr_np = attribution.cpu().numpy().squeeze()

    # Normalize
    if attr_np.max() > attr_np.min():
        attr_np = ((attr_np - attr_np.min()) /
                   (attr_np.max() - attr_np.min()))

    max_region = int(np.argmax(attr_np))

    # Generate explanation
    modality_map = {
        'ecg': 'ECG arrhythmia',
        'eeg': 'EEG seizure',
        'emg': 'EMG anomaly'
    }
    explanation = (
        f"Peak neural activity detected at timestep "
        f"{max_region} indicating potential "
        f"{modality_map.get(request.modality, 'anomaly')} "
        f"pattern."
    )

    return XAIResponse(
        patient_id=request.patient_id,
        modality=request.modality,
        attribution=attr_np.tolist(),
        timesteps=len(attr_np),
        max_region=max_region,
        explanation=explanation
    )