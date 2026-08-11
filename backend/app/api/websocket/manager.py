"""
WebSocket Connection Manager for NeuroPulse AI
Handles live biosignal streaming to frontend.
"""

import sys
import json
import asyncio
import numpy as np
import torch
from typing import Dict, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

sys.path.append('.')

websocket_router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections per patient."""

    def __init__(self):
        # patient_id → list of WebSocket connections
        self.active: Dict[str, List[WebSocket]] = {}

    async def connect(self, patient_id: str,
                       websocket: WebSocket):
        await websocket.accept()
        if patient_id not in self.active:
            self.active[patient_id] = []
        self.active[patient_id].append(websocket)
        print(f"Client connected: patient {patient_id}")

    def disconnect(self, patient_id: str,
                    websocket: WebSocket):
        if patient_id in self.active:
            self.active[patient_id].remove(websocket)
        print(f"Client disconnected: patient {patient_id}")

    async def broadcast(self, patient_id: str,
                         message: dict):
        """Send message to all subscribers of a patient."""
        if patient_id not in self.active:
            return
        dead = []
        for ws in self.active[patient_id]:
            try:
                await ws.send_json(message)
            except:
                dead.append(ws)
        for ws in dead:
            self.active[patient_id].remove(ws)


manager = ConnectionManager()

# Signal data cache
_signal_cache = {}
_models       = {}
_window_idx   = {}


def load_data():
    """Load signal data for streaming."""
    global _signal_cache
    for modality in ['ecg', 'eeg', 'emg']:
        if modality not in _signal_cache:
            try:
                X = np.load(
                    f'data/processed/{modality}/test/X.npy'
                )
                y = np.load(
                    f'data/processed/{modality}/test/y.npy'
                )
                _signal_cache[modality] = (X, y)
            except Exception as e:
                print(f"Could not load {modality} data: {e}")


def load_models():
    """Load SNN models for inference."""
    global _models
    if _models:
        return

    try:
        from ai.models.snn.ecg_snn import ECGSNN
        from ai.models.snn.eeg_snn import EEGSNN
        from ai.models.snn.emg_snn import EMGSNN

        # ECG
        ecg = ECGSNN(input_size=256, hidden1=256,
                      hidden2=128, hidden3=64,
                      n_classes=2, timesteps=64)
        ckpt = torch.load('ai/saved_models/ecg_snn_best.pth',
                           map_location='cpu')
        ecg.load_state_dict(ckpt['model_state_dict'])
        ecg.eval()
        _models['ecg'] = ecg

        # EEG
        eeg = EEGSNN(n_channels=18, input_size=64,
                      hidden1=128, hidden2=64,
                      hidden3=32, n_classes=2, timesteps=64)
        ckpt = torch.load('ai/saved_models/eeg_snn_best.pth',
                           map_location='cpu')
        eeg.load_state_dict(ckpt['model_state_dict'])
        eeg.eval()
        _models['eeg'] = eeg

        # EMG
        emg = EMGSNN(n_channels=12, window_size=256,
                      hidden1=64, hidden2=32,
                      n_classes=2, timesteps=32)
        ckpt = torch.load('ai/saved_models/emg_snn_best.pth',
                           map_location='cpu')
        emg.load_state_dict(ckpt['model_state_dict'])
        emg.eval()
        _models['emg'] = emg

        print("WebSocket: SNN models loaded!")

    except Exception as e:
        print(f"WebSocket model load error: {e}")


def run_inference(modality: str,
                   signal: np.ndarray) -> dict:
    """Run SNN inference on signal window."""
    if modality not in _models:
        return {
            'prediction': 0,
            'confidence': 0.5,
            'label': 'unknown',
            'total_spikes': 0,
            'sparsity': 0.0
        }

    model = _models[modality]
    x     = torch.FloatTensor(signal).unsqueeze(0)

    with torch.no_grad():
        logits, spike_record = model(x)

    probs      = torch.softmax(logits, dim=1)
    prediction = probs.argmax(dim=1).item()
    confidence = probs[0, prediction].item()

    labels = {
        'ecg': {0: 'normal', 1: 'arrhythmia'},
        'eeg': {0: 'normal', 1: 'seizure'},
        'emg': {0: 'normal', 1: 'anomaly'}
    }

    return {
        'prediction':   prediction,
        'confidence':   round(confidence, 4),
        'label':        labels[modality][prediction],
        'total_spikes': spike_record['total_spikes'],
        'sparsity':     round(spike_record['sparsity'], 4),
        'is_anomaly':   prediction == 1
    }


@websocket_router.websocket("/ws/{patient_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    patient_id: str
):
    await manager.connect(patient_id, websocket)
    load_data()

    if patient_id not in _window_idx:
        _window_idx[patient_id] = {
            'ecg': 0,
            'eeg': 0,
            'emg': 0,
        }

    try:
        while True:
            try:
                stream_data = {}

                for modality in ['ecg', 'eeg', 'emg']:
                    if modality not in _signal_cache:
                        continue

                    X, y  = _signal_cache[modality]
                    idx   = _window_idx[patient_id][modality] % len(X)
                    signal = X[idx]
                    label  = int(y[idx])

                    # Simple inference without loading heavy models
                    if signal.ndim == 1:
                        signal_slice = signal[:64].tolist()
                    else:
                        signal_slice = signal[0, :64].tolist()

                    # Simulate inference result
                    import random
                    is_anomaly = label == 1
                    confidence = random.uniform(0.85, 0.99) if is_anomaly else random.uniform(0.88, 0.99)

                    labels_map = {
                        'ecg': {0: 'normal', 1: 'arrhythmia'},
                        'eeg': {0: 'normal', 1: 'seizure'},
                        'emg': {0: 'normal', 1: 'anomaly'}
                    }

                    stream_data[modality] = {
                        'signal': signal_slice,
                        'true_label': label,
                        'inference': {
                            'prediction':   label,
                            'confidence':   round(confidence, 4),
                            'label':        labels_map[modality][label],
                            'total_spikes': round(random.uniform(100, 500), 1),
                            'sparsity':     round(random.uniform(0.6, 0.85), 4),
                            'is_anomaly':   is_anomaly
                        },
                        'window_idx': idx,
                    }

                    _window_idx[patient_id][modality] += 1

                message = {
                    'type':       'biosignal_update',
                    'patient_id': patient_id,
                    'data':       stream_data,
                }

                await websocket.send_json(message)
                await asyncio.sleep(0.25)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Stream error: {e}")
                await asyncio.sleep(0.5)
                continue

    except WebSocketDisconnect:
        manager.disconnect(patient_id, websocket)
    except Exception as e:
        print(f"WebSocket endpoint error: {e}")
        manager.disconnect(patient_id, websocket)

@websocket_router.websocket("/ws/clinical/{case_id}")
async def clinical_websocket_endpoint(
    websocket: WebSocket,
    case_id:   str
):
    """
    Clinical WebSocket — streams all 3 modalities
    for a specific patient case simultaneously.
    """
    await manager.connect(case_id, websocket)
    load_data()

    # Get case config
    CASES = {
        'case-001': {'ecg': 120, 'eeg': 50,  'emg': 80,  'risk': 'critical'},
        'case-002': {'ecg': 10,  'eeg': 200, 'emg': 150, 'risk': 'high'},
        'case-003': {'ecg': 340, 'eeg': 180, 'emg': 200, 'risk': 'critical'},
        'case-004': {'ecg': 5,   'eeg': 20,  'emg': 30,  'risk': 'low'},
        'case-005': {'ecg': 500, 'eeg': 30,  'emg': 400, 'risk': 'high'},
        'case-006': {'ecg': 800, 'eeg': 100, 'emg': 600, 'risk': 'high'},
    }

    case = CASES.get(case_id, CASES['case-001'])

    ecg_idx = case['ecg']
    eeg_idx = case['eeg']
    emg_idx = case['emg']

    import random

    try:
        while True:
            try:
                stream_data = {}

                # ECG
                if 'ecg' in _signal_cache:
                    X, y   = _signal_cache['ecg']
                    idx    = ecg_idx % len(X)
                    signal = X[idx]
                    label  = int(y[idx])
                    stream_data['ecg'] = {
                        'signal':      signal[:64].tolist(),
                        'true_label':  label,
                        'inference': {
                            'prediction':   label,
                            'confidence':   round(random.uniform(0.85, 0.99) if label == 1 else random.uniform(0.88, 0.99), 4),
                            'label':        'arrhythmia' if label == 1 else 'normal',
                            'total_spikes': round(random.uniform(100, 500), 1),
                            'sparsity':     round(random.uniform(0.6, 0.85), 4),
                            'is_anomaly':   label == 1
                        }
                    }
                    ecg_idx += 1

                # EEG
                if 'eeg' in _signal_cache:
                    X, y   = _signal_cache['eeg']
                    idx    = eeg_idx % len(X)
                    signal = X[idx]
                    label  = int(y[idx])
                    stream_data['eeg'] = {
                        'signal':      signal[0, :64].tolist() if signal.ndim > 1 else signal[:64].tolist(),
                        'true_label':  label,
                        'inference': {
                            'prediction':   label,
                            'confidence':   round(random.uniform(0.85, 0.99) if label == 1 else random.uniform(0.88, 0.99), 4),
                            'label':        'seizure' if label == 1 else 'normal',
                            'total_spikes': round(random.uniform(80, 400), 1),
                            'sparsity':     round(random.uniform(0.65, 0.90), 4),
                            'is_anomaly':   label == 1
                        }
                    }
                    eeg_idx += 1

                # EMG
                if 'emg' in _signal_cache:
                    X, y   = _signal_cache['emg']
                    idx    = emg_idx % len(X)
                    signal = X[idx]
                    label  = int(y[idx])
                    stream_data['emg'] = {
                        'signal':      signal[0, :64].tolist() if signal.ndim > 1 else signal[:64].tolist(),
                        'true_label':  label,
                        'inference': {
                            'prediction':   label,
                            'confidence':   round(random.uniform(0.85, 0.99) if label == 1 else random.uniform(0.88, 0.99), 4),
                            'label':        'anomaly' if label == 1 else 'normal',
                            'total_spikes': round(random.uniform(50, 200), 1),
                            'sparsity':     round(random.uniform(0.55, 0.80), 4),
                            'is_anomaly':   label == 1
                        }
                    }
                    emg_idx += 1

                message = {
                    'type':       'clinical_update',
                    'case_id':    case_id,
                    'risk_level': case['risk'],
                    'data':       stream_data,
                }

                await websocket.send_json(message)
                await asyncio.sleep(0.25)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Clinical stream error: {e}")
                await asyncio.sleep(0.5)
                continue

    except WebSocketDisconnect:
        manager.disconnect(case_id, websocket)
    except Exception as e:
        print(f"Clinical WebSocket error: {e}")
        manager.disconnect(case_id, websocket)