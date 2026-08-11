"""
Benchmark API Routes for NeuroPulse AI
Serves model comparison results.
"""

import json
import os
from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/results/{modality}")
async def get_benchmark_results(modality: str):
    """Get benchmark results for a modality."""
    if modality not in ['ecg', 'eeg', 'emg']:
        raise HTTPException(
            status_code=400,
            detail="Modality must be ecg, eeg, or emg"
        )

    path = f'ai/saved_models/{modality}_benchmark_results.json'
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"Benchmark results not found for {modality}"
        )

    with open(path) as f:
        return json.load(f)


@router.get("/results/all")
async def get_all_benchmark_results():
    """Get all benchmark results."""
    results = {}
    for modality in ['ecg', 'eeg', 'emg']:
        path = f'ai/saved_models/{modality}_benchmark_results.json'
        if os.path.exists(path):
            with open(path) as f:
                results[modality] = json.load(f)
    return results


@router.get("/summary")
async def get_benchmark_summary():
    """Get energy efficiency summary."""
    summary = {}
    for modality in ['ecg', 'eeg', 'emg']:
        path = f'ai/saved_models/{modality}_benchmark_results.json'
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)

        snn_key  = f'{modality.upper()}-SNN'
        if snn_key not in data:
            continue

        snn_energy  = data[snn_key]['energy_mj']
        other_energies = [
            v['energy_mj'] for k, v in data.items()
            if k != snn_key and v['energy_mj'] > 0
        ]

        if other_energies:
            max_energy = max(other_energies)
            saving     = (max_energy / snn_energy
                          if snn_energy > 0 else 0)
        else:
            saving = 0

        summary[modality] = {
            'snn_accuracy':   data[snn_key]['accuracy'],
            'snn_energy_mj':  snn_energy,
            'snn_sparsity':   data[snn_key]['sparsity'],
            'energy_saving':  round(saving, 1),
            'snn_f1':         data[snn_key]['f1']
        }

    return summary