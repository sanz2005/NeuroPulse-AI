/**
 * API Client for NeuroPulse AI Backend
 */

import axios from 'axios'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
  headers: { 'Content-Type': 'application/json' }
})

// ── Patient APIs ───────────────────────────────────────────────────────────────
export const patientAPI = {
  getAll: () => api.get('/api/patients/'),
  getById: (id: string) => api.get(`/api/patients/${id}`),
  create: (data: any) => api.post('/api/patients/', data),
  discharge: (id: string) => api.put(`/api/patients/${id}/discharge`),
}

// ── Signal APIs ────────────────────────────────────────────────────────────────
export const signalAPI = {
  getECG:   (idx: number) => api.get(`/api/signals/ecg/${idx}`),
  getEEG:   (idx: number) => api.get(`/api/signals/eeg/${idx}`),
  getEMG:   (idx: number) => api.get(`/api/signals/emg/${idx}`),
  getStats: () => api.get('/api/signals/stats'),
}

// ── Inference APIs ─────────────────────────────────────────────────────────────
export const inferenceAPI = {
  ecg:    (data: any) => api.post('/api/inference/ecg', data),
  eeg:    (data: any) => api.post('/api/inference/eeg', data),
  emg:    (data: any) => api.post('/api/inference/emg', data),
  status: () => api.get('/api/inference/models/status'),
}

// ── Alert APIs ─────────────────────────────────────────────────────────────────
export const alertAPI = {
  getAll:       () => api.get('/api/alerts/'),
  getByPatient: (id: string) => api.get(`/api/alerts/${id}`),
  create:       (data: any) => api.post('/api/alerts/', data),
  resolve:      (id: string) => api.put(`/api/alerts/${id}/resolve`),
}

// ── Benchmark APIs ─────────────────────────────────────────────────────────────
export const benchmarkAPI = {
  getResults:  (modality: string) => api.get(`/api/benchmark/results/${modality}`),
  getAll:      () => api.get('/api/benchmark/results/all'),
  getSummary:  () => api.get('/api/benchmark/summary'),
}

// ── XAI APIs ──────────────────────────────────────────────────────────────────
export const xaiAPI = {
  getAttribution: (data: any) => api.post('/api/xai/attribution', data),
}