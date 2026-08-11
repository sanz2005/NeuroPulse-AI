/**
 * Analysis Store for NeuroPulse AI
 * Holds real analysis results — zero fake data
 */

import { create } from 'zustand'

interface SNNResult {
  prediction:      number
  label:           string
  confidence:      number
  conf_normal:     number
  conf_anomaly:    number
  is_anomaly:      boolean
  total_spikes:    number
  sparsity:        number
  layer1_activity: number[]
  layer2_activity: number[]
  layer3_activity: number[]
  latency_ms:      number
  energy_mj:       number
  model_type:      string
  framework:       string
}

interface BaselineResult {
  model:       string
  prediction:  number
  label:       string
  confidence:  number
  is_anomaly:  boolean
  latency_ms:  number
  energy_mj:   number
  parameters:  number
}

interface ClinicalResult {
  finding:    string
  severity:   string
  actions:    string[]
  icd_code:   string
  is_anomaly: boolean
}

interface AnalysisResult {
  modality:          string
  window_index?:     number
  true_label?:       number
  true_label_text:   string
  record_id?:        string
  start_sample?:     number
  end_sample?:       number
  time_start_sec?:   number
  time_end_sec?:     number
  signal:            number[]
  signal_shape:      number[]
  sample_rate:       number
  spike_train:       number[]
  spike_count:       number
  snn:               SNNResult
  baselines: {
    cnn:         BaselineResult
    lstm:        BaselineResult
    transformer: BaselineResult
  }
  attribution:       number[]
  attribution_peak:  number
  clinical:          ClinicalResult
}

interface SignalPreview {
  sample_rate: any
  modality:     string
  window_index: number
  signal:       number[]
  spike_train:  number[]
  true_label:   number
  label_text:   string
  is_anomaly:   boolean
  spike_count:  number
  spike_rate:   number
}

interface WindowInfo {
  index:      number
  label:      number
  is_anomaly: boolean
  label_text: string
}

interface AnalysisStore {
  // Selection state
  selectedModality:    string | null
  selectedWindowIndex: number | null
  availableWindows:    WindowInfo[]
  totalWindows:        number
  normalCount:         number
  anomalyCount:        number

  // Preview state
  preview:    SignalPreview | null
  isLoading:  boolean
  error:      string | null

  // Analysis result
  result:     AnalysisResult | null
  isAnalyzing: boolean

  // Active tab
  activeTab:  string

  // Actions
  setModality:       (modality: string) => void
  setWindowIndex:    (index: number) => void
  setWindows:        (data: any) => void
  setPreview:        (preview: SignalPreview | null) => void
  setResult:         (result: AnalysisResult | null) => void
  setLoading:        (loading: boolean) => void
  setAnalyzing:      (analyzing: boolean) => void
  setError:          (error: string | null) => void
  setActiveTab:      (tab: string) => void
  reset:             () => void
}

export const useAnalysisStore = create<AnalysisStore>((set) => ({
  selectedModality:    null,
  selectedWindowIndex: null,
  availableWindows:    [],
  totalWindows:        0,
  normalCount:         0,
  anomalyCount:        0,
  preview:             null,
  isLoading:           false,
  error:               null,
  result:              null,
  isAnalyzing:         false,
  activeTab:           'clinical',

  setModality:    (modality) => set({
    selectedModality:    modality,
    selectedWindowIndex: null,
    preview:             null,
    result:              null,
    error:               null,
  }),

  setWindowIndex: (index) => set({ selectedWindowIndex: index }),

  setWindows: (data) => set({
    availableWindows: data.windows,
    totalWindows:     data.total_windows,
    normalCount:      data.normal_count,
    anomalyCount:     data.anomaly_count,
  }),

  setPreview:   (preview)   => set({ preview }),
  setResult:    (result)    => set({ result }),
  setLoading:   (isLoading) => set({ isLoading }),
  setAnalyzing: (isAnalyzing) => set({ isAnalyzing }),
  setError:     (error)     => set({ error }),
  setActiveTab: (activeTab) => set({ activeTab }),

  reset: () => set({
    selectedModality:    null,
    selectedWindowIndex: null,
    availableWindows:    [],
    preview:             null,
    result:              null,
    error:               null,
    activeTab:           'clinical',
  }),
}))