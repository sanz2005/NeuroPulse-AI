'use client'

import { useState, useEffect, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { useAnalysisStore } from '@/store/analysisStore'
import { api } from '@/lib/api'

// ── Modality Card ──────────────────────────────────────────────────────────────
function ModalityCard({
  id, icon, title, subtitle, dataset,
  color, selected, onClick
}: {
  id: string, icon: string, title: string,
  subtitle: string, dataset: string,
  color: string, selected: boolean,
  onClick: () => void
}) {
  return (
    <div
      onClick={onClick}
      style={{
        border:       `2px solid ${selected ? color : '#E5E7EB'}`,
        borderRadius: '16px',
        padding:      '24px',
        cursor:       'pointer',
        background:   selected ? `${color}08` : 'white',
        transition:   'all 0.2s ease',
        boxShadow:    selected
          ? `0 0 0 4px ${color}20`
          : '0 1px 3px rgba(0,0,0,0.08)',
      }}
    >
      <div style={{ fontSize: '40px', marginBottom: '12px' }}>{icon}</div>
      <div style={{
        fontSize:     '18px',
        fontWeight:   700,
        color:        '#111827',
        marginBottom: '4px'
      }}>
        {title}
      </div>
      <div style={{
        fontSize:     '14px',
        color:        '#6B7280',
        marginBottom: '12px'
      }}>
        {subtitle}
      </div>
      <div style={{
        display:      'inline-flex',
        alignItems:   'center',
        gap:          '6px',
        background:   `${color}15`,
        border:       `1px solid ${color}40`,
        borderRadius: '6px',
        padding:      '4px 10px',
        fontSize:     '12px',
        fontWeight:   600,
        color:        color
      }}>
        📂 {dataset}
      </div>
      {selected && (
        <div style={{
          marginTop:  '12px',
          fontSize:   '13px',
          fontWeight: 600,
          color:      color
        }}>
          ✓ Selected
        </div>
      )}
    </div>
  )
}

// ── Main Page ──────────────────────────────────────────────────────────────────
export default function HomePage() {
  const router = useRouter()
  const {
    selectedModality, selectedWindowIndex,
    availableWindows, totalWindows,
    normalCount, anomalyCount,
    preview, isLoading, error,
    setModality, setWindowIndex,
    setWindows, setPreview,
    setLoading, setError, setAnalyzing
  } = useAnalysisStore()

  const [windowInput, setWindowInput] = useState('')
  const [stats, setStats]             = useState<any>(null)

  // Load dataset stats on mount
  useEffect(() => {
    api.get('/api/analyze/dataset-stats')
       .then(r => setStats(r.data))
       .catch(() => {})
  }, [])

  // Load windows when modality selected
  const handleModalitySelect = async (modality: string) => {
    setModality(modality)
    setLoading(true)
    setError(null)
    try {
      const r = await api.get(`/api/analyze/records/${modality}`)
      setWindows(r.data)
    } catch (e: any) {
      setError(`Failed to load ${modality} records`)
    } finally {
      setLoading(false)
    }
  }

  // Load preview when window selected
  const handleWindowSelect = async (index: number) => {
    if (!selectedModality) return
    setWindowIndex(index)
    setWindowInput(String(index))
    setLoading(true)
    setError(null)
    try {
      const r = await api.get(
        `/api/analyze/signal/${selectedModality}/${index}`
      )
      setPreview(r.data)
    } catch (e: any) {
      setError('Failed to load signal preview')
    } finally {
      setLoading(false)
    }
  }

  // Handle window input
  const handleWindowInputChange = (val: string) => {
    setWindowInput(val)
    const num = parseInt(val)
    if (!isNaN(num) && num >= 0 && num < totalWindows) {
      handleWindowSelect(num)
    }
  }

  // Analyze
  const handleAnalyze = async () => {
    if (!selectedModality || selectedWindowIndex === null) return
    setAnalyzing(true)
    setError(null)
    try {
      const r = await api.post('/api/analyze/analyze', {
        modality:     selectedModality,
        window_index: selectedWindowIndex
      })
      useAnalysisStore.getState().setResult(r.data)
      router.push('/analyze')
    } catch (e: any) {
      setError(e.response?.data?.detail || 'Analysis failed')
    } finally {
      setAnalyzing(false)
    }
  }

  const modalities = [
    {
      id:       'ecg',
      icon:     '❤️',
      title:    'ECG Analysis',
      subtitle: 'Cardiac arrhythmia detection',
      dataset:  'MIT-BIH Arrhythmia Database',
      color:    '#EF4444',
    },
    {
      id:       'eeg',
      icon:     '🧠',
      title:    'EEG Analysis',
      subtitle: 'Seizure detection',
      dataset:  'CHB-MIT Scalp EEG Database',
      color:    '#8B5CF6',
    },
    {
      id:       'emg',
      icon:     '💪',
      title:    'EMG Analysis',
      subtitle: 'Muscle anomaly detection',
      dataset:  'NinaPro DB2',
      color:    '#F59E0B',
    },
  ]

  return (
    <div style={{
      background:  '#F9FAFB',
      minHeight:   '100vh',
      fontFamily:  '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    }}>

      {/* Navbar */}
      <nav style={{
        background:   'white',
        borderBottom: '1px solid #E5E7EB',
        padding:      '0 32px',
        display:      'flex',
        alignItems:   'center',
        justifyContent: 'space-between',
        height:       '64px',
        position:     'sticky',
        top:          0,
        zIndex:       100,
        boxShadow:    '0 1px 3px rgba(0,0,0,0.05)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width:        '36px',
            height:       '36px',
            borderRadius: '10px',
            background:   'linear-gradient(135deg, #6366F1, #8B5CF6)',
            display:      'flex',
            alignItems:   'center',
            justifyContent: 'center',
            fontSize:     '18px'
          }}>
            ⚡
          </div>
          <div>
            <div style={{
              fontSize:   '16px',
              fontWeight: 700,
              color:      '#111827'
            }}>
              NeuroPulse AI
            </div>
            <div style={{ fontSize: '11px', color: '#6B7280' }}>
              Neuromorphic Biosignal Analysis Platform
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={() => router.push('/research')}
            style={{
              padding:      '8px 16px',
              background:   'white',
              border:       '1px solid #E5E7EB',
              borderRadius: '8px',
              fontSize:     '13px',
              fontWeight:   500,
              color:        '#374151',
              cursor:       'pointer'
            }}
          >
            🔬 Research
          </button>
        </div>
      </nav>

      <div style={{ maxWidth: '900px', margin: '0 auto', padding: '40px 24px' }}>

        {/* Hero */}
        <div style={{ textAlign: 'center', marginBottom: '48px' }}>
          <div style={{
            display:      'inline-flex',
            alignItems:   'center',
            gap:          '8px',
            background:   '#EEF2FF',
            border:       '1px solid #C7D2FE',
            borderRadius: '100px',
            padding:      '6px 16px',
            fontSize:     '13px',
            fontWeight:   600,
            color:        '#6366F1',
            marginBottom: '16px'
          }}>
            ⚡ Powered by Spiking Neural Networks
          </div>
          <h1 style={{
            fontSize:     '36px',
            fontWeight:   800,
            color:        '#111827',
            marginBottom: '12px',
            lineHeight:   '1.2'
          }}>
            Neuromorphic Biosignal<br />Analysis Platform
          </h1>
          <p style={{
            fontSize:  '16px',
            color:     '#6B7280',
            maxWidth:  '520px',
            margin:    '0 auto'
          }}>
            Select a biosignal from real clinical datasets.
            The SNN analyzes it and provides explainable clinical insights.
          </p>
        </div>

        {/* Dataset Stats */}
        {stats && (
          <div style={{
            display:             'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap:                 '12px',
            marginBottom:        '40px'
          }}>
            {[
              { key: 'ecg', label: 'ECG Windows', color: '#EF4444', icon: '❤️' },
              { key: 'eeg', label: 'EEG Windows', color: '#8B5CF6', icon: '🧠' },
              { key: 'emg', label: 'EMG Windows', color: '#F59E0B', icon: '💪' },
            ].map(({ key, label, color, icon }) => (
              <div key={key} style={{
                background:   'white',
                border:       '1px solid #E5E7EB',
                borderRadius: '12px',
                padding:      '16px',
                boxShadow:    '0 1px 3px rgba(0,0,0,0.05)'
              }}>
                <div style={{
                  display:      'flex',
                  alignItems:   'center',
                  gap:          '8px',
                  marginBottom: '8px'
                }}>
                  <span style={{ fontSize: '20px' }}>{icon}</span>
                  <span style={{
                    fontSize:   '13px',
                    fontWeight: 600,
                    color:      '#374151'
                  }}>
                    {label}
                  </span>
                </div>
                <div style={{
                  fontSize:   '24px',
                  fontWeight: 800,
                  color:      color
                }}>
                  {stats[key]?.total?.toLocaleString() || '--'}
                </div>
                <div style={{ fontSize: '12px', color: '#9CA3AF', marginTop: '2px' }}>
                  {stats[key]?.normal?.toLocaleString()} normal ·{' '}
                  {stats[key]?.anomaly?.toLocaleString()} anomaly
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Step 1 — Select Modality */}
        <div style={{ marginBottom: '32px' }}>
          <div style={{
            display:      'flex',
            alignItems:   'center',
            gap:          '10px',
            marginBottom: '16px'
          }}>
            <div style={{
              width:          '28px',
              height:         '28px',
              borderRadius:   '50%',
              background:     '#6366F1',
              color:          'white',
              display:        'flex',
              alignItems:     'center',
              justifyContent: 'center',
              fontSize:       '14px',
              fontWeight:     700,
              flexShrink:     0
            }}>
              1
            </div>
            <h2 style={{
              fontSize:   '18px',
              fontWeight: 700,
              color:      '#111827',
              margin:     0
            }}>
              Select Signal Type
            </h2>
          </div>

          <div style={{
            display:             'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap:                 '16px'
          }}>
            {modalities.map(m => (
              <ModalityCard
                key={m.id}
                {...m}
                selected={selectedModality === m.id}
                onClick={() => handleModalitySelect(m.id)}
              />
            ))}
          </div>
        </div>

        {/* Step 2 — Select Window */}
        {selectedModality && (
          <div style={{ marginBottom: '32px' }}>
            <div style={{
              display:      'flex',
              alignItems:   'center',
              gap:          '10px',
              marginBottom: '16px'
            }}>
              <div style={{
                width:          '28px',
                height:         '28px',
                borderRadius:   '50%',
                background:     '#6366F1',
                color:          'white',
                display:        'flex',
                alignItems:     'center',
                justifyContent: 'center',
                fontSize:       '14px',
                fontWeight:     700,
                flexShrink:     0
              }}>
                2
              </div>
              <h2 style={{
                fontSize:   '18px',
                fontWeight: 700,
                color:      '#111827',
                margin:     0
              }}>
                Select Signal Window
              </h2>
            </div>

            <div style={{
              background:   'white',
              border:       '1px solid #E5E7EB',
              borderRadius: '16px',
              padding:      '24px',
              boxShadow:    '0 1px 3px rgba(0,0,0,0.05)'
            }}>
              {/* Dataset info */}
              <div style={{
                display:             'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap:                 '12px',
                marginBottom:        '20px'
              }}>
                {[
                  { label: 'Total Windows', value: totalWindows.toLocaleString(), color: '#6366F1' },
                  { label: 'Normal',        value: normalCount.toLocaleString(),  color: '#10B981' },
                  { label: 'Anomaly',       value: anomalyCount.toLocaleString(), color: '#EF4444' },
                ].map(({ label, value, color }) => (
                  <div key={label} style={{
                    background:   '#F9FAFB',
                    borderRadius: '8px',
                    padding:      '12px 16px',
                    textAlign:    'center'
                  }}>
                    <div style={{
                      fontSize:   '22px',
                      fontWeight: 800,
                      color
                    }}>
                      {value}
                    </div>
                    <div style={{ fontSize: '12px', color: '#9CA3AF' }}>
                      {label}
                    </div>
                  </div>
                ))}
              </div>

              {/* Window input */}
              <div style={{ marginBottom: '16px' }}>
                <label style={{
                  display:      'block',
                  fontSize:     '13px',
                  fontWeight:   600,
                  color:        '#374151',
                  marginBottom: '6px'
                }}>
                  Enter Window Index (0 — {totalWindows - 1})
                </label>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <input
                    type="number"
                    value={windowInput}
                    onChange={e => handleWindowInputChange(e.target.value)}
                    placeholder={`0 to ${totalWindows - 1}`}
                    min={0}
                    max={totalWindows - 1}
                    style={{
                      flex:         1,
                      padding:      '10px 14px',
                      border:       '1px solid #D1D5DB',
                      borderRadius: '8px',
                      fontSize:     '14px',
                      outline:      'none',
                      color:        '#111827'
                    }}
                  />
                  {/* Quick picks */}
                  <button
                    onClick={() => {
                      const normal = availableWindows.find(w => !w.is_anomaly)
                      if (normal) handleWindowSelect(normal.index)
                    }}
                    style={{
                      padding:      '10px 14px',
                      background:   '#D1FAE5',
                      border:       '1px solid #6EE7B7',
                      borderRadius: '8px',
                      fontSize:     '13px',
                      fontWeight:   600,
                      color:        '#065F46',
                      cursor:       'pointer',
                      whiteSpace:   'nowrap'
                    }}
                  >
                    Pick Normal
                  </button>
                  <button
                    onClick={() => {
                      const anomaly = availableWindows.find(w => w.is_anomaly)
                      if (anomaly) handleWindowSelect(anomaly.index)
                    }}
                    style={{
                      padding:      '10px 14px',
                      background:   '#FEE2E2',
                      border:       '1px solid #FCA5A5',
                      borderRadius: '8px',
                      fontSize:     '13px',
                      fontWeight:   600,
                      color:        '#991B1B',
                      cursor:       'pointer',
                      whiteSpace:   'nowrap'
                    }}
                  >
                    Pick Anomaly
                  </button>
                </div>
              </div>

              {/* Quick window browser */}
              {availableWindows.length > 0 && (
                <div>
                  <div style={{
                    fontSize:     '12px',
                    fontWeight:   600,
                    color:        '#6B7280',
                    marginBottom: '8px'
                  }}>
                    Quick Browse — click any window:
                  </div>
                  <div style={{
                    display:   'flex',
                    flexWrap:  'wrap',
                    gap:       '6px',
                    maxHeight: '120px',
                    overflowY: 'auto',
                    padding:   '8px',
                    background: '#F9FAFB',
                    borderRadius: '8px'
                  }}>
                    {availableWindows.slice(0, 80).map(w => (
                      <button
                        key={w.index}
                        onClick={() => handleWindowSelect(w.index)}
                        title={w.label_text}
                        style={{
                          width:        '36px',
                          height:       '28px',
                          border:       `1px solid ${
                            selectedWindowIndex === w.index
                              ? '#6366F1'
                              : w.is_anomaly ? '#FCA5A5' : '#BBF7D0'
                          }`,
                          borderRadius: '6px',
                          background:   selectedWindowIndex === w.index
                            ? '#6366F1'
                            : w.is_anomaly ? '#FEF2F2' : '#F0FDF4',
                          color:        selectedWindowIndex === w.index
                            ? 'white'
                            : w.is_anomaly ? '#991B1B' : '#065F46',
                          fontSize:     '10px',
                          fontWeight:   600,
                          cursor:       'pointer',
                          padding:      0
                        }}
                      >
                        {w.index}
                      </button>
                    ))}
                  </div>
                  <div style={{
                    fontSize: '11px', color: '#9CA3AF', marginTop: '6px'
                  }}>
                    🟢 Normal · 🔴 Anomaly
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Step 3 — Signal Preview */}
        {preview && (
          <div style={{ marginBottom: '32px' }}>
            <div style={{
              display:      'flex',
              alignItems:   'center',
              gap:          '10px',
              marginBottom: '16px'
            }}>
              <div style={{
                width:          '28px',
                height:         '28px',
                borderRadius:   '50%',
                background:     '#6366F1',
                color:          'white',
                display:        'flex',
                alignItems:     'center',
                justifyContent: 'center',
                fontSize:       '14px',
                fontWeight:     700,
                flexShrink:     0
              }}>
                3
              </div>
              <h2 style={{
                fontSize:   '18px',
                fontWeight: 700,
                color:      '#111827',
                margin:     0
              }}>
                Signal Preview
              </h2>
              <div style={{
                marginLeft:   'auto',
                background:   preview.is_anomaly ? '#FEE2E2' : '#D1FAE5',
                border:       `1px solid ${preview.is_anomaly ? '#FCA5A5' : '#6EE7B7'}`,
                borderRadius: '100px',
                padding:      '4px 12px',
                fontSize:     '12px',
                fontWeight:   700,
                color:        preview.is_anomaly ? '#991B1B' : '#065F46'
              }}>
                {preview.label_text}
              </div>
            </div>

            <div style={{
              background:   'white',
              border:       `1px solid ${preview.is_anomaly ? '#FCA5A5' : '#E5E7EB'}`,
              borderRadius: '16px',
              padding:      '24px',
              boxShadow:    '0 1px 3px rgba(0,0,0,0.05)'
            }}>
              {/* Signal stats */}
              <div style={{
                display:             'grid',
                gridTemplateColumns: 'repeat(4, 1fr)',
                gap:                 '12px',
                marginBottom:        '20px'
              }}>
                {[
                  { label: 'Window Index', value: preview.window_index },
                  { label: 'Sample Rate',  value: `${preview.sample_rate} Hz` },
                  { label: 'Spike Count',  value: preview.spike_count },
                  { label: 'Spike Rate',   value: `${preview.spike_rate}/s` },
                ].map(({ label, value }) => (
                  <div key={label} style={{
                    background:   '#F9FAFB',
                    borderRadius: '8px',
                    padding:      '10px 14px'
                  }}>
                    <div style={{ fontSize: '11px', color: '#9CA3AF' }}>
                      {label}
                    </div>
                    <div style={{
                      fontSize:   '16px',
                      fontWeight: 700,
                      color:      '#111827',
                      marginTop:  '2px'
                    }}>
                      {value}
                    </div>
                  </div>
                ))}
              </div>

              {/* Signal waveform canvas */}
              <SignalPreviewCanvas
                signal={preview.signal}
                spikeTrain={preview.spike_train}
                isAnomaly={preview.is_anomaly}
                modality={selectedModality || 'ecg'}
              />
            </div>
          </div>
        )}

        {/* Analyze Button */}
        {preview && selectedWindowIndex !== null && (
          <div style={{ textAlign: 'center' }}>
            <button
              onClick={handleAnalyze}
              style={{
                padding:      '16px 48px',
                background:   'linear-gradient(135deg, #6366F1, #8B5CF6)',
                border:       'none',
                borderRadius: '12px',
                fontSize:     '16px',
                fontWeight:   700,
                color:        'white',
                cursor:       'pointer',
                boxShadow:    '0 4px 14px rgba(99,102,241,0.4)',
                transition:   'transform 0.1s ease'
              }}
            >
              ⚡ Analyze with SNN
            </button>
            <div style={{
              fontSize:  '13px',
              color:     '#9CA3AF',
              marginTop: '8px'
            }}>
              Runs SNN + CNN + LSTM + Transformer on this exact signal
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div style={{
            background:   '#FEE2E2',
            border:       '1px solid #FCA5A5',
            borderRadius: '8px',
            padding:      '12px 16px',
            color:        '#991B1B',
            fontSize:     '14px',
            marginTop:    '16px'
          }}>
            ⚠️ {error}
          </div>
        )}

        {/* Loading */}
        {isLoading && (
          <div style={{
            textAlign:  'center',
            padding:    '20px',
            color:      '#6B7280',
            fontSize:   '14px'
          }}>
            Loading...
          </div>
        )}
      </div>
    </div>
  )
}

// ── Signal Preview Canvas ──────────────────────────────────────────────────────
function SignalPreviewCanvas({
  signal, spikeTrain, isAnomaly, modality
}: {
  signal: number[], spikeTrain: number[],
  isAnomaly: boolean, modality: string
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || signal.length === 0) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const w = canvas.width
    const h = canvas.height
    const mid = h * 0.6

    ctx.fillStyle = '#FFFFFF'
    ctx.fillRect(0, 0, w, h)

    // Grid
    ctx.strokeStyle = '#F3F4F6'
    ctx.lineWidth   = 1
    for (let x = 0; x < w; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke()
    }
    for (let y = 0; y < h; y += 20) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke()
    }

    // Signal
    const colors: Record<string, string> = {
      ecg: '#EF4444', eeg: '#8B5CF6', emg: '#F59E0B'
    }
    const color = isAnomaly ? '#EF4444' : (colors[modality] || '#6366F1')

    const min   = Math.min(...signal)
    const max   = Math.max(...signal)
    const range = max - min || 1

    ctx.strokeStyle = color
    ctx.lineWidth   = 1.5
    ctx.beginPath()
    signal.forEach((val, i) => {
      const x = (i / signal.length) * w
      const y = mid - ((val - min) / range) * (mid * 0.8)
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    })
    ctx.stroke()

    // Spike train
    const spikeY = h * 0.82
    spikeTrain.forEach((spike, i) => {
      if (spike > 0.5) {
        const x = (i / spikeTrain.length) * w
        ctx.strokeStyle = '#6366F1'
        ctx.lineWidth   = 1
        ctx.globalAlpha = 0.7
        ctx.beginPath()
        ctx.moveTo(x, spikeY - 6)
        ctx.lineTo(x, spikeY + 6)
        ctx.stroke()
        ctx.globalAlpha = 1
      }
    })

    // Labels
    ctx.fillStyle = color
    ctx.font      = '11px sans-serif'
    ctx.fillText(`${modality.toUpperCase()} Signal`, 8, 16)
    ctx.fillStyle = '#6366F1'
    ctx.fillText('Spike Train ↓', 8, spikeY - 10)

  }, [signal, spikeTrain, isAnomaly, modality])

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={180}
      style={{
        width:        '100%',
        height:       '180px',
        display:      'block',
        borderRadius: '8px',
        border:       '1px solid #F3F4F6'
      }}
    />
  )
}