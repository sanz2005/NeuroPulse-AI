'use client'

import { useEffect, useState, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { useAnalysisStore } from '@/store/analysisStore'
import { api } from '@/lib/api'
import AIReasoningPanel from '@/components/clinical/AIReasoningPanel'
import ClinicalPriorityPanel from '@/components/clinical/ClinicalPriorityPanel'
import DecisionTimeline from '@/components/clinical/DecisionTimeline'
import DoctorSummary from '@/components/clinical/DoctorSummary'
import ExportReportCenter from '@/components/clinical/ExportReportCenter'
import InteractiveSignalCanvas from '@/components/clinical/InteractiveSignalCanvas'

// ── Tab Button ─────────────────────────────────────────────────────────────────
function TabButton({
  id, label, active, onClick
}: {
  id: string, label: string,
  active: boolean, onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding:        '10px 20px',
        background:     active ? '#6366F1' : 'white',
        border:         `1px solid ${active ? '#6366F1' : '#E5E7EB'}`,
        borderRadius:   '8px',
        fontSize:       '13px',
        fontWeight:     600,
        color:          active ? 'white' : '#374151',
        cursor:         'pointer',
        transition:     'all 0.15s ease',
        whiteSpace:     'nowrap'
      }}
    >
      {label}
    </button>
  )
}

// ── Signal Canvas ──────────────────────────────────────────────────────────────
function SignalCanvas({
  signal, spikeTrain, attribution,
  isAnomaly, modality, height = 160
}: {
  signal: number[], spikeTrain: number[],
  attribution: number[], isAnomaly: boolean,
  modality: string, height?: number
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || signal.length === 0) return
    const ctx    = canvas.getContext('2d')
    if (!ctx) return

    const w   = canvas.width
    const h   = canvas.height
    const mid = h * 0.55

    ctx.fillStyle = '#FFFFFF'
    ctx.fillRect(0, 0, w, h)

    // Attribution heatmap background
    if (attribution.length > 0) {
      attribution.forEach((val, i) => {
        const x    = (i / attribution.length) * w
        const barW = w / attribution.length + 1
        ctx.fillStyle = isAnomaly
          ? `rgba(239,68,68,${val * 0.25})`
          : `rgba(99,102,241,${val * 0.15})`
        ctx.fillRect(x, 0, barW, mid * 1.1)
      })
    }

    // Grid
    ctx.strokeStyle = '#F3F4F6'
    ctx.lineWidth   = 0.5
    for (let x = 0; x < w; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke()
    }

    // Signal line
    const colors: Record<string, string> = {
      ecg: '#EF4444', eeg: '#8B5CF6', emg: '#F59E0B'
    }
    const color = colors[modality] || '#6366F1'
    const min   = Math.min(...signal)
    const max   = Math.max(...signal)
    const range = max - min || 1

    ctx.strokeStyle = isAnomaly ? '#EF4444' : color
    ctx.lineWidth   = 2
    ctx.beginPath()
    signal.forEach((val, i) => {
      const x = (i / signal.length) * w
      const y = mid - ((val - min) / range) * (mid * 0.75)
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    })
    ctx.stroke()

    // Anomaly peak marker
    if (isAnomaly && attribution.length > 0) {
      const peakIdx = attribution.indexOf(Math.max(...attribution))
      const peakX   = (peakIdx / attribution.length) * w
      ctx.strokeStyle = '#EF4444'
      ctx.lineWidth   = 2
      ctx.setLineDash([4, 4])
      ctx.beginPath()
      ctx.moveTo(peakX, 0)
      ctx.lineTo(peakX, mid * 1.1)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = '#EF4444'
      ctx.font      = 'bold 11px sans-serif'
      ctx.fillText('⚠ Peak', peakX + 4, 14)
    }

    // Spike train
    const spikeZone = h * 0.78
    ctx.fillStyle   = '#6B7280'
    ctx.font        = '10px sans-serif'
    ctx.fillText('Spike Train:', 4, spikeZone - 8)

    spikeTrain.forEach((spike, i) => {
      if (spike > 0.5) {
        const x = (i / spikeTrain.length) * w
        ctx.strokeStyle = '#6366F1'
        ctx.lineWidth   = 1.2
        ctx.globalAlpha = 0.8
        ctx.beginPath()
        ctx.moveTo(x, spikeZone - 6)
        ctx.lineTo(x, spikeZone + 6)
        ctx.stroke()
        ctx.globalAlpha = 1
      }
    })

    // Border
    ctx.strokeStyle = '#E5E7EB'
    ctx.lineWidth   = 1
    ctx.setLineDash([])
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1)

  }, [signal, spikeTrain, attribution, isAnomaly, modality])

  return (
    <canvas
      ref={canvasRef}
      width={900}
      height={height}
      style={{
        width:        '100%',
        height:       `${height}px`,
        display:      'block',
        borderRadius: '8px'
      }}
    />
  )
}

// ── Confidence Bar ─────────────────────────────────────────────────────────────
function ConfidenceBar({
  label, value, color, max = 1
}: {
  label: string, value: number,
  color: string, max?: number
}) {
  const pct = Math.round((value / max) * 100)
  return (
    <div style={{ marginBottom: '10px' }}>
      <div style={{
        display:        'flex',
        justifyContent: 'space-between',
        marginBottom:   '4px',
        fontSize:       '13px'
      }}>
        <span style={{ color: '#374151', fontWeight: 500 }}>{label}</span>
        <span style={{ color, fontWeight: 700 }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div style={{
        background:   '#F3F4F6',
        borderRadius: '100px',
        height:       '8px',
        overflow:     'hidden'
      }}>
        <div style={{
          width:        `${pct}%`,
          height:       '100%',
          background:   color,
          borderRadius: '100px',
          transition:   'width 0.6s ease'
        }} />
      </div>
    </div>
  )
}

// ── Layer Activity Bar ─────────────────────────────────────────────────────────
function LayerBar({
  label, activity, color
}: {
  label: string, activity: number[], color: string
}) {
  const avg = activity.length > 0
    ? activity.reduce((a, b) => a + b, 0) / activity.length
    : 0
  const pct = Math.min(100, avg * 100 * 5)

  return (
    <div style={{ marginBottom: '8px' }}>
      <div style={{
        display:        'flex',
        justifyContent: 'space-between',
        marginBottom:   '3px',
        fontSize:       '12px'
      }}>
        <span style={{ color: '#6B7280' }}>{label}</span>
        <span style={{ color, fontWeight: 600 }}>
          {avg.toFixed(4)}
        </span>
      </div>
      <div style={{
        background:   '#F3F4F6',
        borderRadius: '100px',
        height:       '6px',
        overflow:     'hidden'
      }}>
        <div style={{
          width:      `${pct}%`,
          height:     '100%',
          background: color,
          borderRadius: '100px'
        }} />
      </div>
    </div>
  )
}

// ── Main Analyze Page ──────────────────────────────────────────────────────────
export default function AnalyzePage() {
  const router = useRouter()
  const {
    selectedModality, selectedWindowIndex,
    result, isAnalyzing, activeTab,
    setResult, setAnalyzing, setError,
    setActiveTab
  } = useAnalysisStore()

  useEffect(() => {
    if (result) {
      setAnalyzing(false)
      return
    }
    if (!selectedModality || selectedWindowIndex === null) {
      router.push('/')
      return
    }

    // Run real analysis
    const runAnalysis = async () => {
      setAnalyzing(true)
      try {
        const r = await api.post('/api/analyze/analyze', {
          modality:     selectedModality,
          window_index: selectedWindowIndex
        })
        setResult(r.data)
      } catch (e: any) {
        setError(e.response?.data?.detail || 'Analysis failed')
        router.push('/')
      } finally {
        setAnalyzing(false)
      }
    }

    if (!result || (result as any).window_index !== selectedWindowIndex) {
      runAnalysis()
    }
  }, [])

  const tabs = [
    { id: 'clinical',    label: '🏥 Clinical Report'   },
    { id: 'signal',      label: '📡 Signal Analysis'   },
    { id: 'xai',         label: '👁 XAI Explanation'   },
    { id: 'comparison',  label: '📊 Model Comparison'  },
    { id: 'neural',      label: '🧠 Neural Activity'   },
    { id: 'reasoning',   label: '🧠 AI Reasoning'      },
    { id: 'timeline',    label: '⏱ Decision Timeline'  },
    { id: 'doctor',      label: '🩺 Doctor Summary'    },
    { id: 'export',      label: '📤 Export Report'     },
  ]

  const severityColors: Record<string, string> = {
    critical: '#EF4444',
    high:     '#F97316',
    medium:   '#F59E0B',
    normal:   '#10B981',
  }

  const severityBg: Record<string, string> = {
    critical: '#FEE2E2',
    high:     '#FFEDD5',
    medium:   '#FEF3C7',
    normal:   '#D1FAE5',
  }

  if (isAnalyzing || !result) {
    return (
      <div style={{
        background:     'white',
        minHeight:      '100vh',
        display:        'flex',
        alignItems:     'center',
        justifyContent: 'center',
        fontFamily:     '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>⚡</div>
          <div style={{
            fontSize:   '20px',
            fontWeight: 700,
            color:      '#111827',
            marginBottom: '8px'
          }}>
            Running SNN Analysis...
          </div>
          <div style={{ fontSize: '14px', color: '#6B7280', marginBottom: '24px' }}>
            Processing {selectedModality?.toUpperCase()} signal window {selectedWindowIndex}
          </div>
          <div style={{ fontSize: '13px', color: '#9CA3AF' }}>
            SNN → CNN → LSTM → Transformer → XAI
          </div>
        </div>
      </div>
    )
  }

  const snn      = result.snn
  const clinical = result.clinical
  const severity = clinical.severity
  const sColor   = severityColors[severity] || '#6B7280'
  const sBg      = severityBg[severity] || '#F9FAFB'

  return (
    <div style={{
      background:  '#F9FAFB',
      minHeight:   '100vh',
      fontFamily:  '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    }}>

      {/* Navbar */}
      <nav style={{
        background:     'white',
        borderBottom:   '1px solid #E5E7EB',
        padding:        '0 32px',
        display:        'flex',
        alignItems:     'center',
        justifyContent: 'space-between',
        height:         '64px',
        position:       'sticky',
        top:            0,
        zIndex:         100,
        boxShadow:      '0 1px 3px rgba(0,0,0,0.05)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button
            onClick={() => router.push('/')}
            style={{
              padding:      '6px 12px',
              background:   'white',
              border:       '1px solid #E5E7EB',
              borderRadius: '6px',
              fontSize:     '13px',
              color:        '#374151',
              cursor:       'pointer'
            }}
          >
            ← Back
          </button>
          <div style={{
            width:          '32px',
            height:         '32px',
            borderRadius:   '8px',
            background:     'linear-gradient(135deg, #6366F1, #8B5CF6)',
            display:        'flex',
            alignItems:     'center',
            justifyContent: 'center',
            fontSize:       '16px'
          }}>
            ⚡
          </div>
          <div>
            <div style={{ fontSize: '15px', fontWeight: 700, color: '#111827' }}>
              Analysis Results
            </div>
            <div style={{ fontSize: '11px', color: '#6B7280' }}>
              {result.modality.toUpperCase()} · Window {result.window_index} ·
              True Label: {result.true_label_text}
            </div>
          </div>
        </div>

        {/* Severity badge */}
        <div style={{
          background:   sBg,
          border:       `1px solid ${sColor}`,
          borderRadius: '100px',
          padding:      '6px 16px',
          fontSize:     '13px',
          fontWeight:   700,
          color:        sColor
        }}>
          {severity.toUpperCase()} — SNN: {(snn.confidence * 100).toFixed(1)}%
        </div>
      </nav>

      <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '24px' }}>

        {/* Tabs */}
        <div style={{
          display:      'flex',
          gap:          '8px',
          marginBottom: '24px',
          overflowX:    'auto',
          paddingBottom: '4px'
        }}>
          {tabs.map(tab => (
            <TabButton
              key={tab.id}
              id={tab.id}
              label={tab.label}
              active={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
            />
          ))}
        </div>

        {/* ── TAB 1: Clinical Report ─────────────────────────────────────────── */}
        {activeTab === 'clinical' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

            {/* Main finding card */}
            <div style={{
              background:   'white',
              border:       `1px solid ${sColor}`,
              borderRadius: '16px',
              padding:      '24px',
              boxShadow:    `0 0 0 4px ${sColor}15`
            }}>
              <div style={{
                display:      'flex',
                alignItems:   'flex-start',
                gap:          '16px',
                marginBottom: '20px'
              }}>
                <div style={{
                  width:          '48px',
                  height:         '48px',
                  borderRadius:   '12px',
                  background:     sBg,
                  display:        'flex',
                  alignItems:     'center',
                  justifyContent: 'center',
                  fontSize:       '24px',
                  flexShrink:     0
                }}>
                  {severity === 'normal' ? '✅' :
                   severity === 'medium' ? '⚠️' : '🚨'}
                </div>
                <div>
                  <div style={{
                    fontSize:   '20px',
                    fontWeight: 800,
                    color:      sColor,
                    marginBottom: '4px'
                  }}>
                    {snn.label}
                  </div>
                  <div style={{ fontSize: '13px', color: '#6B7280' }}>
                    {result.modality.toUpperCase()} Signal ·
                    Window {result.window_index} ·
                    SNN Confidence: {(snn.confidence * 100).toFixed(2)}%
                  </div>
                </div>
                <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
                  <div style={{
                    fontSize:   '11px',
                    color:      '#9CA3AF',
                    marginBottom: '2px'
                  }}>
                    ICD Code
                  </div>
                  <div style={{
                    fontSize:   '12px',
                    fontWeight: 600,
                    color:      '#374151'
                  }}>
                    {clinical.icd_code}
                  </div>
                </div>
              </div>

              {/* Clinical finding */}
              <div style={{
                background:   '#F9FAFB',
                borderRadius: '10px',
                padding:      '16px',
                marginBottom: '20px',
                borderLeft:   `4px solid ${sColor}`
              }}>
                <div style={{
                  fontSize:     '12px',
                  fontWeight:   700,
                  color:        '#6B7280',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  marginBottom: '8px'
                }}>
                  Clinical Finding
                </div>
                <div style={{
                  fontSize:   '14px',
                  color:      '#111827',
                  lineHeight: '1.6'
                }}>
                  {clinical.finding}
                </div>
              </div>

              {/* Clinical Priority Panel */}
              <ClinicalPriorityPanel
                finding={clinical.finding}
                actions={clinical.actions}
                severity={clinical.severity}
                isAnomaly={snn.is_anomaly}
                modality={result.modality}
              />
            </div>

            {/* SNN metrics */}
            <div style={{
              display:             'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap:                 '12px'
            }}>
              {[
                {
                  label: 'SNN Confidence',
                  value: `${(snn.confidence * 100).toFixed(2)}%`,
                  sub:   'Model certainty',
                  color: sColor
                },
                {
                  label: 'Total Spikes',
                  value: Math.round(snn.total_spikes).toLocaleString(),
                  sub:   'Neuron firing events',
                  color: '#6366F1'
                },
                {
                  label: 'Spike Sparsity',
                  value: `${(snn.sparsity * 100).toFixed(1)}%`,
                  sub:   'Inactive neurons',
                  color: '#10B981'
                },
                {
                  label: 'Inference Time',
                  value: `${snn.latency_ms.toFixed(2)}ms`,
                  sub:   'SNN forward pass',
                  color: '#F59E0B'
                },
              ].map(({ label, value, sub, color }) => (
                <div key={label} style={{
                  background:   'white',
                  border:       '1px solid #E5E7EB',
                  borderRadius: '12px',
                  padding:      '16px',
                  boxShadow:    '0 1px 3px rgba(0,0,0,0.05)'
                }}>
                  <div style={{
                    fontSize:   '11px',
                    color:      '#9CA3AF',
                    marginBottom: '4px'
                  }}>
                    {label}
                  </div>
                  <div style={{
                    fontSize:   '22px',
                    fontWeight: 800,
                    color
                  }}>
                    {value}
                  </div>
                  <div style={{
                    fontSize:  '11px',
                    color:     '#9CA3AF',
                    marginTop: '2px'
                  }}>
                    {sub}
                  </div>
                </div>
              ))}
            </div>

            {/* True label verification */}
            <div style={{
              background:   'white',
              border:       '1px solid #E5E7EB',
              borderRadius: '12px',
              padding:      '16px 20px',
              display:      'flex',
              alignItems:   'center',
              gap:          '16px'
            }}>
              <div style={{ fontSize: '20px' }}>🏷️</div>
              <div>
                <div style={{
                  fontSize:   '12px',
                  color:      '#9CA3AF',
                  marginBottom: '2px'
                }}>
                  Ground Truth (Dataset Label)
                </div>
                <div style={{
                  fontSize:   '15px',
                  fontWeight: 700,
                  color:      '#111827'
                }}>
                  {result.true_label_text}
                </div>
              </div>
              <div style={{ marginLeft: 'auto' }}>
                <div style={{
                  background:   snn.prediction === result.true_label
                    ? '#D1FAE5' : '#FEE2E2',
                  border:       `1px solid ${
                    snn.prediction === result.true_label
                      ? '#6EE7B7' : '#FCA5A5'
                  }`,
                  borderRadius: '8px',
                  padding:      '8px 16px',
                  fontSize:     '13px',
                  fontWeight:   700,
                  color:        snn.prediction === result.true_label
                    ? '#065F46' : '#991B1B'
                }}>
                  {snn.prediction === result.true_label
                    ? '✓ SNN Prediction Correct'
                    : '✗ SNN Prediction Incorrect'}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── TAB 2: Signal Analysis ─────────────────────────────────────────── */}
        {activeTab === 'signal' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div style={{
              background:   'white',
              border:       '1px solid #E5E7EB',
              borderRadius: '16px',
              padding:      '24px'
            }}>
              <h3 style={{
                fontSize:     '16px',
                fontWeight:   700,
                color:        '#111827',
                marginBottom: '8px',
                margin:       '0 0 8px 0'
              }}>
                {result.modality.toUpperCase()} Signal with Spike Train
              </h3>
              <p style={{
  fontSize:     '13px',
  color:        '#6B7280',
  marginBottom: '16px',
  margin:       '0 0 16px 0'
}}>
  Top: Raw biosignal with XAI attribution heatmap overlay (red = high attribution).
  Bottom: Spike train generated by delta modulation encoder.
</p>
<SignalCanvas
  signal={result.signal}
  spikeTrain={result.spike_train}
  attribution={result.attribution}
  isAnomaly={snn.is_anomaly}
  modality={result.modality}
  height={200}
/>
            </div>

            {/* Signal stats */}
            <div style={{
              display:             'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap:                 '12px'
            }}>
              {[
                {
                  label: 'Signal Samples',
                  value: result.signal.length,
                  color: '#6366F1'
                },
                {
                  label: 'Sample Rate',
                  value: `${result.sample_rate} Hz`,
                  color: '#6366F1'
                },
                {
                  label: 'Spike Count',
                  value: result.spike_count,
                  color: '#6366F1'
                },
                {
                  label: 'Duration',
                  value: `${(result.signal.length / result.sample_rate).toFixed(2)}s`,
                  color: '#6366F1'
                },
              ].map(({ label, value, color }) => (
                <div key={label} style={{
                  background:   'white',
                  border:       '1px solid #E5E7EB',
                  borderRadius: '12px',
                  padding:      '16px'
                }}>
                  <div style={{ fontSize: '11px', color: '#9CA3AF' }}>
                    {label}
                  </div>
                  <div style={{
                    fontSize:   '20px',
                    fontWeight: 800,
                    color,
                    marginTop:  '4px'
                  }}>
                    {value}
                  </div>
                </div>
              ))}
            </div>

            {/* Spike encoding explanation */}
            <div style={{
              background:   '#EEF2FF',
              border:       '1px solid #C7D2FE',
              borderRadius: '12px',
              padding:      '16px 20px'
            }}>
              <div style={{
                fontSize:     '13px',
                fontWeight:   700,
                color:        '#4338CA',
                marginBottom: '6px'
              }}>
                ⚡ How Spike Encoding Works
              </div>
              <div style={{
                fontSize:   '13px',
                color:      '#4338CA',
                lineHeight: '1.6'
              }}>
                The raw {result.modality.toUpperCase()} signal is encoded into
                spike trains using <strong>temporal delta modulation</strong>.
                A spike fires when the signal changes by more than the threshold (0.05).
                This converts continuous biosignals into discrete binary events —
                the language of spiking neural networks.
                Total spikes generated: <strong>{result.spike_count}</strong> out
                of {result.signal.length} samples
                ({((result.spike_count / result.signal.length) * 100).toFixed(1)}% firing rate).
              </div>
            </div>
          </div>
        )}

        {/* ── TAB 3: XAI Explanation ─────────────────────────────────────────── */}
        {activeTab === 'xai' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

            {/* Attribution explanation */}
            <div style={{
              background:   'white',
              border:       '1px solid #E5E7EB',
              borderRadius: '16px',
              padding:      '24px'
            }}>
              <h3 style={{
                fontSize:     '16px',
                fontWeight:   700,
                color:        '#111827',
                margin:       '0 0 8px 0'
              }}>
                Why Did the SNN Make This Decision?
              </h3>
              <p style={{
                fontSize:     '13px',
                color:        '#6B7280',
                margin:       '0 0 16px 0'
              }}>
                The heatmap shows which timesteps contributed most to the
                SNN&apos;s decision. Red = high attribution (most influential).
                Blue = low attribution (less influential).
              </p>

              {/* Attribution heatmap */}
              <AttributionHeatmap
                attribution={result.attribution}
                peakIndex={result.attribution_peak}
                isAnomaly={snn.is_anomaly}
              />

              {/* Peak region explanation */}
              <div style={{
                marginTop:    '16px',
                background:   snn.is_anomaly ? '#FEF2F2' : '#F0FDF4',
                border:       `1px solid ${snn.is_anomaly ? '#FCA5A5' : '#86EFAC'}`,
                borderRadius: '10px',
                padding:      '14px 16px'
              }}>
                <div style={{
                  fontSize:     '13px',
                  fontWeight:   700,
                  color:        snn.is_anomaly ? '#991B1B' : '#166534',
                  marginBottom: '6px'
                }}>
                  {snn.is_anomaly ? '⚠ Anomaly Region Identified' : '✓ Normal Pattern Confirmed'}
                </div>
                <div style={{
                  fontSize:   '13px',
                  color:      snn.is_anomaly ? '#7F1D1D' : '#14532D',
                  lineHeight: '1.6'
                }}>
                  {snn.is_anomaly
                    ? `Peak neural attribution detected at timestep ${result.attribution_peak} 
                       (${((result.attribution_peak / result.attribution.length) * 100).toFixed(0)}% 
                       into the signal window). The SNN neurons showed maximum firing activity 
                       at this region, indicating the most diagnostically significant portion 
                       of the ${result.modality.toUpperCase()} signal. 
                       Confidence: ${(snn.confidence * 100).toFixed(2)}%.`
                    : `No significant anomaly region identified. Attribution is uniformly 
                       distributed across the signal window, consistent with normal 
                       ${result.modality.toUpperCase()} patterns. 
                       SNN confidence in normal classification: ${(snn.conf_normal * 100).toFixed(2)}%.`
                  }
                </div>
              </div>
            </div>

            {/* Confidence breakdown */}
            <div style={{
              background:   'white',
              border:       '1px solid #E5E7EB',
              borderRadius: '16px',
              padding:      '24px'
            }}>
              <h3 style={{
                fontSize:     '16px',
                fontWeight:   700,
                color:        '#111827',
                margin:       '0 0 16px 0'
              }}>
                SNN Output Confidence Breakdown
              </h3>
              <ConfidenceBar
                label="Normal"
                value={snn.conf_normal}
                color="#10B981"
              />
              <ConfidenceBar
                label={result.modality === 'ecg' ? 'Arrhythmia' :
                       result.modality === 'eeg' ? 'Seizure' : 'Anomaly'}
                value={snn.conf_anomaly}
                color="#EF4444"
              />

              <div style={{
                marginTop:    '16px',
                padding:      '12px 16px',
                background:   '#F9FAFB',
                borderRadius: '8px',
                fontSize:     '13px',
                color:        '#374151'
              }}>
                <strong>Interpretation:</strong>{' '}
                {snn.conf_anomaly > snn.conf_normal
                  ? `The SNN assigns ${(snn.conf_anomaly * 100).toFixed(2)}% probability 
                     to anomaly vs ${(snn.conf_normal * 100).toFixed(2)}% to normal. 
                     The network is confident this signal contains pathological patterns.`
                  : `The SNN assigns ${(snn.conf_normal * 100).toFixed(2)}% probability 
                     to normal vs ${(snn.conf_anomaly * 100).toFixed(2)}% to anomaly. 
                     The signal characteristics are consistent with normal physiology.`
                }
              </div>
            </div>

            {/* SNN layer activity */}
            <div style={{
              background:   'white',
              border:       '1px solid #E5E7EB',
              borderRadius: '16px',
              padding:      '24px'
            }}>
              <h3 style={{
                fontSize:     '16px',
                fontWeight:   700,
                color:        '#111827',
                margin:       '0 0 16px 0'
              }}>
                SNN Layer-by-Layer Spike Activity
              </h3>
              <LayerBar
                label="Layer 1 — Spike Encoding"
                activity={snn.layer1_activity}
                color="#6366F1"
              />
              <LayerBar
                label="Layer 2 — Feature Extraction"
                activity={snn.layer2_activity}
                color="#8B5CF6"
              />
              <LayerBar
                label="Layer 3 — Pattern Recognition"
                activity={snn.layer3_activity}
                color="#A78BFA"
              />
              <div style={{
                marginTop:  '12px',
                fontSize:   '12px',
                color:      '#9CA3AF',
                lineHeight: '1.5'
              }}>
                Each bar shows the average spike activity per layer over all timesteps.
                Higher activity = more neurons firing = stronger signal features detected.
                Total spikes: {Math.round(snn.total_spikes).toLocaleString()} ·
                Sparsity: {(snn.sparsity * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        )}

        {/* ── TAB 4: Model Comparison ────────────────────────────────────────── */}
        {activeTab === 'comparison' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

            {/* Same signal, all models */}
            <div style={{
              background:   'white',
              border:       '1px solid #E5E7EB',
              borderRadius: '16px',
              padding:      '24px'
            }}>
              <h3 style={{
                fontSize:     '16px',
                fontWeight:   700,
                color:        '#111827',
                margin:       '0 0 4px 0'
              }}>
                Same Signal — All 4 Models
              </h3>
              <p style={{
                fontSize:   '13px',
                color:      '#6B7280',
                margin:     '0 0 20px 0'
              }}>
                All models ran on the exact same {result.modality.toUpperCase()} signal
                (Window {result.window_index}). True label: {result.true_label_text}
              </p>

              <div style={{
                display:             'grid',
                gridTemplateColumns: 'repeat(2, 1fr)',
                gap:                 '12px'
              }}>
                {/* SNN — Hero */}
                <div style={{
                  background:   '#EEF2FF',
                  border:       '2px solid #6366F1',
                  borderRadius: '12px',
                  padding:      '20px',
                  gridColumn:   'span 2'
                }}>
                  <div style={{
                    display:        'flex',
                    alignItems:     'center',
                    justifyContent: 'space-between',
                    marginBottom:   '12px'
                  }}>
                    <div>
                      <div style={{
                        fontSize:   '16px',
                        fontWeight: 800,
                        color:      '#4338CA'
                      }}>
                        ⚡ SNN — Leaky Integrate-and-Fire
                      </div>
                      <div style={{ fontSize: '12px', color: '#6366F1' }}>
                        SpikingJelly + PyTorch · {snn.model_type}
                      </div>
                    </div>
                    <div style={{
                      background:   snn.prediction === result.true_label
                        ? '#D1FAE5' : '#FEE2E2',
                      borderRadius: '8px',
                      padding:      '6px 12px',
                      fontSize:     '13px',
                      fontWeight:   700,
                      color:        snn.prediction === result.true_label
                        ? '#065F46' : '#991B1B'
                    }}>
                      {snn.prediction === result.true_label ? '✓ Correct' : '✗ Incorrect'}
                    </div>
                  </div>
                  <div style={{
                    display:             'grid',
                    gridTemplateColumns: 'repeat(4, 1fr)',
                    gap:                 '8px'
                  }}>
                    {[
                      { label: 'Prediction',  value: snn.label },
                      { label: 'Confidence',  value: `${(snn.confidence*100).toFixed(1)}%` },
                      { label: 'Energy',      value: `${snn.energy_mj.toFixed(8)} mJ` },
                      { label: 'Latency',     value: `${snn.latency_ms.toFixed(2)}ms` },
                    ].map(({ label, value }) => (
                      <div key={label} style={{
                        background:   'white',
                        borderRadius: '8px',
                        padding:      '8px 12px'
                      }}>
                        <div style={{ fontSize: '10px', color: '#9CA3AF' }}>{label}</div>
                        <div style={{
                          fontSize:   '13px',
                          fontWeight: 700,
                          color:      '#4338CA',
                          marginTop:  '2px'
                        }}>
                          {value}
                        </div>
                      </div>
                    ))}
                  </div>
                  <div style={{
                    marginTop:  '10px',
                    fontSize:   '12px',
                    color:      '#4338CA',
                    fontStyle:  'italic'
                  }}>
                    ✦ SNN provides spike-level explainability — CNN/LSTM/Transformer cannot show which
                    signal regions activated which neurons.
                  </div>
                </div>

                {/* Baselines */}
                {Object.entries(result.baselines).map(([key, baseline]: [string, any]) => (
                  <div key={key} style={{
                    background:   'white',
                    border:       '1px solid #E5E7EB',
                    borderRadius: '12px',
                    padding:      '16px'
                  }}>
                    <div style={{
                      display:        'flex',
                      alignItems:     'center',
                      justifyContent: 'space-between',
                      marginBottom:   '12px'
                    }}>
                      <div style={{
                        fontSize:   '14px',
                        fontWeight: 700,
                        color:      '#374151'
                      }}>
                        {baseline.model}
                      </div>
                      <div style={{
                        background:   baseline.prediction === result.true_label
                          ? '#D1FAE5' : '#FEE2E2',
                        borderRadius: '6px',
                        padding:      '3px 8px',
                        fontSize:     '11px',
                        fontWeight:   700,
                        color:        baseline.prediction === result.true_label
                          ? '#065F46' : '#991B1B'
                      }}>
                        {baseline.prediction === result.true_label
                          ? '✓ Correct' : '✗ Incorrect'}
                      </div>
                    </div>
                    {[
                      { label: 'Prediction', value: baseline.label },
                      { label: 'Confidence', value: `${(baseline.confidence*100).toFixed(1)}%` },
                      { label: 'Energy',     value: `${baseline.energy_mj.toFixed(4)} mJ` },
                      { label: 'Latency',    value: `${baseline.latency_ms.toFixed(2)}ms` },
                      { label: 'Parameters', value: baseline.parameters?.toLocaleString() },
                    ].map(({ label, value }) => (
                      <div key={label} style={{
                        display:        'flex',
                        justifyContent: 'space-between',
                        fontSize:       '12px',
                        padding:        '4px 0',
                        borderBottom:   '1px solid #F3F4F6'
                      }}>
                        <span style={{ color: '#9CA3AF' }}>{label}</span>
                        <span style={{ color: '#111827', fontWeight: 600 }}>
                          {value}
                        </span>
                      </div>
                    ))}
                    <div style={{
                      marginTop:  '8px',
                      fontSize:   '11px',
                      color:      '#D1D5DB',
                      fontStyle:  'italic'
                    }}>
                      No spike-level explainability available
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Energy comparison */}
            <div style={{
              background:   'white',
              border:       '1px solid #E5E7EB',
              borderRadius: '16px',
              padding:      '24px'
            }}>
              <h3 style={{
                fontSize:     '16px',
                fontWeight:   700,
                color:        '#111827',
                margin:       '0 0 16px 0'
              }}>
                Energy Consumption — Why SNN Matters for Clinical Deployment
              </h3>
              {[
                { name: 'SNN (Ours)',  energy: snn.energy_mj,                       color: '#6366F1' },
                { name: 'CNN',        energy: result.baselines.cnn.energy_mj,       color: '#EF4444' },
                { name: 'LSTM',       energy: result.baselines.lstm.energy_mj,      color: '#F59E0B' },
                { name: 'Transformer',energy: result.baselines.transformer.energy_mj, color: '#8B5CF6' },
              ].map(({ name, energy, color }) => {
                const maxE = Math.max(
                  result.baselines.cnn.energy_mj,
                  result.baselines.lstm.energy_mj,
                  result.baselines.transformer.energy_mj
                )
                const pct  = Math.max(0.5, (energy / maxE) * 100)
                return (
                  <div key={name} style={{ marginBottom: '10px' }}>
                    <div style={{
                      display:        'flex',
                      justifyContent: 'space-between',
                      marginBottom:   '4px',
                      fontSize:       '13px'
                    }}>
                      <span style={{ fontWeight: 600, color: '#374151' }}>{name}</span>
                      <span style={{ color, fontWeight: 700 }}>
                        {energy < 0.001
                          ? `${(energy * 1000000).toFixed(4)} µJ`
                          : `${energy.toFixed(4)} mJ`}
                      </span>
                    </div>
                    <div style={{
                      background:   '#F3F4F6',
                      borderRadius: '100px',
                      height:       '10px',
                      overflow:     'hidden'
                    }}>
                      <div style={{
                        width:        `${pct}%`,
                        height:       '100%',
                        background:   color,
                        borderRadius: '100px'
                      }} />
                    </div>
                  </div>
                )
              })}
              <div style={{
                marginTop:    '12px',
                background:   '#EEF2FF',
                borderRadius: '8px',
                padding:      '10px 14px',
                fontSize:     '13px',
                color:        '#4338CA'
              }}>
                💡 For 24/7 continuous patient monitoring, SNN&apos;s energy advantage
                translates directly to longer battery life, lower infrastructure cost,
                and feasibility of edge deployment at the bedside.
              </div>
            </div>
          </div>
        )}

        {/* ── TAB 5: Neural Activity ─────────────────────────────────────────── */}
        {activeTab === 'neural' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div style={{
              background:   'white',
              border:       '1px solid #E5E7EB',
              borderRadius: '16px',
              padding:      '24px'
            }}>
              <h3 style={{
                fontSize:     '16px',
                fontWeight:   700,
                color:        '#111827',
                margin:       '0 0 8px 0'
              }}>
                🧠 3D Neural Twin — SNN Processing This Signal
              </h3>
              <p style={{
                fontSize:   '13px',
                color:      '#6B7280',
                margin:     '0 0 16px 0'
              }}>
                Drag to rotate. Each sphere = one LIF neuron.
                Glowing = firing. Lines = spike propagation.
              </p>
              <NeuralTwinCanvas
                totalSpikes={snn.total_spikes}
                isAnomaly={snn.is_anomaly}
                layer1={snn.layer1_activity}
                layer2={snn.layer2_activity}
                layer3={snn.layer3_activity}
              />
            </div>

            {/* Spike raster */}
            <div style={{
              background:   'white',
              border:       '1px solid #E5E7EB',
              borderRadius: '16px',
              padding:      '24px'
            }}>
              <h3 style={{
                fontSize:     '16px',
                fontWeight:   700,
                color:        '#111827',
                margin:       '0 0 8px 0'
              }}>
                Spike Raster Plot
              </h3>
              <p style={{
                fontSize:   '13px',
                color:      '#6B7280',
                margin:     '0 0 16px 0'
              }}>
                Each vertical line = one neuron firing event.
                Generated from the spike train of this exact signal.
              </p>
              <SpikeRasterCanvas
                spikeTrain={result.spike_train}
                nNeurons={8}
                isAnomaly={snn.is_anomaly}
              />
            </div>
          </div>
        )}

        {/* ── TAB 6: AI Reasoning ────────────────────────────────────────────── */}
        {activeTab === 'reasoning' && (
          <AIReasoningPanel
            attribution={result.attribution}
            attributionPeak={result.attribution_peak}
            modality={result.modality}
            isAnomaly={snn.is_anomaly}
            confidence={snn.confidence}
            sparsity={snn.sparsity}
            totalSpikes={snn.total_spikes}
          />
        )}

  

        {/* ── TAB 8: Decision Timeline ───────────────────────────────────────── */}
        {activeTab === 'timeline' && (
          <DecisionTimeline
            latencyMs={snn.latency_ms}
            isAnomaly={snn.is_anomaly}
            modality={result.modality}
          />
        )}

        {/* ── TAB 9: Doctor Summary ──────────────────────────────────────────── */}
        {activeTab === 'doctor' && (
          <DoctorSummary
            modality={result.modality}
            isAnomaly={snn.is_anomaly}
            finding={clinical.finding}
            severity={clinical.severity}
          />
        )}

        {/* ── TAB 10: Export Report ──────────────────────────────────────────── */}
        {activeTab === 'export' && (
          <ExportReportCenter result={result} />
        )}
      </div>
    </div>
  )
}

// ── Attribution Heatmap ────────────────────────────────────────────────────────
function AttributionHeatmap({
  attribution, peakIndex, isAnomaly
}: {
  attribution: number[], peakIndex: number, isAnomaly: boolean
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || attribution.length === 0) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const w = canvas.width
    const h = canvas.height
    const n = attribution.length

    ctx.fillStyle = '#FFFFFF'
    ctx.fillRect(0, 0, w, h)

    // Draw attribution bars
    attribution.forEach((val, i) => {
      const x    = (i / n) * w
      const barW = w / n + 1
      const r    = isAnomaly ? Math.round(239 * val + 156 * (1-val)) : 99
      const g    = isAnomaly ? Math.round(68 * val + 102 * (1-val)) : 102
      const b    = isAnomaly ? Math.round(68 * val + 241 * (1-val)) : 241
      ctx.fillStyle = `rgba(${r},${g},${b},${Math.max(0.1, val)})`
      ctx.fillRect(x, 0, barW, h)
    })

    // Peak marker
    if (peakIndex >= 0) {
      const peakX = (peakIndex / n) * w
      ctx.strokeStyle = '#EF4444'
      ctx.lineWidth   = 2
      ctx.setLineDash([5, 3])
      ctx.beginPath()
      ctx.moveTo(peakX, 0)
      ctx.lineTo(peakX, h)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = '#EF4444'
      ctx.font      = 'bold 11px sans-serif'
      ctx.fillText(`Peak at ${peakIndex}`, peakX + 4, 14)
    }

    // Axis label
    ctx.fillStyle = '#9CA3AF'
    ctx.font      = '10px sans-serif'
    ctx.fillText('← Timestep →', w / 2 - 30, h - 4)

    // Border
    ctx.strokeStyle = '#E5E7EB'
    ctx.lineWidth   = 1
    ctx.strokeRect(0.5, 0.5, w-1, h-1)

  }, [attribution, peakIndex, isAnomaly])

  return (
    <canvas
      ref={canvasRef}
      width={900}
      height={80}
      style={{
        width:        '100%',
        height:       '80px',
        display:      'block',
        borderRadius: '8px'
      }}
    />
  )
}

// ── Spike Raster Canvas ────────────────────────────────────────────────────────
function SpikeRasterCanvas({
  spikeTrain, nNeurons, isAnomaly
}: {
  spikeTrain: number[], nNeurons: number, isAnomaly: boolean
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || spikeTrain.length === 0) return
    const ctx    = canvas.getContext('2d')
    if (!ctx) return

    const w      = canvas.width
    const h      = canvas.height
    const laneH  = h / nNeurons
    const n      = spikeTrain.length

    ctx.fillStyle = '#FAFAFA'
    ctx.fillRect(0, 0, w, h)

    // Grid
    ctx.strokeStyle = '#F3F4F6'
    ctx.lineWidth   = 0.5
    for (let t = 0; t < n; t += 20) {
      const x = (t / n) * w
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke()
    }

    // Distribute spikes across neurons
    for (let neuron = 0; neuron < nNeurons; neuron++) {
      const y = neuron * laneH

      // Lane separator
      ctx.strokeStyle = '#E5E7EB'
      ctx.lineWidth   = 0.5
      ctx.beginPath()
      ctx.moveTo(0, y + laneH)
      ctx.lineTo(w, y + laneH)
      ctx.stroke()

      // Neuron label
      ctx.fillStyle = '#9CA3AF'
      ctx.font      = '9px sans-serif'
      ctx.fillText(`N${neuron+1}`, 3, y + laneH * 0.65)

      // Spikes for this neuron (every nNeurons-th spike)
      spikeTrain.forEach((spike, t) => {
        if (spike > 0.5 && t % nNeurons === neuron) {
          const x = (t / n) * w
          ctx.strokeStyle = isAnomaly ? '#EF4444' : '#6366F1'
          ctx.lineWidth   = 1.5
          ctx.globalAlpha = 0.8
          ctx.beginPath()
          ctx.moveTo(x, y + 2)
          ctx.lineTo(x, y + laneH - 2)
          ctx.stroke()
          ctx.globalAlpha = 1
        }
      })
    }

    ctx.strokeStyle = '#E5E7EB'
    ctx.lineWidth   = 1
    ctx.strokeRect(0.5, 0.5, w-1, h-1)

  }, [spikeTrain, nNeurons, isAnomaly])

  return (
    <canvas
      ref={canvasRef}
      width={900}
      height={160}
      style={{
        width:        '100%',
        height:       '160px',
        display:      'block',
        borderRadius: '8px',
        border:       '1px solid #E5E7EB'
      }}
    />
  )
}

// ── Neural Twin Canvas (Three.js) ──────────────────────────────────────────────
function NeuralTwinCanvas({
  totalSpikes, isAnomaly, layer1, layer2, layer3
}: {
  totalSpikes: number, isAnomaly: boolean,
  layer1: number[], layer2: number[], layer3: number[]
}) {
  const mountRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!mountRef.current) return

    // Clear previous canvas
    while (mountRef.current.firstChild) {
      mountRef.current.removeChild(mountRef.current.firstChild)
    }

    let animId: number

    // Load Three.js
    const script    = document.createElement('script')
    script.src      = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js'
    script.onload   = () => initScene()
    document.head.appendChild(script)

    function initScene() {
      if (!mountRef.current) return
      const THREE = (window as any).THREE

      const W = mountRef.current.clientWidth
      const H = 420

      // Scene
      const scene    = new THREE.Scene()
      scene.background = new THREE.Color(0x0D1117)
      scene.fog        = new THREE.Fog(0x0D1117, 30, 60)

      // Camera
      const camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 1000)
      camera.position.set(0, 2, 24)

      // Renderer
      const renderer = new THREE.WebGLRenderer({ antialias: true })
      renderer.setSize(W, H)
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
      renderer.shadowMap.enabled = true
      mountRef.current!.appendChild(renderer.domElement)

      // Lights
      const ambient = new THREE.AmbientLight(0xffffff, 0.5)
      scene.add(ambient)
      const dirLight = new THREE.DirectionalLight(0xffffff, 1.0)
      dirLight.position.set(10, 20, 10)
      scene.add(dirLight)

      // Point lights for atmosphere
      const pointLight1 = new THREE.PointLight(
        isAnomaly ? 0xEF4444 : 0x6366F1, 2, 20
      )
      pointLight1.position.set(-8, 0, 0)
      scene.add(pointLight1)

      const pointLight2 = new THREE.PointLight(0x00D4AA, 1.5, 20)
      pointLight2.position.set(8, 0, 0)
      scene.add(pointLight2)

      // Layer config
      const LAYERS = [
        { count: 10, x: -9,  color: isAnomaly ? 0xFF6B6B : 0x60A5FA, name: 'Input'    },
        { count: 16, x: -3,  color: 0xA78BFA,                         name: 'Hidden 1' },
        { count: 16, x:  3,  color: 0x34D399,                         name: 'Hidden 2' },
        { count:  6, x:  9,  color: isAnomaly ? 0xFF4444 : 0x60A5FA,  name: 'Output'   },
      ]

      const neurons: any[]     = []
      const connections: any[] = []
      const spikeLines: any[]  = []

      // Create neurons
      LAYERS.forEach((layer, li) => {
        const cols = Math.ceil(Math.sqrt(layer.count))
        for (let i = 0; i < layer.count; i++) {
          const row = Math.floor(i / cols)
          const col = i % cols
          const y   = (row - (cols - 1) / 2) * 2.0
          const z   = (col - (cols - 1) / 2) * 2.0

          // Use BoxGeometry instead of SphereGeometry
          // to avoid CapsuleGeometry issues in r128
          const geo = new THREE.SphereGeometry(0.35, 16, 16)
          const mat = new THREE.MeshPhongMaterial({
            color:             layer.color,
            emissive:          layer.color,
            emissiveIntensity: 0.2,
            shininess:         100,
          })
          const mesh = new THREE.Mesh(geo, mat)
          mesh.position.set(layer.x, y, z)
          scene.add(mesh)

          neurons.push({
            mesh,
            layer:     li,
            baseColor: layer.color,
            firing:    false,
            fireTime:  0,
            fireIntensity: 0,
          })
        }
      })

      // Connections between layers
      neurons.forEach((n) => {
        if (n.layer >= LAYERS.length - 1) return
        const nextLayer = neurons.filter(nn => nn.layer === n.layer + 1)
        nextLayer.slice(0, 3).forEach(n2 => {
          const points  = [n.mesh.position, n2.mesh.position]
          const geo     = new THREE.BufferGeometry().setFromPoints(points)
          const mat     = new THREE.LineBasicMaterial({
            color:       0x374151,
            transparent: true,
            opacity:     0.25,
          })
          const line = new THREE.Line(geo, mat)
          scene.add(line)
          connections.push({ line, from: n, to: n2 })
        })
      })

      // Layer labels as sprites
      LAYERS.forEach(layer => {
        const canvas  = document.createElement('canvas')
        canvas.width  = 256
        canvas.height = 64
        const ctx     = canvas.getContext('2d')!
        ctx.fillStyle = 'transparent'
        ctx.clearRect(0, 0, 256, 64)
        ctx.fillStyle  = '#9CA3AF'
        ctx.font       = 'bold 28px Arial'
        ctx.textAlign  = 'center'
        ctx.fillText(layer.name, 128, 40)

        const texture  = new THREE.CanvasTexture(canvas)
        const spriteMat = new THREE.SpriteMaterial({
          map:         texture,
          transparent: true,
          opacity:     0.7,
        })
        const sprite = new THREE.Sprite(spriteMat)
        sprite.position.set(layer.x, -5.5, 0)
        sprite.scale.set(4, 1, 1)
        scene.add(sprite)
      })

      // Animation state
      let frame       = 0
      let rotY        = 0
      let isDragging  = false
      let prevMouseX  = 0
      let autoRotate  = true

      // Mouse interaction
      renderer.domElement.addEventListener('mousedown', (e: MouseEvent) => {
        isDragging = true
        autoRotate = false
        prevMouseX = e.clientX
      })
      renderer.domElement.addEventListener('mousemove', (e: MouseEvent) => {
        if (!isDragging) return
        rotY      += (e.clientX - prevMouseX) * 0.01
        prevMouseX = e.clientX
      })
      renderer.domElement.addEventListener('mouseup',    () => isDragging = false)
      renderer.domElement.addEventListener('mouseleave', () => isDragging = false)

      // Spike trigger
      function triggerSpike() {
        const inputNeurons = neurons.filter(n => n.layer === 0)
        const n = inputNeurons[
          Math.floor(Math.random() * inputNeurons.length)
        ]
        n.firing       = true
        n.fireTime     = frame
        n.fireIntensity = 1.0
      }

      // Spike propagation
      function propagate() {
        neurons.forEach(n => {
          if (!n.firing) {
            // Decay
            n.fireIntensity = Math.max(0, n.fireIntensity - 0.05)
            n.mesh.material.emissiveIntensity = 0.2 + n.fireIntensity * 0.3
            n.mesh.scale.setScalar(1.0 + n.fireIntensity * 0.3)
            return
          }

          const age = frame - n.fireTime

          // Glow effect
          n.fireIntensity = Math.max(0, 1.0 - age * 0.08)
          n.mesh.material.emissiveIntensity = 0.2 + n.fireIntensity * 2.0
          n.mesh.scale.setScalar(1.0 + n.fireIntensity * 0.8)

          // Propagate after delay
          if (age > 8 && n.layer < LAYERS.length - 1) {
            const nextLayer  = neurons.filter(nn => nn.layer === n.layer + 1)
            const targets    = nextLayer
              .sort(() => Math.random() - 0.5)
              .slice(0, 2 + Math.floor(Math.random() * 2))

            targets.forEach(t => {
              t.firing       = true
              t.fireTime     = frame
              t.fireIntensity = 1.0

              // Spike line
              const geo = new THREE.BufferGeometry().setFromPoints([
                n.mesh.position.clone(),
                t.mesh.position.clone()
              ])
              const mat = new THREE.LineBasicMaterial({
                color:       LAYERS[n.layer].color,
                transparent: true,
                opacity:     0.9,
                linewidth:   2,
              })
              const line = new THREE.Line(geo, mat)
              scene.add(line)
              spikeLines.push({ line, born: frame })
            })

            n.firing = false
          }

          if (age > 20) n.firing = false
        })

        // Cleanup old spike lines
        for (let i = spikeLines.length - 1; i >= 0; i--) {
          const age = frame - spikeLines[i].born
          spikeLines[i].line.material.opacity = Math.max(0, 1.0 - age * 0.08)
          if (age > 14) {
            scene.remove(spikeLines[i].line)
            spikeLines[i].line.geometry.dispose()
            spikeLines.splice(i, 1)
          }
        }
      }

      // Spike rate based on real data
      // Higher total spikes = faster firing rate
      const normalizedRate = Math.min(
        0.12, Math.max(0.03, totalSpikes / 5000)
      )

      // Animation loop
      function animate() {
        animId = requestAnimationFrame(animate)
        frame++

        // Auto rotation
        if (autoRotate) rotY += 0.004

        camera.position.x = 24 * Math.sin(rotY)
        camera.position.z = 24 * Math.cos(rotY)
        camera.lookAt(0, 0, 0)

        // Trigger spikes
        if (Math.random() < normalizedRate) triggerSpike()

        // Propagate
        propagate()

        // Pulse point lights
        pointLight1.intensity = 1.5 + Math.sin(frame * 0.05) * 0.5
        pointLight2.intensity = 1.0 + Math.cos(frame * 0.07) * 0.3

        renderer.render(scene, camera)
      }

      animate()
    }

    return () => {
      if (animId) cancelAnimationFrame(animId)
      // Remove script
      const scripts = document.querySelectorAll(
        'script[src*="three.min.js"]'
      )
      scripts.forEach(s => s.remove())
    }
  }, [totalSpikes, isAnomaly])

  return (
    <div
      ref={mountRef}
      style={{
        width:        '100%',
        height:       '420px',
        borderRadius: '12px',
        overflow:     'hidden',
        border:       `1px solid ${isAnomaly ? '#FCA5A5' : '#E5E7EB'}`,
        cursor:       'grab',
        background:   '#0D1117',
      }}
    />
  )
}