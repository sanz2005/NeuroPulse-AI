'use client'

import { useEffect, useRef } from 'react'

interface XAIPanelProps {
  attribution:  number[]
  signal:       number[]
  modality:     string
  label:        string
  confidence:   number
  isAnomaly:    boolean
}

export default function XAIPanel({
  attribution,
  signal,
  modality,
  label,
  confidence,
  isAnomaly
}: XAIPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const COLORS: Record<string, string> = {
    ecg: '#00D4AA',
    eeg: '#7F77DD',
    emg: '#F59E0B',
  }
  const color = COLORS[modality] || '#00D4AA'

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !attribution.length) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const w = canvas.width
    const h = canvas.height

    ctx.clearRect(0, 0, w, h)
    ctx.fillStyle = '#161B22'
    ctx.fillRect(0, 0, w, h)

    const n = attribution.length

    // Draw attribution heatmap bars
    attribution.forEach((val, i) => {
      const x     = (i / n) * w
      const barW  = w / n + 1
      const alpha = Math.max(0, Math.min(1, val))

      // Color: green → red based on attribution strength
      if (isAnomaly) {
        ctx.fillStyle = `rgba(239, 68, 68, ${alpha * 0.7})`
      } else {
        ctx.fillStyle = `rgba(0, 212, 170, ${alpha * 0.5})`
      }
      ctx.fillRect(x, 0, barW, h)
    })

    // Draw signal overlay
    if (signal.length > 0) {
      const min   = Math.min(...signal)
      const max   = Math.max(...signal)
      const range = max - min || 1

      ctx.strokeStyle = isAnomaly ? '#EF4444' : color
      ctx.lineWidth   = 1.5
      ctx.beginPath()

      signal.forEach((val, i) => {
        const x = (i / signal.length) * w
        const y = h - ((val - min) / range) * h * 0.75 - h * 0.1
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      })
      ctx.stroke()
    }

    // Highlight peak attribution region
    const maxAttr  = Math.max(...attribution)
    const peakIdx  = attribution.indexOf(maxAttr)
    const peakX    = (peakIdx / n) * w
    const peakW    = w / n * 3

    ctx.strokeStyle = '#EF4444'
    ctx.lineWidth   = 1.5
    ctx.setLineDash([3, 3])
    ctx.beginPath()
    ctx.rect(peakX - peakW/2, 2, peakW, h - 4)
    ctx.stroke()
    ctx.setLineDash([])

    // Peak label
    ctx.fillStyle = '#EF4444'
    ctx.font      = '9px monospace'
    ctx.fillText('peak', peakX - 10, 10)

  }, [attribution, signal, isAnomaly, color, modality])

  // Generate plain English explanation
  const getExplanation = () => {
    if (!attribution.length) return 'Analyzing...'

    const maxAttr  = Math.max(...attribution)
    const peakIdx  = attribution.indexOf(maxAttr)
    const peakPct  = ((peakIdx / attribution.length) * 100).toFixed(0)

    if (!isAnomaly) {
      return `All ${modality.toUpperCase()} signal features within normal range. No anomalous spike patterns detected. Confidence: ${(confidence * 100).toFixed(1)}%`
    }

    const explanations: Record<string, string> = {
      ecg: `Abnormal spike activity detected at ${peakPct}% of signal window. The QRS complex shows irregular neuromorphic firing pattern suggesting arrhythmia. SNN confidence: ${(confidence * 100).toFixed(1)}%`,
      eeg: `Elevated spike density detected at ${peakPct}% of EEG window. Abnormal neural synchronization pattern detected across channels suggesting potential seizure activity. SNN confidence: ${(confidence * 100).toFixed(1)}%`,
      emg: `Irregular muscle activation pattern detected at ${peakPct}% of EMG window. Abnormal motor unit firing rate suggests muscle anomaly or fatigue. SNN confidence: ${(confidence * 100).toFixed(1)}%`,
    }

    return explanations[modality] || `Anomaly detected at timestep ${peakIdx}. Confidence: ${(confidence * 100).toFixed(1)}%`
  }

  return (
    <div style={{
      background:   '#161B22',
      border:       `0.5px solid ${isAnomaly ? '#EF4444' : '#30363D'}`,
      borderRadius: '8px',
      padding:      '10px 14px',
    }}>
      {/* Header */}
      <div style={{
        display:        'flex',
        justifyContent: 'space-between',
        alignItems:     'center',
        marginBottom:   '8px'
      }}>
        <span style={{
          fontSize:   '11px',
          fontWeight: 500,
          color:      color
        }}>
          👁 XAI — {modality.toUpperCase()} Spike Attribution
        </span>
        <span style={{
          fontSize:     '10px',
          padding:      '2px 8px',
          borderRadius: '4px',
          background:   isAnomaly
            ? 'rgba(239,68,68,0.15)'
            : 'rgba(0,212,170,0.15)',
          color: isAnomaly ? '#EF4444' : '#00D4AA',
          fontWeight: 600
        }}>
          {label?.toUpperCase() || 'NORMAL'}
        </span>
      </div>

      {/* Heatmap Canvas */}
      <canvas
        ref={canvasRef}
        width={500}
        height={60}
        style={{
          width:        '100%',
          height:       '60px',
          display:      'block',
          borderRadius: '4px',
          marginBottom: '8px'
        }}
      />

      {/* Plain English Explanation */}
      <div style={{
        background:   'rgba(0,0,0,0.3)',
        borderRadius: '5px',
        padding:      '8px 10px',
        fontSize:     '10px',
        lineHeight:   '1.5',
        color:        isAnomaly ? '#FCA5A5' : '#A7F3D0',
        borderLeft:   `2px solid ${isAnomaly ? '#EF4444' : '#00D4AA'}`
      }}>
        <span style={{
          fontWeight:    600,
          marginRight:   '6px',
          color:         isAnomaly ? '#EF4444' : '#00D4AA'
        }}>
          {isAnomaly ? '⚠ ANOMALY:' : '✓ NORMAL:'}
        </span>
        {getExplanation()}
      </div>
    </div>
  )
}