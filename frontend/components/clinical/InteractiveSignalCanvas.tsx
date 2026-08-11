'use client'

import { useRef, useState, useEffect, useMemo, useCallback } from 'react'

interface InteractiveSignalCanvasProps {
  signal:       number[]
  spikeTrain:   number[]
  attribution:  number[]
  isAnomaly:    boolean
  modality:     string
  sampleRate:   number
  height?:      number
}

interface PeakInfo {
  index:      number
  peakNumber: number
  time:       number
  amplitude:  number
  spikeCount: number
  importance: number
  reason:     string
}

function detectPeaks(signal: number[], spikeTrain: number[], attribution: number[], sampleRate: number): PeakInfo[] {
  if (!signal.length) return []
  const n = signal.length
  const minDistance = Math.max(4, Math.round(n / 60))
  const mean = signal.reduce((a, b) => a + b, 0) / n
  const std = Math.sqrt(signal.reduce((a, b) => a + (b - mean) ** 2, 0) / n) || 1
  const threshold = mean + std * 0.8

  const rawPeaks: number[] = []
  for (let i = 2; i < n - 2; i++) {
    if (
      signal[i] > threshold &&
      signal[i] >= signal[i - 1] &&
      signal[i] >= signal[i + 1] &&
      signal[i] >= signal[i - 2] &&
      signal[i] >= signal[i + 2]
    ) {
      if (!rawPeaks.length || i - rawPeaks[rawPeaks.length - 1] >= minDistance) {
        rawPeaks.push(i)
      } else if (signal[i] > signal[rawPeaks[rawPeaks.length - 1]]) {
        rawPeaks[rawPeaks.length - 1] = i
      }
    }
  }

  const capped = rawPeaks.slice(0, 60)

  return capped.map((idx, pi) => {
    const win = Math.max(3, Math.round(n * 0.02))
    const winStart = Math.max(0, idx - win)
    const winEnd = Math.min(n, idx + win)
    const spikeCount = spikeTrain.slice(winStart, winEnd).filter(s => s > 0.5).length
    const importance = Math.round((attribution[idx] ?? 0) * 100)

    let reason = 'Moderate neural attribution in this region.'
    if (importance > 70) reason = 'High neural attribution around this region — strong contributor to the decision.'
    else if (importance > 40) reason = 'Elevated neural attribution — meaningful contributor to the decision.'
    else reason = 'Low neural attribution — minor contributor to the decision.'

    return {
      index: idx,
      peakNumber: pi + 1,
      time: idx / sampleRate,
      amplitude: signal[idx],
      spikeCount,
      importance,
      reason,
    }
  })
}

export default function InteractiveSignalCanvas({
  signal, spikeTrain, attribution, isAnomaly, sampleRate, height = 220
}: InteractiveSignalCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [zoom, setZoom] = useState<[number, number]>([0, 1])
  const [hovered, setHovered] = useState<PeakInfo | null>(null)
  const [tooltipPos, setTooltipPos] = useState<{ x: number, y: number }>({ x: 0, y: 0 })
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [containerWidth, setContainerWidth] = useState(900)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const update = () => setContainerWidth(el.clientWidth)
    update()
    const observer = new ResizeObserver(update)
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const peaks = useMemo(
    () => detectPeaks(signal, spikeTrain, attribution, sampleRate),
    [signal, spikeTrain, attribution, sampleRate]
  )

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !signal.length) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const w = canvas.width
    const h = canvas.height
    const n = signal.length
    const [zStart, zEnd] = zoom
    const startIdx = Math.floor(zStart * n)
    const endIdx = Math.max(startIdx + 4, Math.floor(zEnd * n))
    const visibleN = endIdx - startIdx

    ctx.fillStyle = '#FFFFFF'
    ctx.fillRect(0, 0, w, h)

    const visSignal = signal.slice(startIdx, endIdx)
    const min = Math.min(...visSignal)
    const max = Math.max(...visSignal)
    const range = max - min || 1

    // Attribution heatmap strip (top 20% of canvas)
    const heatH = h * 0.18
    for (let i = 0; i < visibleN; i++) {
      const val = attribution[startIdx + i] ?? 0
      const x = (i / visibleN) * w
      const barW = w / visibleN + 1
      const r = isAnomaly ? Math.round(239 * val + 249 * (1 - val)) : 191
      const g = isAnomaly ? Math.round(68 * val + 250 * (1 - val)) : 219
      const b = isAnomaly ? Math.round(68 * val + 249 * (1 - val)) : 254
      ctx.fillStyle = `rgba(${r},${g},${b},${Math.max(0.15, val)})`
      ctx.fillRect(x, 0, barW, heatH)
    }

    // Signal line
    const plotTop = heatH + 10
    const plotH = h - heatH - 30
    ctx.strokeStyle = isAnomaly ? '#EF4444' : '#6366F1'
    ctx.lineWidth = 1.8
    ctx.beginPath()
    visSignal.forEach((val, i) => {
      const x = (i / visibleN) * w
      const y = plotTop + plotH - ((val - min) / range) * plotH
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    })
    ctx.stroke()

    // Peak markers (only those in visible range)
    peaks.forEach(p => {
      if (p.index < startIdx || p.index > endIdx) return
      const relI = p.index - startIdx
      const x = (relI / visibleN) * w
      const y = plotTop + plotH - ((p.amplitude - min) / range) * plotH
      const isSel = selected.has(p.index)
      const isHov = hovered?.index === p.index

      ctx.beginPath()
      ctx.arc(x, y, isSel || isHov ? 6 : 4, 0, Math.PI * 2)
      ctx.fillStyle = isSel ? '#10B981' : (isHov ? '#F59E0B' : (isAnomaly ? '#EF4444' : '#6366F1'))
      ctx.globalAlpha = 0.9
      ctx.fill()
      ctx.globalAlpha = 1
      ctx.strokeStyle = 'white'
      ctx.lineWidth = 1.5
      ctx.stroke()
    })

    // Spike train ticks at bottom
    const spikeY = h - 14
    ctx.strokeStyle = '#D1D5DB'
    for (let i = 0; i < visibleN; i++) {
      if (spikeTrain[startIdx + i] > 0.5) {
        const x = (i / visibleN) * w
        ctx.beginPath()
        ctx.moveTo(x, spikeY)
        ctx.lineTo(x, spikeY + 10)
        ctx.stroke()
      }
    }

    ctx.strokeStyle = '#E5E7EB'
    ctx.lineWidth = 1
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1)
  }, [signal, spikeTrain, attribution, isAnomaly, zoom, peaks, hovered, selected])

  useEffect(() => { draw() }, [draw])

  const findNearestPeak = (clientX: number): PeakInfo | null => {
    const canvas = canvasRef.current
    if (!canvas) return null
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const mx = (clientX - rect.left) * scaleX

    const n = signal.length
    const [zStart, zEnd] = zoom
    const startIdx = Math.floor(zStart * n)
    const endIdx = Math.max(startIdx + 4, Math.floor(zEnd * n))
    const visibleN = endIdx - startIdx

    let nearest: PeakInfo | null = null
    let minDist = Infinity
    peaks.forEach(p => {
      if (p.index < startIdx || p.index > endIdx) return
      const relI = p.index - startIdx
      const x = (relI / visibleN) * canvas.width
      const dist = Math.abs(x - mx)
      if (dist < minDist && dist < 14) {
        minDist = dist
        nearest = p
      }
    })
    return nearest
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const p = findNearestPeak(e.clientX)
    setHovered(p)
    if (p && containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect()
      setTooltipPos({ x: e.clientX - rect.left, y: e.clientY - rect.top })
    }
  }

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const p = findNearestPeak(e.clientX)
    if (!p) return
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(p.index)) { next.delete(p.index) } else { next.add(p.index) }
      return next
    })
  }

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    const [zStart, zEnd] = zoom
    const span = zEnd - zStart
    const factor = e.deltaY < 0 ? 0.85 : 1.18
    const newSpan = Math.min(1, Math.max(0.05, span * factor))
    const center = zStart + span / 2
    let newStart = center - newSpan / 2
    let newEnd = center + newSpan / 2
    if (newStart < 0) { newEnd -= newStart; newStart = 0 }
    if (newEnd > 1) { newStart -= (newEnd - 1); newEnd = 1 }
    setZoom([Math.max(0, newStart), Math.min(1, newEnd)])
  }

  return (
    <div ref={containerRef} style={{ position: 'relative' }}>
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px'
      }}>
        <span style={{ fontSize: '11px', color: '#9CA3AF' }}>
          {peaks.length} peaks detected · scroll to zoom · click to select
        </span>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {selected.size > 0 && (
            <span style={{ fontSize: '11px', fontWeight: 700, color: '#10B981' }}>
              {selected.size} selected
            </span>
          )}
          {(zoom[0] > 0 || zoom[1] < 1) && (
            <button
              onClick={() => setZoom([0, 1])}
              style={{
                fontSize: '11px', fontWeight: 600, color: '#6366F1', background: '#EEF2FF',
                border: 'none', borderRadius: '6px', padding: '4px 10px', cursor: 'pointer'
              }}
            >
              ↺ Reset Zoom
            </button>
          )}
        </div>
      </div>

      <canvas
        ref={canvasRef}
        width={900}
        height={height}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHovered(null)}
        onClick={handleClick}
        onWheel={handleWheel}
        style={{
          width: '100%', height: `${height}px`, display: 'block',
          borderRadius: '8px', cursor: hovered ? 'pointer' : 'crosshair'
        }}
      />

      {hovered && (
        <div style={{
          position: 'absolute',
          left: Math.min(tooltipPos.x + 14, containerWidth - 200),
          top: Math.max(tooltipPos.y - 100, 4),
          background: '#111827',
          color: 'white',
          borderRadius: '10px',
          padding: '12px 14px',
          fontSize: '12px',
          width: '190px',
          pointerEvents: 'none',
          boxShadow: '0 8px 24px rgba(0,0,0,0.25)',
          zIndex: 10,
        }}>
          <div style={{ fontWeight: 800, marginBottom: '6px', color: '#93C5FD' }}>
            Peak #{hovered.peakNumber}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
            <span style={{ color: '#9CA3AF' }}>Time</span><span>{hovered.time.toFixed(3)} s</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
            <span style={{ color: '#9CA3AF' }}>Amplitude</span><span>{hovered.amplitude.toFixed(3)} mV</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
            <span style={{ color: '#9CA3AF' }}>Spike Count</span><span>{hovered.spikeCount}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span style={{ color: '#9CA3AF' }}>Importance</span>
            <span style={{ color: '#34D399', fontWeight: 700 }}>{hovered.importance}%</span>
          </div>
          <div style={{ fontSize: '11px', color: '#D1D5DB', lineHeight: 1.4, borderTop: '1px solid #374151', paddingTop: '6px' }}>
            {hovered.reason}
          </div>
        </div>
      )}
    </div>
  )
}