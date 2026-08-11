'use client'

import { useEffect, useRef } from 'react'

interface SpikeRasterPlotProps {
  spikeData: number[][]  // [neurons x timesteps]
  neuronLabels?: string[]
  color?: string
  title?: string
  height?: number
}

export default function SpikeRasterPlot({
  spikeData,
  neuronLabels,
  color = '#00D4AA',
  title = 'Spike Raster Plot',
  height = 120
}: SpikeRasterPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !spikeData || spikeData.length === 0) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const w = canvas.width
    const h = canvas.height
    const nNeurons   = spikeData.length
    const nTimesteps = spikeData[0]?.length || 0

    // Background
    ctx.fillStyle = '#161B22'
    ctx.fillRect(0, 0, w, h)

    // Grid lines
    ctx.strokeStyle = '#21262D'
    ctx.lineWidth   = 0.5
    for (let t = 0; t < nTimesteps; t += 10) {
      const x = (t / nTimesteps) * w
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, h)
      ctx.stroke()
    }

    // Neuron lanes
    const laneH = h / nNeurons

    spikeData.forEach((neuronSpikes, nIdx) => {
      const y = nIdx * laneH

      // Lane separator
      ctx.strokeStyle = '#21262D'
      ctx.lineWidth   = 0.5
      ctx.beginPath()
      ctx.moveTo(0, y + laneH)
      ctx.lineTo(w, y + laneH)
      ctx.stroke()

      // Neuron label
      const label = neuronLabels?.[nIdx] || `N-${String(nIdx + 1).padStart(2, '0')}`
      ctx.fillStyle = '#7D8590'
      ctx.font      = '8px monospace'
      ctx.fillText(label, 3, y + laneH * 0.65)

      // Spikes
      neuronSpikes.forEach((spike, t) => {
        if (spike > 0.5) {
          const x = (t / nTimesteps) * w
          // Color based on neuron index
          const hue = (nIdx / nNeurons) * 180 + 150
          ctx.strokeStyle = color
          ctx.lineWidth   = 1.5
          ctx.globalAlpha = 0.85
          ctx.beginPath()
          ctx.moveTo(x, y + 2)
          ctx.lineTo(x, y + laneH - 2)
          ctx.stroke()
          ctx.globalAlpha = 1
        }
      })
    })

    // Border
    ctx.strokeStyle = '#30363D'
    ctx.lineWidth   = 0.5
    ctx.strokeRect(0, 0, w, h)

  }, [spikeData, color])

  return (
    <div>
      <div style={{
        fontSize:  '10px',
        color:     '#7D8590',
        marginBottom: '4px',
        letterSpacing: '0.06em'
      }}>
        {title}
      </div>
      <canvas
        ref={canvasRef}
        width={500}
        height={height}
        style={{
          width:   '100%',
          height:  `${height}px`,
          display: 'block',
          borderRadius: '4px'
        }}
      />
    </div>
  )
}