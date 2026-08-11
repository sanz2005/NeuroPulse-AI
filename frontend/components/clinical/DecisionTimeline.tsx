'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'

interface DecisionTimelineProps {
  latencyMs: number
  isAnomaly: boolean
  modality:  string
}

interface Step {
  label: string
  fraction: number // fraction of total latency at which this step completes
}

const STEPS: Step[] = [
  { label: 'Signal Loaded',            fraction: 0     },
  { label: 'Spike Encoding Started',   fraction: 0.11  },
  { label: 'Feature Extraction',       fraction: 0.37  },
  { label: 'Pattern Identified',       fraction: 0.70  },
  { label: 'Spike Burst Detected',     fraction: 0.86  },
  { label: 'Decision Locked',          fraction: 0.92  },
  { label: 'Prediction Completed',     fraction: 1.0   },
]

export default function DecisionTimeline({ latencyMs, isAnomaly, modality }: DecisionTimelineProps) {
  const [activeStep, setActiveStep] = useState(-1)

  const times = STEPS.map(s => s.fraction * latencyMs)

  useEffect(() => {
    const resetRaf = requestAnimationFrame(() => setActiveStep(-1))
    const timers = times.map((t, i) =>
      setTimeout(() => setActiveStep(i), Math.min(t, 400) * (i + 1) / STEPS.length + i * 90)
    )
    return () => {
      cancelAnimationFrame(resetRaf)
      timers.forEach(clearTimeout)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latencyMs])

  return (
    <div style={{
      background: 'white', border: '1px solid #E5E7EB', borderRadius: '16px', padding: '24px'
    }}>
      <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#111827', margin: '0 0 4px 0' }}>
        ⏱ Decision Timeline
      </h3>
      <p style={{ fontSize: '13px', color: '#6B7280', margin: '0 0 24px 0' }}>
        Millisecond-level breakdown of the SNN forward pass for this{' '}
        {modality.toUpperCase()} window (total: {latencyMs.toFixed(2)}ms).
      </p>

      <div style={{ position: 'relative', paddingLeft: '8px' }}>
        <div style={{
          position: 'absolute', left: '19px', top: '10px', bottom: '10px',
          width: '2px', background: '#E5E7EB'
        }} />
        <motion.div
          initial={{ height: 0 }}
          animate={{ height: activeStep >= 0 ? `${(activeStep / (STEPS.length - 1)) * 100}%` : 0 }}
          transition={{ duration: 0.4, ease: 'easeOut' }}
          style={{
            position: 'absolute', left: '19px', top: '10px',
            width: '2px', background: isAnomaly ? '#EF4444' : '#6366F1'
          }}
        />

        {STEPS.map((step, i) => {
          const done = i <= activeStep
          return (
            <motion.div
              key={step.label}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
              style={{
                display: 'flex', alignItems: 'center', gap: '16px',
                padding: '10px 0', position: 'relative'
              }}
            >
              <motion.div
                animate={{
                  scale: done ? 1 : 0.7,
                  backgroundColor: done ? (isAnomaly ? '#EF4444' : '#6366F1') : '#E5E7EB',
                }}
                transition={{ duration: 0.25 }}
                style={{
                  width: '20px', height: '20px', borderRadius: '50%',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  flexShrink: 0, zIndex: 1,
                  boxShadow: done ? `0 0 0 4px ${isAnomaly ? '#FEE2E2' : '#EEF2FF'}` : 'none',
                }}
              >
                {done && (
                  <motion.span
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    style={{ color: 'white', fontSize: '10px', fontWeight: 900 }}
                  >
                    ✓
                  </motion.span>
                )}
              </motion.div>
              <div style={{ flex: 1 }}>
                <div style={{
                  fontSize: '13px', fontWeight: done ? 700 : 500,
                  color: done ? '#111827' : '#9CA3AF',
                }}>
                  {step.label}
                </div>
              </div>
              <div style={{
                fontSize: '12px', fontWeight: 700,
                color: done ? (isAnomaly ? '#EF4444' : '#6366F1') : '#D1D5DB',
                fontVariantNumeric: 'tabular-nums'
              }}>
                {times[i].toFixed(1)} ms
              </div>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}