'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'

interface ClinicalPriorityPanelProps {
  finding:   string
  actions:   string[]
  severity:  string
  isAnomaly: boolean
  modality:  string
}

interface PriorityTier {
  tier:        number
  emoji:       string
  title:       string
  urgency:     string
  color:       string
  bg:          string
  border:      string
  eta:         string
}

const TIERS: PriorityTier[] = [
  { tier: 1, emoji: '🔴', title: 'Immediate ECG Confirmation', urgency: 'Immediate', color: '#DC2626', bg: '#FEF2F2', border: '#FCA5A5', eta: 'Within 15 minutes' },
  { tier: 2, emoji: '🟠', title: 'Cardiology Consultation',    urgency: 'High',      color: '#EA580C', bg: '#FFF7ED', border: '#FDBA74', eta: 'Within 4 hours' },
  { tier: 3, emoji: '🟡', title: 'Continuous Monitoring',      urgency: 'Moderate',  color: '#CA8A04', bg: '#FEFCE8', border: '#FDE68A', eta: 'Ongoing, 24-48h' },
  { tier: 4, emoji: '🟢', title: 'Lifestyle Guidance',         urgency: 'Routine',   color: '#16A34A', bg: '#F0FDF4', border: '#86EFAC', eta: 'Next follow-up' },
]

const MODALITY_TITLES: Record<string, string[]> = {
  ecg: ['Immediate ECG Confirmation', 'Cardiology Consultation', 'Continuous Monitoring', 'Lifestyle Guidance'],
  eeg: ['Immediate EEG Confirmation', 'Neurology Consultation', 'Continuous Monitoring', 'Lifestyle Guidance'],
  emg: ['Immediate EMG Confirmation', 'Neuromuscular Consultation', 'Continuous Monitoring', 'Lifestyle Guidance'],
}

function activeTierFromSeverity(severity: string, isAnomaly: boolean): number {
  if (!isAnomaly) return 4
  const s = severity.toLowerCase()
  if (s.includes('critical') || s.includes('high')) return 1
  if (s.includes('moderate') || s.includes('medium')) return 2
  return 3
}

export default function ClinicalPriorityPanel({
  finding, actions, severity, isAnomaly, modality
}: ClinicalPriorityPanelProps) {
  const [hovered, setHovered] = useState<number | null>(null)
  const activeTier = activeTierFromSeverity(severity, isAnomaly)
  const titles = MODALITY_TITLES[modality] || MODALITY_TITLES.ecg

  // Distribute real clinical.actions across the 4 tiers so content stays grounded
  // in the actual model output rather than being invented.
  const distributed: string[][] = [[], [], [], []]
  actions.forEach((action, i) => {
    distributed[i % 4].push(action)
  })

  return (
    <div>
      <div style={{
        fontSize: '12px', fontWeight: 700, color: '#6B7280',
        textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '12px'
      }}>
        Clinical Priority
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {TIERS.map((tier, i) => {
          const isActive = tier.tier === activeTier
          const cardActions = distributed[i].length ? distributed[i] : [finding]
          return (
            <motion.div
              key={tier.tier}
              onMouseEnter={() => setHovered(tier.tier)}
              onMouseLeave={() => setHovered(null)}
              animate={{
                scale: hovered === tier.tier ? 1.01 : 1,
                boxShadow: isActive
                  ? `0 0 0 2px ${tier.color}30, 0 4px 12px rgba(0,0,0,0.06)`
                  : hovered === tier.tier
                    ? '0 4px 12px rgba(0,0,0,0.06)'
                    : '0 1px 3px rgba(0,0,0,0.03)'
              }}
              transition={{ duration: 0.15 }}
              style={{
                background:   isActive ? tier.bg : 'white',
                border:       `1px solid ${isActive ? tier.border : '#E5E7EB'}`,
                borderRadius: '12px',
                padding:      '16px 18px',
                cursor:       'default',
                opacity:      isActive ? 1 : 0.75,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '14px' }}>
                <div style={{
                  fontSize: '22px', flexShrink: 0, lineHeight: 1,
                  filter: isActive ? 'none' : 'grayscale(40%)'
                }}>
                  {tier.emoji}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '4px' }}>
                    <span style={{
                      fontSize: '10px', fontWeight: 800, color: tier.color,
                      background: `${tier.color}15`, padding: '2px 8px', borderRadius: '100px',
                      textTransform: 'uppercase', letterSpacing: '0.04em'
                    }}>
                      Priority {tier.tier}
                    </span>
                    <span style={{ fontSize: '11px', fontWeight: 700, color: tier.color }}>
                      {tier.urgency}
                    </span>
                    {isActive && (
                      <span style={{
                        fontSize: '10px', fontWeight: 700, color: 'white',
                        background: tier.color, padding: '2px 8px', borderRadius: '100px'
                      }}>
                        ACTIVE
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: '14px', fontWeight: 700, color: '#111827', marginBottom: '6px' }}>
                    {titles[i]}
                  </div>
                  <div style={{ fontSize: '12px', color: '#374151', lineHeight: 1.5, marginBottom: '8px' }}>
                    {cardActions.join(' · ')}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ fontSize: '11px', color: '#9CA3AF' }}>⏱ Suggested timing:</span>
                    <span style={{ fontSize: '11px', fontWeight: 600, color: '#6B7280' }}>{tier.eta}</span>
                  </div>
                </div>
              </div>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}