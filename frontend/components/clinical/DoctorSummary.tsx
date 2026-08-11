'use client'

interface DoctorSummaryProps {
  modality:  string
  isAnomaly: boolean
  finding:   string
  severity:  string
}

export function generateSummary(modality: string, isAnomaly: boolean): string {
  if (!isAnomaly) {
    const normal: Record<string, string> = {
      ecg: 'The ECG shows a regular heart rhythm with no signs of abnormal electrical activity. The heartbeat pattern falls within a healthy range, and no further cardiac evaluation is required at this time.',
      eeg: 'The EEG shows normal brain electrical activity with no signs of seizure-like patterns. Brain wave patterns fall within a healthy range.',
      emg: 'The muscle activity shows normal firing patterns with no signs of neuromuscular abnormality.',
    }
    return normal[modality] || normal.ecg
  }

  const abnormal: Record<string, string> = {
    ecg: 'The ECG demonstrates abnormal ventricular electrical activity. Irregular rhythm is visible near the QRS complex. These findings suggest cardiac rhythm abnormalities and further clinical evaluation is recommended.',
    eeg: 'The EEG demonstrates seizure-like electrical activity in multiple channels. Neurological evaluation is recommended.',
    emg: 'The muscle activity shows abnormal firing patterns indicating possible neuromuscular dysfunction.',
  }
  return abnormal[modality] || abnormal.ecg
}

export default function DoctorSummary({ modality, isAnomaly, severity }: DoctorSummaryProps) {
  const summary = generateSummary(modality, isAnomaly)

  return (
    <div style={{
      background: 'white', border: '1px solid #E5E7EB', borderRadius: '16px', padding: '24px'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
        <span style={{ fontSize: '20px' }}>🩺</span>
        <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#111827', margin: 0 }}>
          Doctor Summary
        </h3>
      </div>
      <p style={{ fontSize: '13px', color: '#6B7280', margin: '0 0 16px 0' }}>
        Plain-language summary intended for clinical staff — no technical or AI terminology.
      </p>

      <div style={{
        background:   isAnomaly ? '#FEF2F2' : '#F0FDF4',
        border:       `1px solid ${isAnomaly ? '#FCA5A5' : '#86EFAC'}`,
        borderRadius: '10px',
        padding:      '18px 20px',
      }}>
        <div style={{
          fontSize: '14px', color: isAnomaly ? '#7F1D1D' : '#14532D',
          lineHeight: 1.7,
        }}>
          {summary}
        </div>
      </div>

      <div style={{
        marginTop: '14px', display: 'flex', alignItems: 'center', gap: '8px',
        fontSize: '12px', color: '#9CA3AF'
      }}>
        <span>Modality: <strong style={{ color: '#6B7280' }}>{modality.toUpperCase()}</strong></span>
        <span>·</span>
        <span>Status: <strong style={{ color: '#6B7280' }}>{severity}</strong></span>
      </div>
    </div>
  )
}