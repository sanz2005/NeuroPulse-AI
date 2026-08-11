'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { api } from '@/lib/api'

export default function ResearchPage() {
  const router         = useRouter()
  const [data, setData] = useState<any>(null)
  const [tab, setTab]   = useState('ecg')

  useEffect(() => {
    api.get('/api/benchmark/results/all')
       .then(r => setData(r.data))
       .catch(() => {})
  }, [])

  const tabs = [
    { id: 'ecg', label: '❤️ ECG' },
    { id: 'eeg', label: '🧠 EEG' },
    { id: 'emg', label: '💪 EMG' },
  ]

  const metrics = [
    { key: 'accuracy',  label: 'Accuracy',  fmt: (v: number) => `${(v*100).toFixed(2)}%` },
    { key: 'precision', label: 'Precision', fmt: (v: number) => `${(v*100).toFixed(2)}%` },
    { key: 'recall',    label: 'Recall',    fmt: (v: number) => `${(v*100).toFixed(2)}%` },
    { key: 'f1',        label: 'F1 Score',  fmt: (v: number) => `${(v*100).toFixed(2)}%` },
    { key: 'auc',       label: 'AUC',       fmt: (v: number) => `${(v*100).toFixed(2)}%` },
    { key: 'energy_mj', label: 'Energy',    fmt: (v: number) => `${v.toFixed(6)} mJ` },
    { key: 'latency_ms',label: 'Latency',   fmt: (v: number) => `${v.toFixed(2)} ms` },
    { key: 'sparsity',  label: 'Sparsity',  fmt: (v: number) => `${(v*100).toFixed(1)}%` },
  ]

  return (
    <div style={{
      background:  '#F9FAFB',
      minHeight:   '100vh',
      fontFamily:  '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    }}>
      <nav style={{
        background:     'white',
        borderBottom:   '1px solid #E5E7EB',
        padding:        '0 32px',
        display:        'flex',
        alignItems:     'center',
        gap:            '16px',
        height:         '64px',
        boxShadow:      '0 1px 3px rgba(0,0,0,0.05)'
      }}>
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
        <div style={{ fontSize: '16px', fontWeight: 700, color: '#111827' }}>
          🔬 Research Benchmarks
        </div>
        <div style={{ fontSize: '13px', color: '#6B7280' }}>
          SNN vs CNN vs LSTM vs Transformer — Real Results
        </div>
      </nav>

      <div style={{ maxWidth: '1000px', margin: '0 auto', padding: '32px 24px' }}>

        {/* Tabs */}
        <div style={{ display: 'flex', gap: '8px', marginBottom: '24px' }}>
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                padding:      '8px 20px',
                background:   tab === t.id ? '#6366F1' : 'white',
                border:       `1px solid ${tab === t.id ? '#6366F1' : '#E5E7EB'}`,
                borderRadius: '8px',
                fontSize:     '13px',
                fontWeight:   600,
                color:        tab === t.id ? 'white' : '#374151',
                cursor:       'pointer'
              }}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Results table */}
        {data && data[tab] && (
          <div style={{
            background:   'white',
            border:       '1px solid #E5E7EB',
            borderRadius: '16px',
            overflow:     'hidden',
            boxShadow:    '0 1px 3px rgba(0,0,0,0.05)'
          }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: '#F9FAFB' }}>
                  <th style={{
                    padding:   '12px 16px',
                    textAlign: 'left',
                    fontSize:  '12px',
                    fontWeight: 700,
                    color:     '#6B7280',
                    borderBottom: '1px solid #E5E7EB'
                  }}>
                    Model
                  </th>
                  {metrics.map(m => (
                    <th key={m.key} style={{
                      padding:   '12px 16px',
                      textAlign: 'right',
                      fontSize:  '12px',
                      fontWeight: 700,
                      color:     '#6B7280',
                      borderBottom: '1px solid #E5E7EB'
                    }}>
                      {m.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(data[tab]).map(([model, vals]: [string, any], i) => (
                  <tr key={model} style={{
                    background: model.includes('SNN') ? '#EEF2FF' : 'white',
                    borderBottom: '1px solid #F3F4F6'
                  }}>
                    <td style={{
                      padding:    '12px 16px',
                      fontSize:   '13px',
                      fontWeight: model.includes('SNN') ? 700 : 500,
                      color:      model.includes('SNN') ? '#4338CA' : '#111827'
                    }}>
                      {model.includes('SNN') ? '⚡ ' : ''}{model}
                    </td>
                    {metrics.map(m => (
                      <td key={m.key} style={{
                        padding:   '12px 16px',
                        textAlign: 'right',
                        fontSize:  '13px',
                        fontWeight: model.includes('SNN') ? 700 : 400,
                        color:     model.includes('SNN') ? '#4338CA' : '#374151'
                      }}>
                        {vals[m.key] !== undefined ? m.fmt(vals[m.key]) : '--'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {!data && (
          <div style={{
            textAlign: 'center', padding: '48px',
            color: '#9CA3AF', fontSize: '14px'
          }}>
            Loading benchmark results...
          </div>
        )}

        <div style={{
          marginTop:  '16px',
          fontSize:   '12px',
          color:      '#9CA3AF',
          textAlign:  'center'
        }}>
          All results from real trained models evaluated on held-out test sets.
          Datasets: MIT-BIH (ECG) · CHB-MIT (EEG) · NinaPro (EMG)
        </div>
      </div>
    </div>
  )
}