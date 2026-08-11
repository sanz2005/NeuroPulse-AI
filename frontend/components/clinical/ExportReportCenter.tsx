'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { generateSummary } from './DoctorSummary'

interface ReportResult {
  modality:        string
  window_index?:   number
  sample_rate:     number
  signal:          number[]
  spike_count:     number
  true_label_text?: string
  snn: {
    label:        string
    confidence:   number
    conf_normal:  number
    conf_anomaly: number
    is_anomaly:   boolean
    total_spikes: number
    sparsity:     number
    latency_ms:   number
    energy_mj:    number
    model_type?:  string
  }
  clinical: {
    finding:   string
    severity:  string
    actions:   string[]
    icd_code:  string
  }
  baselines?: {
    cnn?:         { energy_mj: number, confidence: number, latency_ms: number }
    lstm?:        { energy_mj: number, confidence: number, latency_ms: number }
    transformer?: { energy_mj: number, confidence: number, latency_ms: number }
  }
}

interface ExportReportCenterProps {
  result: ReportResult
}

function downloadBlob(content: string, filename: string, mime: string) {
  const blob = new Blob([content], { type: mime })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href     = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

function buildFhirReport(result: ReportResult) {
  const snn      = result.snn
  const clinical = result.clinical
  const now      = new Date().toISOString()

  return {
    resourceType: 'DiagnosticReport',
    id: `neuropulse-${result.window_index ?? 0}-${Date.now()}`,
    status: 'final',
    category: [{
      coding: [{
        system: 'http://terminology.hl7.org/CodeSystem/v2-0074',
        code: result.modality === 'ecg' ? 'CG' : result.modality === 'eeg' ? 'NR' : 'MB',
        display: `${(result.modality || '').toUpperCase()} Signal Analysis`,
      }],
    }],
    code: {
      coding: [{ system: 'http://hl7.org/fhir/sid/icd-10', code: clinical?.icd_code, display: snn?.label }],
    },
    subject: { reference: 'Patient/simulated-subject' },
    effectiveDateTime: now,
    issued: now,
    performer: [{ display: 'NeuroPulse AI — Neuromorphic Biosignal Analysis Platform' }],
    result: [{ display: `${result.modality?.toUpperCase()} Window ${result.window_index}` }],
    conclusion: clinical?.finding,
    extension: [
      { url: 'prediction',      valueString: snn?.label },
      { url: 'confidence',      valueDecimal: snn?.confidence },
      { url: 'spikeCount',      valueInteger: result.spike_count },
      { url: 'inferenceTimeMs', valueDecimal: snn?.latency_ms },
      { url: 'energyMj',        valueDecimal: snn?.energy_mj },
      { url: 'device',          valueString: 'NeuroPulse SNN Edge Model' },
      { url: 'model',           valueString: snn?.model_type || 'Spiking Neural Network' },
    ],
  }
}

async function buildAndDownloadPdf(result: ReportResult) {
  const { jsPDF } = await import('jspdf')
  const autoTableModule = await import('jspdf-autotable')
  const autoTable = autoTableModule.default

  const snn      = result.snn
  const clinical = result.clinical
  const isAnomaly = snn.is_anomaly
  const accent: [number, number, number] = isAnomaly ? [239, 68, 68] : [16, 185, 129]
  const indigo: [number, number, number] = [99, 102, 241]

  const doc = new jsPDF({ unit: 'pt', format: 'a4' })
  const pageW = doc.internal.pageSize.getWidth()
  const marginX = 40
  let y = 0

  doc.setFillColor(17, 24, 39)
  doc.rect(0, 0, pageW, 80, 'F')
  doc.setTextColor(255, 255, 255)
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(18)
  doc.text('NeuroPulse AI', marginX, 34)
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(10)
  doc.setTextColor(200, 205, 215)
  doc.text('Neuromorphic Biosignal Analysis Platform — Clinical Diagnostic Report', marginX, 50)
  doc.setFontSize(9)
  doc.text(`Generated ${new Date().toLocaleString()}`, pageW - marginX, 34, { align: 'right' })
  doc.text(`Report ID: NP-${Date.now().toString().slice(-8)}`, pageW - marginX, 48, { align: 'right' })
  y = 105

  doc.setFillColor(...accent)
  doc.setDrawColor(...accent)
  doc.roundedRect(marginX, y, pageW - marginX * 2, 56, 6, 6, 'F')
  doc.setTextColor(255, 255, 255)
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(16)
  doc.text(snn.label, marginX + 16, y + 24)
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(10)
  doc.text(
    `${result.modality.toUpperCase()} Signal · Confidence ${(snn.confidence * 100).toFixed(2)}% · ICD ${clinical.icd_code}`,
    marginX + 16, y + 42
  )
  y += 76

  doc.setTextColor(17, 24, 39)
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(12)
  doc.text('Signal Information', marginX, y)
  y += 8

  autoTable(doc, {
    startY: y,
    theme: 'grid',
    margin: { left: marginX, right: marginX },
    styles: { fontSize: 9, cellPadding: 6 },
    headStyles: { fillColor: indigo, textColor: 255 },
    head: [['Modality', 'Window', 'Sample Rate', 'Duration', 'Ground Truth']],
    body: [[
      result.modality.toUpperCase(),
      String(result.window_index ?? '—'),
      `${result.sample_rate} Hz`,
      `${(result.signal.length / result.sample_rate).toFixed(2)} s`,
      result.true_label_text || '—',
    ]],
  })
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  y = (doc as any).lastAutoTable.finalY + 24

  doc.setFont('helvetica', 'bold')
  doc.setFontSize(12)
  doc.text('SNN Model Metrics', marginX, y)
  y += 8

  autoTable(doc, {
    startY: y,
    theme: 'grid',
    margin: { left: marginX, right: marginX },
    styles: { fontSize: 9, cellPadding: 6 },
    headStyles: { fillColor: indigo, textColor: 255 },
    head: [['Confidence', 'Total Spikes', 'Sparsity', 'Inference Time', 'Energy']],
    body: [[
      `${(snn.confidence * 100).toFixed(2)}%`,
      Math.round(snn.total_spikes).toLocaleString(),
      `${(snn.sparsity * 100).toFixed(1)}%`,
      `${snn.latency_ms.toFixed(2)} ms`,
      `${snn.energy_mj.toFixed(5)} mJ`,
    ]],
  })
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  y = (doc as any).lastAutoTable.finalY + 24

  doc.setFont('helvetica', 'bold')
  doc.setFontSize(12)
  doc.text('Clinical Finding', marginX, y)
  y += 10
  doc.setFillColor(249, 250, 251)
  const findingLines = doc.splitTextToSize(clinical.finding, pageW - marginX * 2 - 24)
  const findingH = findingLines.length * 13 + 16
  doc.roundedRect(marginX, y, pageW - marginX * 2, findingH, 4, 4, 'F')
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(10)
  doc.setTextColor(31, 41, 55)
  doc.text(findingLines, marginX + 12, y + 18)
  y += findingH + 20

  doc.setFont('helvetica', 'bold')
  doc.setFontSize(12)
  doc.setTextColor(17, 24, 39)
  doc.text('Recommended Actions', marginX, y)
  y += 8

  autoTable(doc, {
    startY: y,
    theme: 'striped',
    margin: { left: marginX, right: marginX },
    styles: { fontSize: 9, cellPadding: 6 },
    headStyles: { fillColor: [55, 65, 81], textColor: 255 },
    head: [['#', 'Action']],
    body: clinical.actions.map((a, i) => [String(i + 1), a]),
    columnStyles: { 0: { cellWidth: 30 } },
  })
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  y = (doc as any).lastAutoTable.finalY + 24

  if (y > 650) { doc.addPage(); y = 40 }
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(12)
  doc.text('Doctor Summary (Plain Language)', marginX, y)
  y += 10
  const summary = generateSummary(result.modality, isAnomaly)
  const summaryLines = doc.splitTextToSize(summary, pageW - marginX * 2 - 24)
  const summaryH = summaryLines.length * 13 + 16
  doc.setFillColor(isAnomaly ? 254 : 240, isAnomaly ? 242 : 253, isAnomaly ? 242 : 244)
  doc.roundedRect(marginX, y, pageW - marginX * 2, summaryH, 4, 4, 'F')
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(10)
  doc.setTextColor(isAnomaly ? 127 : 20, isAnomaly ? 29 : 83, isAnomaly ? 29 : 45)
  doc.text(summaryLines, marginX + 12, y + 18)
  y += summaryH + 24

  if (y > 620) { doc.addPage(); y = 40 }
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(12)
  doc.setTextColor(17, 24, 39)
  doc.text('Model Comparison — Energy & Latency', marginX, y)
  y += 8

  const b = result.baselines
  autoTable(doc, {
    startY: y,
    theme: 'grid',
    margin: { left: marginX, right: marginX },
    styles: { fontSize: 9, cellPadding: 6 },
    headStyles: { fillColor: indigo, textColor: 255 },
    head: [['Model', 'Confidence', 'Latency', 'Energy']],
    body: [
      ['SNN (Ours)', `${(snn.confidence * 100).toFixed(1)}%`, `${snn.latency_ms.toFixed(2)} ms`, `${snn.energy_mj.toFixed(5)} mJ`],
      ...(b?.cnn ? [['CNN', `${(b.cnn.confidence * 100).toFixed(1)}%`, `${b.cnn.latency_ms.toFixed(2)} ms`, `${b.cnn.energy_mj.toFixed(5)} mJ`]] : []),
      ...(b?.lstm ? [['LSTM', `${(b.lstm.confidence * 100).toFixed(1)}%`, `${b.lstm.latency_ms.toFixed(2)} ms`, `${b.lstm.energy_mj.toFixed(5)} mJ`]] : []),
      ...(b?.transformer ? [['Transformer', `${(b.transformer.confidence * 100).toFixed(1)}%`, `${b.transformer.latency_ms.toFixed(2)} ms`, `${b.transformer.energy_mj.toFixed(5)} mJ`]] : []),
    ],
  })
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  y = (doc as any).lastAutoTable.finalY + 30

  const pageCount = doc.internal.pages.length - 1
  for (let p = 1; p <= pageCount; p++) {
    doc.setPage(p)
    const pageH = doc.internal.pageSize.getHeight()
    doc.setDrawColor(229, 231, 235)
    doc.line(marginX, pageH - 36, pageW - marginX, pageH - 36)
    doc.setFont('helvetica', 'normal')
    doc.setFontSize(8)
    doc.setTextColor(156, 163, 175)
    doc.text(
      'Generated by NeuroPulse AI. Simulated report for explainability and demonstration purposes.',
      marginX, pageH - 22
    )
    doc.text(`Page ${p} of ${pageCount}`, pageW - marginX, pageH - 22, { align: 'right' })
  }

  doc.save(`neuropulse-report-${result.modality}-${Date.now()}.pdf`)
}

function ReportRow({ label, value }: { label: string, value: string }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', padding: '8px 0',
      borderBottom: '1px solid #F3F4F6', fontSize: '12px'
    }}>
      <span style={{ color: '#9CA3AF' }}>{label}</span>
      <span style={{ color: '#111827', fontWeight: 700 }}>{value}</span>
    </div>
  )
}

// ── On-dashboard report preview ─────────────────────────────────────────
function ReportPreview({ result }: { result: ReportResult }) {
  const snn = result.snn
  const clinical = result.clinical
  const isAnomaly = snn.is_anomaly
  const accent = isAnomaly ? '#EF4444' : '#10B981'
  const summary = generateSummary(result.modality, isAnomaly)
  const b = result.baselines

  return (
    <div style={{
      background: 'white', border: '1px solid #E5E7EB', borderRadius: '16px',
      overflow: 'hidden'
    }}>
      {/* Header band, mirrors the PDF header */}
      <div style={{ background: '#111827', padding: '20px 24px' }}>
        <div style={{ fontSize: '16px', fontWeight: 800, color: 'white' }}>NeuroPulse AI</div>
        <div style={{ fontSize: '11px', color: '#9CA3AF', marginTop: '2px' }}>
          Neuromorphic Biosignal Analysis Platform — Clinical Diagnostic Report
        </div>
      </div>

      <div style={{ padding: '24px' }}>
        {/* Prediction banner */}
        <div style={{
          background: accent, borderRadius: '10px', padding: '16px 18px', marginBottom: '20px'
        }}>
          <div style={{ fontSize: '18px', fontWeight: 800, color: 'white' }}>{snn.label}</div>
          <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.85)', marginTop: '4px' }}>
            {result.modality.toUpperCase()} Signal · Confidence {(snn.confidence * 100).toFixed(2)}% · ICD {clinical.icd_code}
          </div>
        </div>

        {/* Signal info */}
        <div style={{ fontSize: '12px', fontWeight: 700, color: '#6B7280', marginBottom: '6px' }}>
          SIGNAL INFORMATION
        </div>
        <ReportRow label="Modality" value={result.modality.toUpperCase()} />
        <ReportRow label="Window" value={String(result.window_index ?? '—')} />
        <ReportRow label="Sample Rate" value={`${result.sample_rate} Hz`} />
        <ReportRow label="Duration" value={`${(result.signal.length / result.sample_rate).toFixed(2)} s`} />
        <ReportRow label="Ground Truth" value={result.true_label_text || '—'} />

        <div style={{ fontSize: '12px', fontWeight: 700, color: '#6B7280', margin: '20px 0 6px 0' }}>
          SNN MODEL METRICS
        </div>
        <ReportRow label="Confidence" value={`${(snn.confidence * 100).toFixed(2)}%`} />
        <ReportRow label="Total Spikes" value={Math.round(snn.total_spikes).toLocaleString()} />
        <ReportRow label="Sparsity" value={`${(snn.sparsity * 100).toFixed(1)}%`} />
        <ReportRow label="Inference Time" value={`${snn.latency_ms.toFixed(2)} ms`} />
        <ReportRow label="Energy" value={`${snn.energy_mj.toFixed(5)} mJ`} />

        {/* Clinical finding */}
        <div style={{ fontSize: '12px', fontWeight: 700, color: '#6B7280', margin: '20px 0 8px 0' }}>
          CLINICAL FINDING
        </div>
        <div style={{
          background: '#F9FAFB', borderRadius: '8px', padding: '14px 16px',
          fontSize: '13px', color: '#374151', lineHeight: 1.6
        }}>
          {clinical.finding}
        </div>

        {/* Recommended actions */}
        <div style={{ fontSize: '12px', fontWeight: 700, color: '#6B7280', margin: '20px 0 8px 0' }}>
          RECOMMENDED ACTIONS
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {clinical.actions.map((a, i) => (
            <div key={i} style={{
              display: 'flex', gap: '10px', fontSize: '12px', color: '#374151',
              background: '#FAFAFA', borderRadius: '6px', padding: '8px 12px'
            }}>
              <span style={{ color: '#9CA3AF', fontWeight: 700 }}>{i + 1}.</span>
              <span>{a}</span>
            </div>
          ))}
        </div>

        {/* Doctor summary */}
        <div style={{ fontSize: '12px', fontWeight: 700, color: '#6B7280', margin: '20px 0 8px 0' }}>
          DOCTOR SUMMARY (PLAIN LANGUAGE)
        </div>
        <div style={{
          background: isAnomaly ? '#FEF2F2' : '#F0FDF4',
          border: `1px solid ${isAnomaly ? '#FCA5A5' : '#86EFAC'}`,
          borderRadius: '8px', padding: '14px 16px', fontSize: '13px',
          color: isAnomaly ? '#7F1D1D' : '#14532D', lineHeight: 1.6
        }}>
          {summary}
        </div>

        {/* Model comparison */}
        <div style={{ fontSize: '12px', fontWeight: 700, color: '#6B7280', margin: '20px 0 8px 0' }}>
          MODEL COMPARISON — ENERGY &amp; LATENCY
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {[
            { name: 'SNN (Ours)', conf: snn.confidence, lat: snn.latency_ms, e: snn.energy_mj, hl: true },
            ...(b?.cnn ? [{ name: 'CNN', conf: b.cnn.confidence, lat: b.cnn.latency_ms, e: b.cnn.energy_mj, hl: false }] : []),
            ...(b?.lstm ? [{ name: 'LSTM', conf: b.lstm.confidence, lat: b.lstm.latency_ms, e: b.lstm.energy_mj, hl: false }] : []),
            ...(b?.transformer ? [{ name: 'Transformer', conf: b.transformer.confidence, lat: b.transformer.latency_ms, e: b.transformer.energy_mj, hl: false }] : []),
          ].map(row => (
            <div key={row.name} style={{
              display: 'flex', justifyContent: 'space-between', fontSize: '12px',
              padding: '8px 12px', borderRadius: '6px',
              background: row.hl ? '#EEF2FF' : '#FAFAFA',
              fontWeight: row.hl ? 700 : 500, color: row.hl ? '#4338CA' : '#374151'
            }}>
              <span>{row.name}</span>
              <span>{(row.conf * 100).toFixed(1)}% · {row.lat.toFixed(2)}ms · {row.e.toFixed(5)}mJ</span>
            </div>
          ))}
        </div>

        <div style={{
          marginTop: '20px', fontSize: '10px', color: '#D1D5DB', textAlign: 'center',
          borderTop: '1px solid #F3F4F6', paddingTop: '12px'
        }}>
          Generated by NeuroPulse AI · Simulated report for explainability and demonstration purposes.
        </div>
      </div>
    </div>
  )
}

export default function ExportReportCenter({ result }: ExportReportCenterProps) {
  const [lastAction, setLastAction] = useState<string | null>(null)
  const [generating, setGenerating] = useState(false)

  const handlePdf = async () => {
    setGenerating(true)
    try {
      await buildAndDownloadPdf(result)
      setLastAction('pdf')
    } finally {
      setGenerating(false)
    }
  }

  const handleJson = () => {
    downloadBlob(JSON.stringify(result, null, 2), `neuropulse-report-${Date.now()}.json`, 'application/json')
    setLastAction('json')
  }

  const handleFhir = () => {
    const fhir = buildFhirReport(result)
    downloadBlob(JSON.stringify(fhir, null, 2), `neuropulse-fhir-${Date.now()}.json`, 'application/fhir+json')
    setLastAction('fhir')
  }

  const buttons = [
    { key: 'pdf',  label: 'Download PDF',       icon: '📄', color: '#EF4444', onClick: handlePdf,
      desc: 'Downloads the report above as a .pdf file' },
    { key: 'json', label: 'Download JSON',      icon: '🧾', color: '#6366F1', onClick: handleJson,
      desc: 'Complete raw prediction object' },
    { key: 'fhir', label: 'Download FHIR Report', icon: '🏥', color: '#10B981', onClick: handleFhir,
      desc: 'Simulated FHIR DiagnosticReport resource' },
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{
        background: 'white', border: '1px solid #E5E7EB', borderRadius: '16px', padding: '24px'
      }}>
        <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#111827', margin: '0 0 4px 0' }}>
          📤 Export Report
        </h3>
        <p style={{ fontSize: '13px', color: '#6B7280', margin: '0 0 20px 0' }}>
          Preview below is exactly what gets exported. Everything is generated locally —
          no backend call is made.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
          {buttons.map(b => (
            <motion.button
              key={b.key}
              onClick={b.onClick}
              disabled={b.key === 'pdf' && generating}
              whileHover={{ y: -2, boxShadow: '0 6px 16px rgba(0,0,0,0.08)' }}
              whileTap={{ scale: 0.98 }}
              style={{
                background:   'white',
                border:       `1px solid ${lastAction === b.key ? b.color : '#E5E7EB'}`,
                borderRadius: '12px',
                padding:      '18px 14px',
                cursor:       b.key === 'pdf' && generating ? 'default' : 'pointer',
                textAlign:    'left',
                display: 'flex', flexDirection: 'column', gap: '8px',
                opacity: b.key === 'pdf' && generating ? 0.6 : 1,
              }}
            >
              <div style={{
                width: '36px', height: '36px', borderRadius: '10px',
                background: `${b.color}15`, display: 'flex', alignItems: 'center',
                justifyContent: 'center', fontSize: '18px'
              }}>
                {b.icon}
              </div>
              <div style={{ fontSize: '13px', fontWeight: 700, color: '#111827' }}>
                {b.key === 'pdf' && generating ? 'Generating…' : b.label}
              </div>
              <div style={{ fontSize: '11px', color: '#9CA3AF', lineHeight: 1.4 }}>{b.desc}</div>
              {lastAction === b.key && (
                <div style={{ fontSize: '11px', fontWeight: 700, color: b.color }}>✓ Downloaded</div>
              )}
            </motion.button>
          ))}
        </div>
      </div>

      {/* Live report preview — always visible, mirrors the downloaded PDF */}
      <ReportPreview result={result} />
    </div>
  )
}