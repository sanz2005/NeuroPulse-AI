'use client'

interface EnergyGaugeProps {
  snnEnergy:  number
  cnnEnergy:  number
  lstmEnergy: number
  transEnergy: number
}

export default function EnergyGauge({
  snnEnergy  = 0.000205,
  cnnEnergy  = 3.109,
  lstmEnergy = 2.663,
  transEnergy = 0.328
}: EnergyGaugeProps) {
  const maxE   = Math.max(cnnEnergy, lstmEnergy, transEnergy)
  const saving = Math.round(cnnEnergy / snnEnergy)

  const models = [
    { name: 'SNN',   energy: snnEnergy,  color: '#00D4AA', width: `${(snnEnergy/maxE)*100}%` },
    { name: 'CNN',   energy: cnnEnergy,  color: '#EF4444', width: '100%' },
    { name: 'LSTM',  energy: lstmEnergy, color: '#F59E0B', width: `${(lstmEnergy/maxE)*100}%` },
    { name: 'Trans', energy: transEnergy,color: '#3B82F6', width: `${(transEnergy/maxE)*100}%` },
  ]

  return (
    <div style={{
      background:   '#161B22',
      border:       '0.5px solid #30363D',
      borderRadius: '8px',
      padding:      '10px 14px',
    }}>
      <div style={{
        fontSize:      '9px',
        color:         '#7D8590',
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        marginBottom:  '10px'
      }}>
        ⚡ Energy Efficiency — Clinical Deployment
      </div>

      {/* Big saving number */}
      <div style={{
        textAlign:    'center',
        marginBottom: '12px',
        padding:      '8px',
        background:   'rgba(0,212,170,0.08)',
        borderRadius: '6px',
        border:       '0.5px solid rgba(0,212,170,0.2)'
      }}>
        <div style={{
          fontSize:   '28px',
          fontWeight: 700,
          color:      '#00D4AA'
        }}>
          {saving.toLocaleString()}x
        </div>
        <div style={{ fontSize: '10px', color: '#7D8590' }}>
          more energy efficient than CNN
        </div>
        <div style={{ fontSize: '9px', color: '#7D8590', marginTop: '2px' }}>
          Critical for 24/7 hospital monitoring
        </div>
      </div>

      {/* Bars */}
      {models.map((m, i) => (
        <div key={i} style={{
          display:       'flex',
          alignItems:    'center',
          gap:           '8px',
          marginBottom:  '6px'
        }}>
          <span style={{
            fontSize:  '10px',
            color:     m.color,
            width:     '36px',
            flexShrink: 0,
            fontWeight: m.name === 'SNN' ? 700 : 400
          }}>
            {m.name}
          </span>
          <div style={{
            flex:         1,
            background:   '#21262D',
            borderRadius: '3px',
            height:       '8px',
            overflow:     'hidden'
          }}>
            <div style={{
              width:        m.width,
              height:       '100%',
              background:   m.color,
              borderRadius: '3px',
              transition:   'width 0.5s ease',
              minWidth:     m.name === 'SNN' ? '3px' : '0'
            }} />
          </div>
          <span style={{
            fontSize:  '10px',
            color:     m.name === 'SNN' ? '#00D4AA' : '#E6EDF3',
            width:     '60px',
            textAlign: 'right',
            fontWeight: m.name === 'SNN' ? 700 : 400
          }}>
            {m.energy.toFixed(m.name === 'SNN' ? 6 : 3)} mJ
          </span>
        </div>
      ))}

      <div style={{
        marginTop:  '8px',
        fontSize:   '9px',
        color:      '#7D8590',
        textAlign:  'center',
        fontStyle:  'italic'
      }}>
        SNN fires only when signal changes — like the brain
      </div>
    </div>
  )
}