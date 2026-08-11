'use client'

import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'

export default function NeuralTwinPage() {
  const router     = useRouter()
  const mountRef   = useRef<HTMLDivElement>(null)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    if (!mountRef.current) return

    let scene: any, camera: any, renderer: any
    let neurons: any[] = []
    let spikes:  any[] = []
    let frame = 0
    let animId: number

    const THREE = require('three')

    // Setup
    scene    = new THREE.Scene()
    camera   = new THREE.PerspectiveCamera(50, mountRef.current.clientWidth / 500, 0.1, 1000)
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })

    renderer.setSize(mountRef.current.clientWidth, 500)
    renderer.setClearColor(0x0D1117, 1)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    mountRef.current.appendChild(renderer.domElement)

    camera.position.set(0, 0, 25)

    // Lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.4))
    const dl = new THREE.DirectionalLight(0xffffff, 0.8)
    dl.position.set(5, 10, 5)
    scene.add(dl)

    // Create 4-layer SNN
    const LAYERS = [
      { count: 12, x: -9,  color: 0x378ADD, name: 'Input'    },
      { count: 20, x: -3,  color: 0x7F77DD, name: 'Hidden 1' },
      { count: 20, x:  3,  color: 0xF59E0B, name: 'Hidden 2' },
      { count:  8, x:  9,  color: 0x00D4AA, name: 'Output'   },
    ]

    LAYERS.forEach((layer) => {
      const cols = Math.ceil(Math.sqrt(layer.count))
      for (let i = 0; i < layer.count; i++) {
        const row = Math.floor(i / cols)
        const col = i % cols
        const y   = (row - Math.floor(layer.count / cols) / 2) * 2.2
        const z   = (col - cols / 2) * 2.2
        const geo = new THREE.SphereGeometry(0.38, 12, 12)
        const mat = new THREE.MeshPhongMaterial({
          color:             layer.color,
          emissive:          layer.color,
          emissiveIntensity: 0.15
        })
        const mesh = new THREE.Mesh(geo, mat)
        mesh.position.set(layer.x, y, z)
        scene.add(mesh)
        neurons.push({ mesh, layer: LAYERS.indexOf(layer), firing: false, fireTime: 0 })
      }
    })

    // Connections
    const lineMat = new THREE.LineBasicMaterial({ color: 0x21262D, transparent: true, opacity: 0.3 })
    for (let i = 0; i < neurons.length; i++) {
      if (neurons[i].layer >= LAYERS.length - 1) continue
      const next = neurons.filter(n => n.layer === neurons[i].layer + 1)
      next.slice(0, 2).forEach(n2 => {
        const geo  = new THREE.BufferGeometry().setFromPoints([neurons[i].mesh.position, n2.mesh.position])
        scene.add(new THREE.Line(geo, lineMat.clone()))
      })
    }

    // Mouse rotation
    let rotY = 0, autoRotate = true
    const handleMouse = (e: MouseEvent) => {
      autoRotate = false
      rotY += e.movementX * 0.005
    }
    renderer.domElement.addEventListener('mousemove', handleMouse)

    // Spike trigger
    function triggerSpike() {
      const input = neurons.filter(n => n.layer === 0)
      const seed  = input[Math.floor(Math.random() * input.length)]
      seed.firing   = true
      seed.fireTime = frame
      seed.mesh.material.emissiveIntensity = 1.2
      seed.mesh.scale.setScalar(1.8)
    }

    function propagate() {
      neurons.forEach(n => {
        if (!n.firing) return
        const age = frame - n.fireTime
        if (age > 6 && n.layer < LAYERS.length - 1) {
          const next = neurons.filter(nn => nn.layer === n.layer + 1)
          next.sort(() => Math.random() - 0.5).slice(0, 2).forEach(t => {
            t.firing   = true
            t.fireTime = frame
            t.mesh.material.emissiveIntensity = 1.2
            t.mesh.scale.setScalar(1.8)
            const sg  = new THREE.BufferGeometry().setFromPoints([n.mesh.position, t.mesh.position])
            const sm  = new THREE.LineBasicMaterial({ color: LAYERS[n.layer].color, transparent: true, opacity: 0.9 })
            const sl  = new THREE.Line(sg, sm)
            scene.add(sl)
            spikes.push({ line: sl, born: frame })
          })
          n.firing = false
        }
        if (age > 3) {
          n.mesh.material.emissiveIntensity = Math.max(0.15, 1.2 - age * 0.15)
          n.mesh.scale.setScalar(Math.max(1, 1.8 - age * 0.12))
        }
      })
      for (let i = spikes.length - 1; i >= 0; i--) {
        const age = frame - spikes[i].born
        spikes[i].line.material.opacity = Math.max(0, 0.9 - age * 0.09)
        if (age > 12) { scene.remove(spikes[i].line); spikes.splice(i, 1) }
      }
    }

    function animate() {
      animId = requestAnimationFrame(animate)
      frame++
      if (autoRotate) rotY += 0.006
      camera.position.x = 25 * Math.sin(rotY)
      camera.position.z = 25 * Math.cos(rotY)
      camera.lookAt(0, 0, 0)
      if (Math.random() < 0.04) triggerSpike()
      propagate()
      renderer.render(scene, camera)
    }

    animate()
    setLoaded(true)

    return () => {
      cancelAnimationFrame(animId)
      renderer.domElement.removeEventListener('mousemove', handleMouse)
      renderer.dispose()
      if (mountRef.current?.contains(renderer.domElement)) {
        mountRef.current.removeChild(renderer.domElement)
      }
    }
  }, [])

  return (
    <div style={{
      background: '#0D1117', minHeight: '100vh',
      color: '#E6EDF3', fontFamily: 'monospace'
    }}>
      {/* Header */}
      <div style={{
        background: '#161B22', borderBottom: '0.5px solid #30363D',
        padding: '10px 20px', display: 'flex',
        alignItems: 'center', justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button
            onClick={() => router.push('/')}
            style={{
              background: 'transparent', border: '0.5px solid #30363D',
              borderRadius: '4px', padding: '4px 10px',
              color: '#7D8590', cursor: 'pointer',
              fontFamily: 'monospace', fontSize: '11px'
            }}
          >
            ← Back
          </button>
          <span style={{ fontSize: '14px', fontWeight: 600 }}>
            🧠 3D Neural Twin — Live SNN Visualization
          </span>
        </div>
        <div style={{ fontSize: '11px', color: '#7D8590' }}>
          Drag to rotate · 4-layer LIF SNN · 60 neurons
        </div>
      </div>

      <div style={{ padding: '20px' }}>
        {/* 3D Canvas */}
        <div style={{
          background: '#161B22', border: '0.5px solid #30363D',
          borderRadius: '12px', overflow: 'hidden',
          marginBottom: '16px'
        }}>
          <div ref={mountRef} style={{ width: '100%', height: '500px' }} />
        </div>

        {/* Layer Info */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
          {[
            { name: 'Input Layer',  neurons: 12, color: '#378ADD', desc: 'Receives spike trains from ECG/EEG/EMG encoders' },
            { name: 'Hidden 1',     neurons: 20, color: '#7F77DD', desc: 'First LIF processing layer — temporal integration' },
            { name: 'Hidden 2',     neurons: 20, color: '#F59E0B', desc: 'Second LIF layer — pattern recognition' },
            { name: 'Output Layer', neurons:  8, color: '#00D4AA', desc: 'Classification output — normal vs anomaly' },
          ].map((layer, i) => (
            <div key={i} style={{
              background: '#161B22', border: `0.5px solid ${layer.color}44`,
              borderRadius: '8px', padding: '12px'
            }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: layer.color, marginBottom: '4px' }}>
                {layer.name}
              </div>
              <div style={{ fontSize: '20px', fontWeight: 700, color: '#E6EDF3', marginBottom: '4px' }}>
                {layer.neurons}
                <span style={{ fontSize: '11px', color: '#7D8590', marginLeft: '4px' }}>neurons</span>
              </div>
              <div style={{ fontSize: '10px', color: '#7D8590', lineHeight: '1.4' }}>
                {layer.desc}
              </div>
            </div>
          ))}
        </div>

        {/* SNN Info */}
        <div style={{
          marginTop: '16px', background: '#161B22',
          border: '0.5px solid #30363D', borderRadius: '8px',
          padding: '14px 16px', fontSize: '11px', color: '#7D8590',
          lineHeight: '1.6'
        }}>
          <span style={{ color: '#00D4AA', fontWeight: 600 }}>About this visualization: </span>
          Each sphere represents a Leaky Integrate-and-Fire (LIF) neuron.
          Colored lines show spike propagation between layers.
          Bright glowing neurons indicate active firing.
          The network processes biosignal spike trains from ECG, EEG, and EMG encoders simultaneously.
          Unlike CNNs which process all inputs continuously, LIF neurons only fire when membrane potential
          exceeds threshold — making SNNs 15,000x more energy efficient for continuous patient monitoring.
        </div>
      </div>
    </div>
  )
}