/**
 * WebSocket Client for NeuroPulse AI
 * Handles live biosignal streaming.
 */

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8001'

export class NeuroPulseWebSocket {
  private ws:        WebSocket | null = null
  private patientId: string
  private onMessage: (data: any) => void
  private onConnect: () => void
  private onDisconnect: () => void
  private reconnectTimer: any = null

  constructor(
    patientId:    string,
    onMessage:    (data: any) => void,
    onConnect:    () => void = () => {},
    onDisconnect: () => void = () => {}
  ) {
    this.patientId    = patientId
    this.onMessage    = onMessage
    this.onConnect    = onConnect
    this.onDisconnect = onDisconnect
  }

  connect() {
    try {
      this.ws = new WebSocket(`${WS_BASE}/ws/${this.patientId}`)

      this.ws.onopen = () => {
        console.log(`WebSocket connected: patient ${this.patientId}`)
        this.onConnect()
        if (this.reconnectTimer) {
          clearTimeout(this.reconnectTimer)
          this.reconnectTimer = null
        }
      }

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          this.onMessage(data)
        } catch (e) {
          console.error('WebSocket parse error:', e)
        }
      }

      this.ws.onclose = () => {
        console.log('WebSocket disconnected')
        this.onDisconnect()
        // Reconnect after 3 seconds
        this.reconnectTimer = setTimeout(() => {
          this.connect()
        }, 3000)
      }

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error)
      }

    } catch (e) {
      console.error('WebSocket connection failed:', e)
    }
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
    }
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }
}