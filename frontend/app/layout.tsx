import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'NeuroPulse AI — Neuromorphic Healthcare Platform',
  description: 'Multi-Modal Biosignal Monitoring and Anomaly Detection Using SNNs',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}