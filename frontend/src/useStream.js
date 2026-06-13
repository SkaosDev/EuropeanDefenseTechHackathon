import { useEffect, useRef, useState } from 'react'
import { WS_BACKEND } from './api'

// Ouvre un WebSocket /stream pour la session courante et renvoie le dernier tick reçu :
//   { type, clock, drone_pos:[lat,lon], n_events, new_events:[...], prediction:{...} }
// La session = { scenario_id, speed }. Un changement de session rouvre un nouveau flux.
export function useStream(session) {
  const [tick, setTick] = useState(null)
  const [status, setStatus] = useState('idle') // idle | live | done | error
  const wsRef = useRef(null)

  useEffect(() => {
    if (!session) return
    setTick(null)
    setStatus('connecting')
    const ws = new WebSocket(
      `${WS_BACKEND}/stream?scenario_id=${session.scenario_id}&speed=${session.speed}`,
    )
    wsRef.current = ws
    ws.onopen = () => setStatus('live')
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data)
      if (m.type === 'tick') setTick(m)
      else if (m.type === 'done') setStatus('done')
      else if (m.type === 'error') setStatus('error')
    }
    ws.onerror = () => setStatus('error')
    return () => ws.close()
  }, [session])

  return { tick, status }
}
