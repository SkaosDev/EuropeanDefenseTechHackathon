import { useEffect, useMemo, useState } from 'react'
import MapView from './MapView.jsx'
import ControlPanel from './ControlPanel.jsx'
import { getJSON, postSpawn } from './api'
import { useStream } from './useStream'

export default function App() {
  const [targets, setTargets] = useState([])
  const [origins, setOrigins] = useState([])
  const [classes, setClasses] = useState([])
  const [borders, setBorders] = useState(null)
  const [form, setForm] = useState({ drone_class: '', target: '', origin: '', speed: 'auto' })
  const [spawn, setSpawn] = useState(null)
  const [session, setSession] = useState(null)
  const [busy, setBusy] = useState(false)
  const [firedRaw, setFiredRaw] = useState([])

  const { tick: live, status } = useStream(session)

  useEffect(() => {
    (async () => {
      try {
        const [t, o, c] = await Promise.all([
          getJSON('/targets'), getJSON('/origins'), getJSON('/classes'),
        ])
        setTargets(t); setOrigins(o); setClasses(c)
      } catch (e) { console.error('metadata load failed', e) }
      try {
        const r = await fetch('/borders.geojson')
        if (r.ok) setBorders(await r.json())
      } catch { /* fallback: no borders layer */ }
    })()
  }, [])

  // Accumule les détections capteur (pour l'effet "bruit observé" sur la carte).
  useEffect(() => {
    if (live?.new_events?.length) setFiredRaw((prev) => [...prev, ...live.new_events])
  }, [live])

  const fired = useMemo(() => {
    const a = firedRaw.slice(-250)
    const n = a.length
    return a.map((e, i) => ({ ...e, fade: n <= 1 ? 0.9 : 0.2 + 0.8 * (i / (n - 1)) }))
  }, [firedRaw])

  async function launch(override) {
    if (override) {
      setForm((f) => ({
        ...f, drone_class: override.drone_class || '',
        target: override.target || '', origin: override.origin || '',
      }))
    }
    const body = override || {
      drone_class: form.drone_class || undefined,
      target: form.target || undefined,
      origin: form.origin || undefined,
    }
    setBusy(true)
    setSpawn(null); setSession(null); setFiredRaw([])
    try {
      const info = await postSpawn(body)
      setSpawn(info)
      setSession({ scenario_id: info.scenario_id, speed: form.speed })
    } catch (e) {
      alert('Spawn failed: ' + e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <ControlPanel
        targets={targets} origins={origins} classes={classes}
        form={form} setForm={setForm} onLaunch={launch} busy={busy}
        spawn={spawn} live={live} status={status}
      />
      <MapView
        targets={targets} origins={origins} borders={borders}
        spawn={spawn} live={live} fired={fired}
      />
    </div>
  )
}
