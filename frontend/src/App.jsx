import { useEffect, useMemo, useState } from 'react'
import MapView from './MapView.jsx'
import SetupPanel from './SetupPanel.jsx'
import FusionPanel from './FusionPanel.jsx'
import PredictionPanel from './PredictionPanel.jsx'
import { getJSON, postSpawn } from './api'
import { useStream } from './useStream'

function fmtClock(s) {
  if (s == null) return '—'
  const m = Math.floor(s / 60), sec = Math.floor(s % 60)
  return `${m}:${String(sec).padStart(2, '0')}`
}

export default function App() {
  const [targets, setTargets] = useState([])
  const [origins, setOrigins] = useState([])
  const [classes, setClasses] = useState([])
  const [sensors, setSensors] = useState([])
  const [dasLines, setDasLines] = useState([])
  const [borders, setBorders] = useState(null)
  const [form, setForm] = useState({ drone_class: '', target: '', origin: '', speed: 'auto' })
  const [spawn, setSpawn] = useState(null)
  const [session, setSession] = useState(null)
  const [view, setView] = useState('setup')
  const [busy, setBusy] = useState(false)
  const [firedRaw, setFiredRaw] = useState([])
  const [zoom, setZoom] = useState(6)

  const { tick: live } = useStream(session)

  useEffect(() => {
    (async () => {
      try {
        const [t, o, c] = await Promise.all([getJSON('/targets'), getJSON('/origins'), getJSON('/classes')])
        setTargets(t); setOrigins(o); setClasses(c)
      } catch (e) { console.error('metadata load failed', e) }
      try { setSensors(await getJSON('/sensors')) } catch { /* optional */ }
      try { setDasLines(await getJSON('/das_lines')) } catch { /* optional */ }
      try { const r = await fetch('/borders.geojson'); if (r.ok) setBorders(await r.json()) } catch { /* fallback */ }
    })()
  }, [])

  useEffect(() => {
    if (live?.new_events?.length) setFiredRaw((prev) => [...prev, ...live.new_events].slice(-60))
  }, [live])

  const classLabel = useMemo(() => Object.fromEntries(classes.map((c) => [c.name, c.label])), [classes])
  const detected = (live?.n_events ?? 0) > 0

  const fired = useMemo(() => {
    const a = firedRaw.slice(-40)
    const n = a.length
    return a.map((e, i) => ({ ...e, fade: n <= 1 ? 0.9 : 0.12 + 0.88 * (i / (n - 1)) }))
  }, [firedRaw])
  const recentEvents = useMemo(() => firedRaw.slice(-12).reverse(), [firedRaw])

  async function launch(override) {
    if (override) {
      setForm((f) => ({ ...f, drone_class: override.drone_class || '', target: override.target || '', origin: override.origin || '' }))
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
      setView('live')
    } catch (e) {
      alert('Spawn failed: ' + e.message)
    } finally { setBusy(false) }
  }

  function reconfigure() {
    setSession(null); setSpawn(null); setFiredRaw([]); setView('setup')
  }

  return (
    <div className="app">
      <MapView
        targets={targets} origins={origins} sensors={sensors} dasLines={dasLines}
        borders={borders} spawn={spawn} live={live} fired={fired} detected={detected}
        onZoom={setZoom}
      />

      {view === 'setup' && (
        <SetupPanel targets={targets} origins={origins} classes={classes}
          form={form} setForm={setForm} onLaunch={launch} busy={busy} />
      )}

      {view === 'live' && (
        <>
          <div className="hud">
            <button className="back" onClick={reconfigure}>← Reconfigure</button>
            <div className="item"><div className="k">Mission clock</div>
              <div className="v">{fmtClock(live?.clock)} / {fmtClock(spawn?.t_max)}</div></div>
            <div className="item"><div className="k">Detections</div>
              {detected ? <div className="v">{live?.n_events} / {spawn?.n_events}</div>
                : <div className="v nocontact">● NO CONTACT</div>}</div>
          </div>
          <FusionPanel fusion={live?.fusion} recentEvents={recentEvents} detected={detected} />
          <PredictionPanel spawn={spawn} live={live} classLabel={classLabel} detected={detected} />
        </>
      )}

      {zoom < 8 && view === 'live' && <div className="zoomhint">Zoom in to reveal individual sensors + fiber lines</div>}
    </div>
  )
}
