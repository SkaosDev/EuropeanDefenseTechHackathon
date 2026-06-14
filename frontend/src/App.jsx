import { useEffect, useMemo, useState } from 'react'
import MapView from './MapView.jsx'
import SetupPanel from './SetupPanel.jsx'
import FusionPanel from './FusionPanel.jsx'
import PredictionPanel, { GroundTruthCard } from './PredictionPanel.jsx'
import InterventionPanel from './InterventionPanel.jsx'
import { getJSON, postSpawn, ZONE_COLORS, ZONE_LABELS, MOD, MOD_ORDER, fmtClock } from './api'
import { useStream } from './useStream'

export default function App() {
  const [targets, setTargets] = useState([])
  const [origins, setOrigins] = useState([])
  const [classes, setClasses] = useState([])
  const [sensors, setSensors] = useState([])
  const [dasLines, setDasLines] = useState([])
  const [assets, setAssets] = useState([])
  const [borders, setBorders] = useState(null)
  const [form, setForm] = useState({ drone_class: '', target: '', origin: '', speed: 'auto', prefer_hit: false, seed: null })
  const [spawn, setSpawn] = useState(null)
  const [session, setSession] = useState(null)
  const [view, setView] = useState('setup')
  const [busy, setBusy] = useState(false)
  const [firedRaw, setFiredRaw] = useState([])
  const [showSensors, setShowSensors] = useState(false)

  const { tick: live } = useStream(session)

  useEffect(() => {
    (async () => {
      try {
        const [t, o, c] = await Promise.all([getJSON('/targets'), getJSON('/origins'), getJSON('/classes')])
        setTargets(t); setOrigins(o); setClasses(c)
      } catch (e) { console.error('metadata load failed', e) }
      try { setSensors(await getJSON('/sensors')) } catch { /* optional */ }
      try { setDasLines(await getJSON('/das_lines')) } catch { /* optional */ }
      try { setAssets(await getJSON('/assets')) } catch { /* optional */ }
      try { const r = await fetch('/borders.geojson'); if (r.ok) setBorders(await r.json()) } catch { /* fallback */ }
    })()
  }, [])

  useEffect(() => {
    if (live?.new_events?.length) setFiredRaw((prev) => [...prev, ...live.new_events].slice(-60))
  }, [live])

  const classLabel = useMemo(() => Object.fromEntries(classes.map((c) => [c.name, c.label])), [classes])
  const detected = (live?.n_events ?? 0) > 0
  const ivActive = ['ASSESSING', 'ENGAGED'].includes(live?.intervention?.state)
  const fired = useMemo(() => {
    const a = firedRaw.slice(-40); const n = a.length
    return a.map((e, i) => ({ ...e, fade: n <= 1 ? 0.9 : 0.12 + 0.88 * (i / (n - 1)) }))
  }, [firedRaw])
  const recentEvents = useMemo(() => firedRaw.slice(-12).reverse(), [firedRaw])

  function preset(body) {
    // Graine fixée -> scénario reproductible (prefer_hit inutile) ; sinon best-of-seeds.
    setForm((f) => ({ ...f, drone_class: body.drone_class || '', target: body.target || '',
      origin: body.origin || '', seed: body.seed ?? null, prefer_hit: body.seed == null }))
  }
  const pickOrigin = (name) => setForm((f) => ({ ...f, origin: name, prefer_hit: false }))
  const pickTarget = (name) => setForm((f) => ({ ...f, target: name, prefer_hit: false }))

  async function launch() {
    const body = {
      drone_class: form.drone_class || undefined,
      target: form.target || undefined,
      origin: form.origin || undefined,
      seed: form.seed ?? undefined,
      prefer_hit: !!form.prefer_hit,
    }
    setBusy(true)
    setSpawn(null); setSession(null); setFiredRaw([])
    try {
      const info = await postSpawn(body)
      setSpawn(info)
      setSession({ scenario_id: info.scenario_id, speed: form.speed })
      setView('live')
    } catch (e) { alert('Spawn failed: ' + e.message) } finally { setBusy(false) }
  }

  function reconfigure() {
    setSession(null); setSpawn(null); setFiredRaw([]); setView('setup')
  }

  return (
    <div className="app">
      <MapView
        targets={targets} origins={origins} sensors={sensors} dasLines={dasLines}
        borders={borders} spawn={spawn} live={live} fired={fired} detected={detected}
        showSensors={showSensors} onPickOrigin={pickOrigin} onPickTarget={pickTarget}
        assets={assets} intervention={live?.intervention}
      />

      {view === 'setup' && (
        <SetupPanel targets={targets} origins={origins} classes={classes}
          form={form} setForm={setForm} onPreset={preset} onLaunch={launch} busy={busy} />
      )}

      {view === 'live' && (
        <>
          <div className="hud">
            <button className="back" onClick={reconfigure}>← Reconfigure</button>
            <div className="item"><div className="k">Mission clock</div>
              <div className="v">{fmtClock(live?.clock)} / {fmtClock(spawn?.t_max)}</div></div>
          </div>
          <div className="panel-col left">
            <FusionPanel fusion={live?.fusion} recentEvents={recentEvents} detected={detected} />
            <GroundTruthCard spawn={spawn} />
          </div>
          <div className="panel-col right">
            <PredictionPanel spawn={spawn} live={live} classLabel={classLabel}
              detected={detected} compact={ivActive} />
            <InterventionPanel live={live} />
          </div>
        </>
      )}

      {/* Légende + toggle capteurs, en bas à droite de la carte */}
      <div className="layers-box" style={{ right: view === 'live' ? 346 : 14 }}>
        <button className={`toggle ${showSensors ? 'on' : ''}`} onClick={() => setShowSensors((v) => !v)}>
          <span className="sw" /> Sensors {showSensors ? 'on' : 'off'}
        </button>
        {showSensors && (
          <>
            <div className="legend-cap">Sensors</div>
            <div className="legend">
              {MOD_ORDER.map((m) => (
                <div className="item" key={m}><span className="dot" style={{ background: MOD[m].color }} />{MOD[m].label}</div>
              ))}
            </div>
            <div className="legend-sep" />
          </>
        )}
        <div className="legend-cap">Targets</div>
        <div className="legend">
          {Object.keys(ZONE_LABELS).map((z) => (
            <div className="item" key={z}><span className="dot tri" style={{ background: ZONE_COLORS[z] }} />{ZONE_LABELS[z]}</div>
          ))}
        </div>
      </div>
    </div>
  )
}
