// Scénarios de démo calibrés (graine fixée -> piste précise + interception viable).
const PRESETS = [
  { pt: 'Shahed-136 · → Kharkiv', pd: 'best autonomous case',
    body: { drone_class: 'shahed136', target: 'Kharkiv', seed: 1 } },
  { pt: 'Shahed-136 · → Sumy',
    body: { drone_class: 'shahed136', target: 'Sumy', seed: 1 } },
  { pt: 'Shahed-136 · → Lviv',
    body: { drone_class: 'shahed136', target: 'Lviv', seed: 5 } },
]

export default function SetupPanel({ targets, origins, classes, form, setForm, onPreset, onLaunch, busy }) {
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value, prefer_hit: false, seed: null })
  return (
    <div className="panel left">
      <div className="panel-head">
        <div className="logo">Δ</div>
        <div><h1>TOIS</h1></div>
      </div>
      <div className="panel-body">
        <div className="panel-title">Demo scenarios</div>
        <div className="presets">
          {PRESETS.map((p) => (
            <button key={p.pt} className="preset" disabled={busy} onClick={() => onPreset(p.body)}>
              <div className="pt">{p.pt}</div>
            </button>
          ))}
        </div>

        <div className="divider" />
        <div className="panel-title">Custom launch</div>
        <div className="field">
          <label>Drone class</label>
          <select value={form.drone_class} onChange={set('drone_class')}>
            <option value="">— random —</option>
            {classes.map((c) => <option key={c.name} value={c.name}>{c.label}</option>)}
          </select>
        </div>
        <div className="field">
          <label>Target <span style={{ color: '#7b8aa3' }}>(or click the map)</span></label>
          <select value={form.target} onChange={set('target')}>
            <option value="">— auto (plausible) —</option>
            {targets.map((t) => <option key={t.dest_id} value={t.name}>{t.name} · {t.zone_type}</option>)}
          </select>
        </div>
        <div className="field">
          <label>Origin <span style={{ color: '#7b8aa3' }}>(or click a launch site)</span></label>
          <select value={form.origin} onChange={set('origin')}>
            <option value="">— auto —</option>
            {origins.map((o) => <option key={o.name} value={o.name}>{o.name}</option>)}
          </select>
        </div>
        <div className="field">
          <label>Playback speed <span className="range-val">{form.speed === 'auto' ? 'auto' : `${form.speed}×`}</span></label>
          <input type="range" min="50" max="1500" step="50"
            value={form.speed === 'auto' ? 400 : form.speed}
            onChange={(e) => setForm({ ...form, speed: Number(e.target.value) })} />
        </div>
        <button className="btn" disabled={busy} onClick={onLaunch}>
          {busy ? 'SPAWNING…' : 'LAUNCH SCENARIO'}
        </button>
      </div>
    </div>
  )
}
