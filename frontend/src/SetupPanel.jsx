import { ZONE_COLORS, ZONE_LABELS } from './api'

const PRESETS = [
  { pt: 'Shahed-136 · North → Kyiv', pd: 'Long-range strike on the capital',
    body: { drone_class: 'shahed136', target: 'Kyiv', origin: 'Seshcha', prefer_hit: true } },
  { pt: 'Shahed-136 · Black Sea → Odesa port', pd: 'Port / grain-corridor campaign',
    body: { drone_class: 'shahed136', target: 'Pivdennyi port', origin: 'Primorsko-Akhtarsk', prefer_hit: true } },
  { pt: 'Lancet · front airbase (Mykolaiv)', pd: 'Loitering munition vs aviation',
    body: { drone_class: 'lancet', target: 'Kulbakino AB', prefer_hit: true } },
  { pt: 'FPV fibre · front (Kharkiv) — 0 RF', pd: 'Fiber-optic: invisible to RF',
    body: { drone_class: 'fpv_fiber', target: 'Kharkiv', prefer_hit: true } },
]

export default function SetupPanel({ targets, origins, classes, form, setForm, onLaunch, busy }) {
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value })
  return (
    <div className="panel left">
      <div className="panel-head">
        <div className="logo">Δ</div>
        <div><h1>AEGIS</h1><div className="sub">Mission setup</div></div>
      </div>
      <div className="panel-body">
        <div className="panel-title">Typical real-world scenarios</div>
        <div className="presets">
          {PRESETS.map((p) => (
            <button key={p.pt} className="preset" disabled={busy} onClick={() => onLaunch(p.body)}>
              <div className="pt">{p.pt}</div><div className="pd">{p.pd}</div>
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
          <label>Target</label>
          <select value={form.target} onChange={set('target')}>
            <option value="">— auto (plausible) —</option>
            {targets.map((t) => <option key={t.dest_id} value={t.name}>{t.name} · {t.zone_type}</option>)}
          </select>
        </div>
        <div className="field">
          <label>Origin (launch site)</label>
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
        <button className="btn" disabled={busy} onClick={() => onLaunch()}>
          {busy ? 'SPAWNING…' : 'LAUNCH SCENARIO'}
        </button>

        <div className="divider" />
        <div className="hint">The model sees only noisy sensor detections (clutter included) — never the
          true target. It stays blind until the drone is detected on Ukrainian territory.</div>
        <div className="panel-title" style={{ marginTop: 14 }}>Target types</div>
        <div className="legend">
          {Object.keys(ZONE_LABELS).map((z) => (
            <div className="item" key={z}><span className="dot" style={{ background: ZONE_COLORS[z] }} />{ZONE_LABELS[z]}</div>
          ))}
        </div>
      </div>
    </div>
  )
}
