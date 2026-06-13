import { useMemo } from 'react'
import { ZONE_COLORS, ZONE_LABELS } from './api'

const PRESETS = [
  { pt: 'Shahed-136 · North → Kyiv', pd: 'Long-range one-way attack drone',
    body: { drone_class: 'shahed136', target: 'Kyiv', origin: 'Seshcha', prefer_hit: true } },
  { pt: 'Gerbera wave · South → Odesa', pd: 'Decoys saturating air defense',
    body: { drone_class: 'gerbera', target: 'Odesa', origin: 'Primorsko-Akhtarsk', prefer_hit: true } },
  { pt: 'FPV fibre · near the front', pd: 'Short-range, zero RF signature',
    body: { drone_class: 'fpv_fiber', target: 'Kharkiv', prefer_hit: true } },
]

export default function ControlPanel({
  targets, origins, classes, form, setForm, onLaunch, busy, spawn, live, status,
}) {
  const classLabel = useMemo(() => {
    const m = {}
    for (const c of classes) m[c.name] = c.label
    return m
  }, [classes])

  const pred = live?.prediction
  const topk = pred?.target_topk || []
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value })

  return (
    <div className="sidebar">
      <div className="brand">
        <div className="logo">Δ</div>
        <div>
          <h1>AEGIS</h1>
          <div className="sub">Counter-UAS early warning</div>
        </div>
      </div>

      <div className="scroll">
        <div className="section-title">Demo scenarios</div>
        <div className="presets">
          {PRESETS.map((p) => (
            <button key={p.pt} className="preset" disabled={busy}
              onClick={() => onLaunch(p.body)}>
              <div className="pt">{p.pt}</div>
              <div className="pd">{p.pd}</div>
            </button>
          ))}
        </div>

        <div className="divider" />
        <div className="section-title">Custom launch</div>
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
        <div className="section-title">
          <span className={`status-dot ${status === 'live' ? 'live' : status === 'done' ? 'done' : 'idle'}`} />
          Live track
        </div>

        {!spawn && <div className="hint">Launch a scenario to begin. The model sees only noisy
          sensor detections (clutter included) — never the true target — and predicts where the
          drone is heading, tightening as more events arrive.</div>}

        {spawn && (
          <div className="telemetry">
            <div className="stat-row">
              <span className="k">Mission clock</span>
              <span className="v">{fmtClock(live?.clock)} <span style={{ color: '#7b8aa3' }}>/ {fmtClock(spawn.t_max)}</span></span>
            </div>
            <div className="stat-row">
              <span className="k">Detections observed</span>
              <span className="v">{live?.n_events ?? 0} <span style={{ color: '#7b8aa3' }}>of {spawn.n_events}</span></span>
            </div>
            <div className="stat-row">
              <span className="k">Estimated class</span>
              <span className="badge cls">
                {pred ? `${classLabel[pred.pred_class] || pred.pred_class} · ${Math.round(pred.pred_class_p * 100)}%` : '—'}
              </span>
            </div>

            <div className="pred-list">
              {topk.length === 0 && <div className="hint" style={{ marginTop: 6 }}>Awaiting first detection…</div>}
              {topk.map((p) => {
                const correct = p.dest_id === spawn.true_dest_id
                return (
                  <div key={p.dest_id} className={`pred ${correct ? 'correct' : ''}`}>
                    <div className="bar" style={{ width: `${Math.max(3, p.p * 100)}%` }} />
                    <div className="lbl">
                      <span className="nm">{correct && <span className="tick">✓</span>}{p.name}</span>
                      <span className="pc">{(p.p * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                )
              })}
            </div>

            <div className="truth">
              Ground truth (hidden from model): <b>{spawn.drone_class_label}</b> from <b>{spawn.origin}</b> → <b>{spawn.true_dest_name}</b>
            </div>
          </div>
        )}

        <div className="divider" />
        <div className="section-title">Target legend</div>
        <div className="legend">
          {Object.keys(ZONE_LABELS).map((z) => (
            <div className="item" key={z}>
              <span className="dot" style={{ background: ZONE_COLORS[z] }} />{ZONE_LABELS[z]}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function fmtClock(s) {
  if (s == null) return '—'
  const m = Math.floor(s / 60), sec = Math.floor(s % 60)
  return `${m}:${String(sec).padStart(2, '0')}`
}
