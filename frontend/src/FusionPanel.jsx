import { MOD, MOD_ORDER, fmtClock } from './api'

export default function FusionPanel({ fusion, recentEvents, detected }) {
  const byMod = fusion?.by_modality || {}
  const total = MOD_ORDER.reduce((s, m) => s + (byMod[m] || 0), 0)
  return (
    <div className="panel flow fill">
      <div className="panel-head">
        <div className="logo">⊹</div>
        <div><h1>SENSOR FUSION</h1></div>
      </div>
      <div className="panel-body">
        {!detected && <div className="hint">Awaiting first detection…</div>}

        {detected && (
          <>
            <div className="panel-title">Contribution by modality</div>
            <div className="modbars">
              {MOD_ORDER.map((m) => {
                const c = byMod[m] || 0
                const pct = total ? (c / total) * 100 : 0
                return (
                  <div className="mod" key={m}>
                    <span className="mdot" style={{ background: MOD[m].color }} />
                    <span className="mname">{MOD[m].label}</span>
                    <span className="mtrack"><span className="mfill" style={{ width: `${pct}%`, background: MOD[m].color }} /></span>
                    <span className="mcount">{Math.round(pct)}%</span>
                  </div>
                )
              })}
            </div>
            {(byMod.rf || 0) === 0 && (
              <div className="hint" style={{ marginTop: 6, color: '#fc8181' }}>RF: 0 — consistent with a fiber-optic drone (no radio emission).</div>
            )}
            {fusion?.n_clutter > 0 && (
              <div className="hint" style={{ marginTop: 4 }}>{fusion.n_clutter} false positives ignored by the model.</div>
            )}

            <div className="divider" />
            <div className="panel-title">Live detections</div>
            <div className="feed">
              {recentEvents.length === 0 && <div className="hint">…</div>}
              {recentEvents.map((e, i) => (
                <div key={i} className={`ev ${e.is_clutter ? 'clutter' : ''}`}
                  style={{ borderLeftColor: MOD[e.modality]?.color || '#888' }}>
                  <span className="em" style={{ color: MOD[e.modality]?.color }}>{MOD[e.modality]?.label}</span>
                  <span className="et">{Math.round(e.confidence * 100)}% · {fmtClock(e.t)}</span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
