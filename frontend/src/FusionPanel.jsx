import { MOD, MOD_ORDER } from './api'

export default function FusionPanel({ fusion, recentEvents, detected }) {
  const byMod = fusion?.by_modality || {}
  const maxC = Math.max(1, ...MOD_ORDER.map((m) => byMod[m] || 0))
  return (
    <div className="panel left">
      <div className="panel-head">
        <div className="logo">⊹</div>
        <div><h1>SENSOR FUSION</h1><div className="sub">Multi-modal detection → single track</div></div>
      </div>
      <div className="panel-body">
        {!detected && <div className="hint">No sensor contact yet — the drone is en route but outside
          Ukrainian sensor coverage. Detection begins as it approaches the territory.</div>}

        {detected && (
          <>
            <div className="fusion-hero">
              <div className="big">{fusion?.n_sensors ?? 0}<span className="arrow">·</span>{fusion?.n_modalities ?? 0}<span className="arrow">→</span>1</div>
              <div className="lbl">sensors · modalities → fused track</div>
            </div>

            <div className="panel-title">Contribution by modality</div>
            <div className="modbars">
              {MOD_ORDER.map((m) => {
                const c = byMod[m] || 0
                return (
                  <div className="mod" key={m}>
                    <span className="mdot" style={{ background: MOD[m].color }} />
                    <span className="mname">{MOD[m].label}</span>
                    <span className="mtrack"><span className="mfill" style={{ width: `${(c / maxC) * 100}%`, background: MOD[m].color }} /></span>
                    <span className="mcount">{c}</span>
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
                  <span>{MOD[e.modality]?.glyph}</span>
                  <span className="em" style={{ color: MOD[e.modality]?.color }}>{MOD[e.modality]?.label}</span>
                  <span>{e.is_clutter ? 'clutter' : e.est_class}</span>
                  <span className="et">{Math.round(e.confidence * 100)}% · {Math.round(e.t)}s</span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
