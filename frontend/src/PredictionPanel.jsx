export default function PredictionPanel({ spawn, live, classLabel, detected, compact = false }) {
  const pred = detected ? live?.prediction : null
  const topk = pred?.target_topk || []
  const shown = compact ? topk.slice(0, 3) : topk
  return (
    <div className={`panel flow ${compact ? 'card' : 'fill'}`}>
      <div className="panel-head">
        <div className="logo" style={{ background: 'linear-gradient(135deg,#ff4d5e,#a01f2b)', color: '#fff' }}>!</div>
        <div><h1>THREAT ASSESSMENT</h1></div>
      </div>
      <div className="panel-body">
        <div className="stat-row">
          <span className="k">Estimated class</span>
          <span className="badge cls">{pred ? `${classLabel[pred.pred_class] || pred.pred_class} · ${Math.round(pred.pred_class_p * 100)}%` : '—'}</span>
        </div>

        <div className="panel-title" style={{ marginTop: 12 }}>Estimated targets</div>
        {!detected && <div className="hint">Awaiting first detection…</div>}
        {shown.map((p) => {
          const correct = spawn && p.dest_id === spawn.true_dest_id
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
    </div>
  )
}

// Carte vérité-terrain (overlay), rendue séparément pour la placer SOUS l'interception.
export function GroundTruthCard({ spawn }) {
  if (!spawn) return null
  return (
    <div className="panel flow card mission-card">
      <div className="mission-head">
        <span className="mission-tag">GROUND TRUTH</span>
      </div>
      <div className="mission-rows">
        <div className="mrow">
          <span className="mk">Class</span>
          <span className="mv">{spawn.drone_class_label}</span>
        </div>
        <div className="mrow">
          <span className="mk">Origin</span>
          <span className="mv">{spawn.origin}</span>
        </div>
        <div className="mrow target">
          <span className="mk">Target</span>
          <span className="mv">{spawn.true_dest_name}</span>
        </div>
      </div>
    </div>
  )
}
