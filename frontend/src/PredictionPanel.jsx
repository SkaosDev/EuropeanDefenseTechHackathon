export default function PredictionPanel({ spawn, live, classLabel, detected }) {
  const pred = detected ? live?.prediction : null
  const topk = pred?.target_topk || []
  return (
    <div className="panel right">
      <div className="panel-head">
        <div className="logo" style={{ background: 'linear-gradient(135deg,#ff4d5e,#a01f2b)', color: '#fff' }}>!</div>
        <div><h1>THREAT ASSESSMENT</h1><div className="sub">Predicted target</div></div>
      </div>
      <div className="panel-body">
        <div className="stat-row">
          <span className="k">Estimated class</span>
          <span className="badge cls">{pred ? `${classLabel[pred.pred_class] || pred.pred_class} · ${Math.round(pred.pred_class_p * 100)}%` : '—'}</span>
        </div>

        <div className="panel-title" style={{ marginTop: 12 }}>Most likely targets</div>
        {!detected && <div className="hint">Awaiting first detection…</div>}
        {topk.map((p) => {
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

        {spawn && (
          <div className="truth">
            Ground truth (hidden from model):<br />
            <b>{spawn.drone_class_label}</b> from <b>{spawn.origin}</b><br />→ <b>{spawn.true_dest_name}</b>
          </div>
        )}
      </div>
    </div>
  )
}
