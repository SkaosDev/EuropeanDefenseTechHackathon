// Aide à la décision d'interception (consultatif) : sites capables d'intercepter la menace
// prédite, classés par probabilité de réussite estimée, et bascule en engagement autonome
// figé quand la solution est très fiable et que la menace entre dans le rayon de la ville.

export default function InterventionPanel({ live }) {
  const iv = live?.intervention
  if (!iv) return null

  if (iv.state === 'MONITORING') {
    return (
      <div className="panel flow card iv-card">
        <div className="panel-head iv-head-row">
          <div className="logo iv-logo">◎</div>
          <div><h1>INTERCEPTION</h1></div>
        </div>
        <div className="panel-body">
          <div className="hint">Awaiting threat detection…</div>
        </div>
      </div>
    )
  }

  if (iv.state === 'ENGAGED') {
    const e = iv.engaged || {}
    return (
      <div className="panel flow fill iv-card iv-engaged">
        <div className="iv-eng-banner">✓ AUTONOMOUS INTERCEPTION ENGAGED</div>
        <div className="panel-body">
          <div className="iv-eng-box">
            <div className="mission-rows">
              <div className="mrow target"><span className="mk">Defending</span><span className="mv">{e.target_name}</span></div>
              <div className="mrow"><span className="mk">Salvo</span><span className="mv">{e.chosen?.n_interceptors}× interceptor{e.chosen?.n_interceptors > 1 ? 's' : ''}</span></div>
              <div className="mrow"><span className="mk">Est. success</span><span className="mv">{Math.round((e.best_p || 0) * 100)}%</span></div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // ASSESSING
  const th = iv.threat || {}
  const v = iv.verdict || {}
  const opts = iv.options || []
  const pMax = Math.max(0.0001, ...opts.map((o) => o.p_success || 0))

  return (
    <div className="panel flow fill iv-card">
      <div className="panel-head iv-head-row">
        <div className="logo iv-logo">◎</div>
        <div><h1>INTERCEPTION</h1></div>
      </div>

      <div className="panel-body">
        <div className="iv-verdict">
          {v.autonomous_viable
            ? <span className="iv-verdict-ok">✓ AUTONOMOUS INTERCEPTION VIABLE</span>
            : <span className="iv-verdict-na">INTERCEPTION OPTIONS</span>}
          {v.best_p != null && <span className="iv-verdict-p">{Math.round(v.best_p * 100)}%</span>}
        </div>

        <div className="stat-row" style={{ marginTop: 12 }}>
          <span className="k">Estimated class</span>
          <span className="badge cls">{th.pred_class} · {Math.round((th.pred_class_p || 0) * 100)}%</span>
        </div>
        <div className="stat-row">
          <span className="k">Estimated target</span>
          <span className="v">{th.target_name}</span>
        </div>

        <div className="panel-title" style={{ marginTop: 12 }}>Launch options</div>
        {opts.length === 0 && <div className="hint">No solution in range.</div>}
        {opts.map((o) => (
          <div key={o.site_id} className="iv-cand">
            <div className="iv-cand-bar" style={{ width: `${Math.max(6, (o.p_success / pMax) * 100)}%` }} />
            <div className="iv-cand-row">
              <span className="iv-cand-city">{o.site_name}</span>
              <span className="iv-cand-p">{Math.round(o.p_success * 100)}%</span>
            </div>
            <div className="iv-cand-meta">{o.n_interceptors}× interceptor{o.n_interceptors > 1 ? 's' : ''}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
