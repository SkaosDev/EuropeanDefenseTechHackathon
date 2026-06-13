// Le backend FastAPI écoute sur :8000 (CORS ouvert). URLs absolues -> pas de proxy.
const HOST = window.location.hostname || 'localhost'
export const BACKEND = `http://${HOST}:8000`
export const WS_BACKEND = `ws://${HOST}:8000`

export async function getJSON(path) {
  const r = await fetch(BACKEND + path)
  if (!r.ok) throw new Error(`GET ${path} -> ${r.status}`)
  return r.json()
}

export async function postSpawn(body) {
  const r = await fetch(BACKEND + '/spawn', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`POST /spawn -> ${r.status}`)
  return r.json()
}

export const ZONE_COLORS = {
  city: '#dfe6ee',
  power_tpp: '#ffb74d',
  power_hpp: '#4fc3f7',
  power_npp: '#ff5252',
  airbase: '#b388ff',
  port: '#4db6ac',
  defense_industry: '#fff176',
}
export const ZONE_LABELS = {
  city: 'City', power_npp: 'Nuclear plant', airbase: 'Airbase', port: 'Port',
  defense_industry: 'Defense industry',
}

// Modalités de capteurs : couleur, libellé, glyphe (icône au zoom).
export const MOD = {
  acoustic: { color: '#9ae6b4', label: 'Acoustic', glyph: '🎙' },
  optical: { color: '#7fd1ff', label: 'Optical', glyph: '📷' },
  rf: { color: '#fc8181', label: 'RF / EW', glyph: '📡' },
  das: { color: '#f6ad55', label: 'DAS fiber', glyph: '〜' },
  observer: { color: '#d6bcfa', label: 'Observer', glyph: '👁' },
}
export const MOD_ORDER = ['acoustic', 'optical', 'rf', 'das', 'observer']
