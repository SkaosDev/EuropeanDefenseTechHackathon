import { useEffect, useMemo } from 'react'
import {
  MapContainer, TileLayer, GeoJSON, CircleMarker, Polyline, Marker, Tooltip, useMap,
} from 'react-leaflet'
import L from 'leaflet'
import { ZONE_COLORS } from './api'

const MOD_COLORS = {
  optical: '#7fd1ff', acoustic: '#9ae6b4', vibration: '#d6bcfa',
  das: '#f6ad55', rf: '#fc8181',
}

const droneIcon = L.divIcon({ className: '', html: '<div class="drone-icon"></div>',
  iconSize: [16, 16], iconAnchor: [8, 8] })
const originIcon = L.divIcon({ className: '',
  html: '<div style="width:11px;height:11px;background:#ffd24d;transform:rotate(45deg);border:1px solid #7a5a00"></div>',
  iconSize: [11, 11], iconAnchor: [6, 6] })

// Recadre la carte sur le segment origine→cible au lancement d'un scénario.
function FitOnSpawn({ spawn }) {
  const map = useMap()
  useEffect(() => {
    if (!spawn?.ground_truth?.length) return
    const pts = spawn.ground_truth.map((r) => [r.lat, r.lon])
    pts.push([spawn.true_dest_lat, spawn.true_dest_lon])
    map.fitBounds(L.latLngBounds(pts).pad(0.25), { animate: true })
  }, [spawn, map])
  return null
}

export default function MapView({ targets, origins, borders, spawn, live, fired }) {
  const targetById = useMemo(() => {
    const m = {}
    for (const t of targets) m[t.dest_id] = t
    return m
  }, [targets])

  const clock = live?.clock ?? 0
  const gt = spawn?.ground_truth || []
  const { traveled, remaining } = useMemo(() => {
    const tr = [], rem = []
    for (const r of gt) (r.t <= clock ? tr : rem).push([r.lat, r.lon])
    if (tr.length && rem.length) rem.unshift(tr[tr.length - 1])
    return { traveled: tr, remaining: rem }
  }, [gt, clock])

  const dronePos = live?.drone_pos || (gt[0] ? [gt[0].lat, gt[0].lon] : null)
  const topk = live?.prediction?.target_topk || []
  const future = live?.prediction?.pred_future || []
  const trueDestId = spawn?.true_dest_id

  return (
    <div className="map-wrap">
      <MapContainer center={[48.6, 31.5]} zoom={6} zoomControl={false} preferCanvas>
        {/* Fond sombre CARTO ; la couche frontières (vendue localement) reste visible si les tuiles ne chargent pas. */}
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution="&copy; OpenStreetMap &copy; CARTO" subdomains="abcd" maxZoom={19}
        />
        {borders && (
          <GeoJSON data={borders}
            style={{ color: '#33415c', weight: 1, fillColor: '#0e1422', fillOpacity: 0.55 }} />
        )}

        {/* Origines (sites de lancement) */}
        {origins.map((o) => (
          <Marker key={o.name} position={[o.lat, o.lon]} icon={originIcon}>
            <Tooltip>{o.name} <span style={{ color: '#888' }}>· launch site</span></Tooltip>
          </Marker>
        ))}

        {/* Cibles sensibles */}
        {targets.map((t) => {
          const isTrue = spawn && t.dest_id === trueDestId
          return (
            <CircleMarker key={t.dest_id} center={[t.lat, t.lon]}
              radius={isTrue ? 7 : 4}
              pathOptions={{
                color: isTrue ? '#35d0d6' : (ZONE_COLORS[t.zone_type] || '#aaa'),
                weight: isTrue ? 2 : 1, fillColor: ZONE_COLORS[t.zone_type] || '#aaa',
                fillOpacity: isTrue ? 0.9 : 0.55,
              }}>
              <Tooltip>{t.name} · {t.zone_type}</Tooltip>
            </CircleMarker>
          )
        })}

        {/* Capteurs ayant tiré (bruit observé, clutter inclus) */}
        {fired.map((e, i) => (
          <CircleMarker key={i} center={[e.sensor_lat, e.sensor_lon]} radius={e.is_clutter ? 3 : 4}
            pathOptions={{
              color: e.is_clutter ? '#5a6577' : (MOD_COLORS[e.modality] || '#fff'),
              weight: 1, opacity: e.fade, fillOpacity: e.fade * 0.5,
              dashArray: e.is_clutter ? '2,2' : null,
            }} />
        ))}

        {spawn && (
          <>
            <FitOnSpawn spawn={spawn} />
            {remaining.length > 1 && (
              <Polyline positions={remaining}
                pathOptions={{ color: '#9fb6ff', weight: 2, opacity: 0.35, dashArray: '5,7' }} />
            )}
            {traveled.length > 1 && (
              <Polyline positions={traveled} pathOptions={{ color: '#2b6cff', weight: 4, opacity: 0.95 }} />
            )}

            {/* Vecteurs de menace : drone -> top-k cibles, épaisseur/opacité ∝ probabilité */}
            {dronePos && topk.map((p, i) => {
              const t = targetById[p.dest_id]
              if (!t) return null
              const correct = p.dest_id === trueDestId
              return (
                <Polyline key={p.dest_id} positions={[dronePos, [t.lat, t.lon]]}
                  pathOptions={{
                    color: correct ? '#35d0d6' : '#ff4d5e',
                    weight: 1.5 + 9 * p.p, opacity: 0.22 + 0.7 * p.p,
                  }}>
                  {i < 3 && (
                    <Tooltip permanent direction="center" className="vec-label" opacity={1}>
                      {`${Math.round(p.p * 100)}% ${t.name}`}
                    </Tooltip>
                  )}
                </Polyline>
              )
            })}

            {/* Trajectoire future prédite (tête b) */}
            {future.length > 1 && (
              <Polyline positions={future}
                pathOptions={{ color: '#ffb84d', weight: 2, opacity: 0.7, dashArray: '2,6' }} />
            )}

            {dronePos && <Marker position={dronePos} icon={droneIcon} zIndexOffset={1000} />}
          </>
        )}
      </MapContainer>
      <div className="offline-badge">map tiles via CARTO · borders cached locally</div>
    </div>
  )
}
