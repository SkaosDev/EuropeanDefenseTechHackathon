import { useEffect, useMemo } from 'react'
import {
  MapContainer, TileLayer, GeoJSON, CircleMarker, Polyline, Marker, Tooltip, useMap,
} from 'react-leaflet'
import L from 'leaflet'
import { ZONE_COLORS, MOD } from './api'

const droneIcon = (detected) => L.divIcon({
  className: '', html: `<div class="drone-icon ${detected ? '' : 'undetected'}"></div>`,
  iconSize: [16, 16], iconAnchor: [8, 8],
})
const originIcon = L.divIcon({
  className: '', html: '<div style="width:11px;height:11px;background:#ffd24d;transform:rotate(45deg);border:1px solid #7a5a00"></div>',
  iconSize: [11, 11], iconAnchor: [6, 6],
})
const targetIcon = (color, isTrue) => {
  const sz = isTrue ? 20 : 14
  return L.divIcon({
    className: '',
    html: `<div class="target-mark ${isTrue ? 'is-true' : ''}" style="--tc:${color}"></div>`,
    iconSize: [sz, sz], iconAnchor: [sz / 2, sz / 2],
  })
}
const siteIcon = (active) => L.divIcon({
  className: '', html: `<div class="iv-site ${active ? 'active' : ''}">⬡</div>`,
  iconSize: [22, 22], iconAnchor: [11, 11],
})

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

export default function MapView({
  targets, origins, sensors, dasLines, borders, spawn, live, fired, detected,
  showSensors, onPickOrigin, onPickTarget, assets, intervention,
}) {
  const targetById = useMemo(() => Object.fromEntries(targets.map((t) => [t.dest_id, t])), [targets])
  const clock = live?.clock ?? 0
  const gt = spawn?.ground_truth || []
  const { traveled, remaining } = useMemo(() => {
    const tr = [], rem = []
    for (const r of gt) (r.t <= clock ? tr : rem).push([r.lat, r.lon])
    if (tr.length && rem.length) rem.unshift(tr[tr.length - 1])
    return { traveled: tr, remaining: rem }
  }, [gt, clock])

  const dronePos = live?.drone_pos || (gt[0] ? [gt[0].lat, gt[0].lon] : null)
  const topk = (detected && live?.prediction?.target_topk) || []
  const trueDestId = spawn?.true_dest_id

  return (
    <div className="map-wrap">
      <MapContainer center={[48.6, 31.5]} zoom={6} zoomControl={false} attributionControl={false} preferCanvas>
        <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          subdomains="abcd" maxZoom={19} />
        {borders && <GeoJSON data={borders} style={{ color: '#33415c', weight: 1, fillColor: '#0e1422', fillOpacity: 0.55 }} />}

        {/* Réseau de capteurs : masqué par défaut, affiché via le toggle */}
        {showSensors && dasLines?.map((line, i) => (
          <Polyline key={`das${i}`} positions={line}
            pathOptions={{ color: MOD.das.color, weight: 2, opacity: 0.55, dashArray: '1,5' }} />
        ))}
        {showSensors && sensors?.map((s, i) => (
          <CircleMarker key={`s${i}`} center={[s.lat, s.lon]} radius={2.5} interactive={false}
            pathOptions={{ color: MOD[s.modality]?.color || '#888', weight: 0,
              fillColor: MOD[s.modality]?.color || '#888', fillOpacity: 0.55 }} />
        ))}

        {origins.map((o) => (
          <Marker key={o.name} position={[o.lat, o.lon]} icon={originIcon}
            eventHandlers={{ click: () => onPickOrigin?.(o.name) }}>
            <Tooltip>{o.name} · launch site</Tooltip>
          </Marker>
        ))}

        {targets.map((t) => {
          const isTrue = spawn && t.dest_id === trueDestId
          return (
            <Marker key={t.dest_id} position={[t.lat, t.lon]}
              icon={targetIcon(ZONE_COLORS[t.zone_type] || '#aaa', isTrue)}
              zIndexOffset={isTrue ? 600 : 500}
              eventHandlers={{ click: () => onPickTarget?.(t.name) }}>
              <Tooltip direction="top" offset={[0, -8]}>{t.name} · {t.zone_type}</Tooltip>
            </Marker>
          )
        })}

        {/* Lignes de détection capteur -> drone */}
        {fired.map((e, i) => (
          e.drone_pos && (
            <Polyline key={`l${i}`} positions={[[e.sensor_lat, e.sensor_lon], e.drone_pos]}
              pathOptions={{ color: e.is_clutter ? '#5a6577' : (MOD[e.modality]?.color || '#fff'),
                weight: e.is_clutter ? 1 : 1.6, opacity: e.fade * (e.is_clutter ? 0.4 : 0.85),
                dashArray: e.is_clutter ? '2,4' : null }} />
          )
        ))}
        {fired.map((e, i) => (
          <CircleMarker key={`p${i}`} center={[e.sensor_lat, e.sensor_lon]} radius={3 + 3 * e.fade} interactive={false}
            pathOptions={{ color: e.is_clutter ? '#5a6577' : (MOD[e.modality]?.color || '#fff'),
              weight: 1, opacity: e.fade, fillOpacity: e.fade * 0.6 }} />
        ))}

        {spawn && (
          <>
            <FitOnSpawn spawn={spawn} />
            {remaining.length > 1 && <Polyline positions={remaining} pathOptions={{ color: '#9fb6ff', weight: 2, opacity: 0.35, dashArray: '5,7' }} />}
            {traveled.length > 1 && <Polyline positions={traveled} pathOptions={{ color: '#2b6cff', weight: 4, opacity: 0.95 }} />}

            {dronePos && topk.map((p, i) => {
              const t = targetById[p.dest_id]
              if (!t) return null
              const correct = p.dest_id === trueDestId
              return (
                <Polyline key={p.dest_id} positions={[dronePos, [t.lat, t.lon]]}
                  pathOptions={{ color: correct ? '#35d0d6' : '#ff4d5e', weight: 1.5 + 9 * p.p, opacity: 0.22 + 0.7 * p.p }}>
                  {i < 3 && <Tooltip permanent direction="center" className="vec-label" opacity={1}>{`${Math.round(p.p * 100)}% ${t.name}`}</Tooltip>}
                </Polyline>
              )
            })}
            {dronePos && <Marker position={dronePos} icon={droneIcon(detected)} zIndexOffset={1000} />}
          </>
        )}

        {/* ---- Couche d'interception : on surligne les sites capables d'intercepter ---- */}
        {(intervention?.state === 'ASSESSING' || intervention?.state === 'ENGAGED') && assets?.length > 0 && (() => {
          const capable = new Set((intervention.options || []).map((o) => o.site_id))
          const pById = Object.fromEntries((intervention.options || []).map((o) => [o.site_id, o.p_success]))
          return assets.filter((a) => capable.has(a.site_id)).map((a) => (
            <Marker key={`site${a.site_id}`} position={[a.lat, a.lon]} icon={siteIcon(true)} zIndexOffset={550}>
              <Tooltip direction="top" offset={[0, -10]}>
                {a.name} · {a.n_interceptors} intc · {Math.round((pById[a.site_id] || 0) * 100)}%
              </Tooltip>
            </Marker>
          ))
        })()}
      </MapContainer>
    </div>
  )
}
