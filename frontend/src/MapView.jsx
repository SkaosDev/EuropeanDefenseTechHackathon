import { useEffect, useMemo, useState } from 'react'
import {
  MapContainer, TileLayer, GeoJSON, CircleMarker, Polyline, Marker, Tooltip, useMap, useMapEvents,
} from 'react-leaflet'
import L from 'leaflet'
import { ZONE_COLORS, MOD } from './api'

const SENSOR_ZOOM = 8      // au-delà : on affiche les capteurs ponctuels
const DAS_ZOOM = 7         // au-delà : on affiche les câbles fibre

const droneIcon = (detected) => L.divIcon({
  className: '', html: `<div class="drone-icon ${detected ? '' : 'undetected'}"></div>`,
  iconSize: [16, 16], iconAnchor: [8, 8],
})
const originIcon = L.divIcon({
  className: '', html: '<div style="width:11px;height:11px;background:#ffd24d;transform:rotate(45deg);border:1px solid #7a5a00"></div>',
  iconSize: [11, 11], iconAnchor: [6, 6],
})
const sensorIcons = {}
function sensorIcon(mod) {
  if (!sensorIcons[mod]) {
    sensorIcons[mod] = L.divIcon({
      className: '', html: `<div class="sensor-icon">${MOD[mod]?.glyph || '•'}</div>`,
      iconSize: [16, 16], iconAnchor: [8, 8],
    })
  }
  return sensorIcons[mod]
}

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

// Capteurs + câbles DAS, affichés seulement au zoom (sinon illisible), filtrés à la vue.
function SensorsLayer({ sensors, dasLines, onZoom }) {
  const map = useMap()
  const [z, setZ] = useState(map.getZoom())
  const [, setTick] = useState(0)
  useMapEvents({ zoomend: () => { setZ(map.getZoom()); onZoom?.(map.getZoom()) },
                 moveend: () => setTick((t) => t + 1) })
  const showSensors = z >= SENSOR_ZOOM
  const showDas = z >= DAS_ZOOM
  const inView = useMemo(() => {
    if (!showSensors) return []
    const b = map.getBounds()
    return sensors.filter((s) => b.contains([s.lat, s.lon])).slice(0, 1200)
  }, [showSensors, sensors, z, map])
  return (
    <>
      {showDas && dasLines.map((line, i) => (
        <Polyline key={`das${i}`} positions={line}
          pathOptions={{ color: MOD.das.color, weight: 2, opacity: 0.5, dashArray: '1,5' }} />
      ))}
      {inView.map((s, i) => (
        <Marker key={i} position={[s.lat, s.lon]} icon={sensorIcon(s.modality)}
          interactive={false} opacity={0.85} />
      ))}
    </>
  )
}

export default function MapView({ targets, origins, sensors, dasLines, borders, spawn, live, fired, detected, onZoom }) {
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
  const future = (detected && live?.prediction?.pred_future) || []
  const trueDestId = spawn?.true_dest_id

  return (
    <div className="map-wrap">
      <MapContainer center={[48.6, 31.5]} zoom={6} zoomControl={false} preferCanvas>
        <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution="&copy; OpenStreetMap &copy; CARTO" subdomains="abcd" maxZoom={19} />
        {borders && <GeoJSON data={borders} style={{ color: '#33415c', weight: 1, fillColor: '#0e1422', fillOpacity: 0.55 }} />}

        {sensors?.length > 0 && <SensorsLayer sensors={sensors} dasLines={dasLines || []} onZoom={onZoom} />}

        {origins.map((o) => (
          <Marker key={o.name} position={[o.lat, o.lon]} icon={originIcon}>
            <Tooltip>{o.name} · launch site</Tooltip>
          </Marker>
        ))}

        {targets.map((t) => {
          const isTrue = spawn && t.dest_id === trueDestId
          return (
            <CircleMarker key={t.dest_id} center={[t.lat, t.lon]} radius={isTrue ? 7 : 4}
              pathOptions={{ color: isTrue ? '#35d0d6' : (ZONE_COLORS[t.zone_type] || '#aaa'),
                weight: isTrue ? 2 : 1, fillColor: ZONE_COLORS[t.zone_type] || '#aaa',
                fillOpacity: isTrue ? 0.9 : 0.5 }}>
              <Tooltip>{t.name} · {t.zone_type}</Tooltip>
            </CircleMarker>
          )
        })}

        {/* Lignes de détection capteur -> drone (montre la détection multi-capteurs) */}
        {fired.map((e, i) => (
          e.drone_pos && (
            <Polyline key={`l${i}`} positions={[[e.sensor_lat, e.sensor_lon], e.drone_pos]}
              pathOptions={{ color: e.is_clutter ? '#5a6577' : (MOD[e.modality]?.color || '#fff'),
                weight: e.is_clutter ? 1 : 1.6, opacity: e.fade * (e.is_clutter ? 0.4 : 0.85),
                dashArray: e.is_clutter ? '2,4' : null }} />
          )
        ))}
        {fired.map((e, i) => (
          <CircleMarker key={`p${i}`} center={[e.sensor_lat, e.sensor_lon]} radius={3 + 3 * e.fade}
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
            {future.length > 1 && <Polyline positions={future} pathOptions={{ color: '#ffb84d', weight: 2, opacity: 0.7, dashArray: '2,6' }} />}
            {dronePos && <Marker position={dronePos} icon={droneIcon(detected)} zIndexOffset={1000} />}
          </>
        )}
      </MapContainer>
    </div>
  )
}
