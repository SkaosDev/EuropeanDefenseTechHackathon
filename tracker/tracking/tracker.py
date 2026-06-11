"""Suivi temporel à partir des identifiants ByteTrack.

Les IDs viennent du détecteur (`model.track` de YOLO). Cette couche ajoute seulement
l'historique de trajectoire, la durée de vie (apparition / disparition) et conserve l'objet
entre les frames (ce qui permet de lisser la distance dans `main.py`).
"""

from __future__ import annotations

from collections import deque

from common.types import Detection, TrackedObject


class ObjectTracker:
    """Gère le cycle de vie et la trajectoire des objets suivis."""

    def __init__(self, config: dict) -> None:
        self.max_lost = config.get("max_lost_frames", 30)
        self.min_hits = config.get("min_hits", 3)
        self.traj_len = config.get("trajectory_length", 50)
        self.tracks: dict[int, TrackedObject] = {}
        self._seq = 0  # compteur croissant : estampille l'ordre de création des tracks

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        """Met à jour l'état avec les détections du frame et renvoie les objets confirmés."""
        seen: set[int] = set()
        for det in detections:
            tid = det.track_id
            if tid is None:
                continue
            obj = self.tracks.get(tid)
            if obj is None:
                self._seq += 1
                obj = TrackedObject(track_id=tid, detection=det,
                                    trajectory=deque(maxlen=self.traj_len),
                                    first_seq=self._seq)
                self.tracks[tid] = obj
            obj.detection = det
            obj.hits += 1
            obj.lost_frames = 0
            obj.trajectory.append(det.center)
            seen.add(tid)

        # Vieillissement / suppression des objets non revus.
        for tid in list(self.tracks):
            if tid not in seen:
                self.tracks[tid].lost_frames += 1
                if self.tracks[tid].lost_frames > self.max_lost:
                    del self.tracks[tid]

        # On n'expose que les objets visibles et assez vus (anti-faux positifs).
        return [o for o in self.tracks.values()
                if o.lost_frames == 0 and o.hits >= self.min_hits]


if __name__ == "__main__":
    # Test standalone : deux frames, même ID ByteTrack -> trajectoire accumulée.
    trk = ObjectTracker({"min_hits": 1, "trajectory_length": 10})
    print("f1:", [(o.track_id, o.detection.center)
                  for o in trk.update([Detection((100, 100, 160, 200), 0.9, 0, "drone", 1)])])
    print("f2:", [(o.track_id, o.detection.center)
                  for o in trk.update([Detection((108, 104, 168, 204), 0.9, 0, "drone", 1)])])
    print("trajectoire:", list(trk.tracks[1].trajectory))
