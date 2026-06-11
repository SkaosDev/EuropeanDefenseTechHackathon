"""Sélection de la cible à suivre parmi les objets confirmés.

Règle (validée produit) : on verrouille la **première** cible observée (en mode base = une
personne, en mode drone = un drone) et on la garde tant qu'elle reste visible. Dès qu'on la
perd, on bascule sur la cible **la plus ancienne encore visible** (plus petit `first_seq`,
estampillé par le tracker à la création). Plus aucune cible visible -> `None` : l'appelant
laisse alors la caméra sur sa dernière orientation.

Seule la classe `target_class` (le `priority_class` du modèle) est éligible : les autres
classes détectées (p.ex. bird/airplane en mode drone, anti-faux-positifs) ne sont jamais suivies.
"""

from __future__ import annotations

from common.types import TrackedObject


class TargetSelector:
    """Choisit et conserve l'objet suivi par la caméra."""

    def __init__(self, target_class: str) -> None:
        self.target_class = target_class.lower()
        self.current_id: int | None = None

    def select(self, objects: list[TrackedObject]) -> TrackedObject | None:
        """Renvoie la cible courante (ou la nouvelle) parmi les objets visibles, sinon None."""
        candidates = [o for o in objects
                      if o.detection.class_name.lower() == self.target_class]
        if not candidates:
            self.current_id = None
            return None

        # Cible courante toujours visible -> on la garde (verrouillage).
        if self.current_id is not None:
            for o in candidates:
                if o.track_id == self.current_id:
                    return o

        # Sinon : la plus ancienne encore visible (acquise en premier).
        chosen = min(candidates, key=lambda o: o.first_seq)
        self.current_id = chosen.track_id
        return chosen


if __name__ == "__main__":
    # Test standalone : verrouillage, persistance, bascule, perte.
    from common.types import Detection

    def obj(tid: int, seq: int, cls: str = "person") -> TrackedObject:
        o = TrackedObject(track_id=tid, detection=Detection((0, 0, 10, 10), 0.9, 0, cls))
        o.first_seq = seq
        return o

    sel = TargetSelector("person")
    a, b, c = obj(10, 1), obj(11, 2), obj(12, 3)

    # 1) Première personne observée -> verrouillée (la plus ancienne du lot).
    assert sel.select([a, b]).track_id == 10, "doit verrouiller la 1re (seq mini)"
    # 2) Reste visible même si une autre, plus ancienne en id, apparaît -> on garde 10.
    assert sel.select([a, b, c]).track_id == 10, "doit garder la cible verrouillée"
    # 3) On perd 10 -> bascule sur la plus ancienne restante (11).
    assert sel.select([b, c]).track_id == 11, "doit basculer sur la plus ancienne visible"
    # 4) Un drone seul n'est pas une cible en mode 'person' -> None.
    assert sel.select([obj(99, 5, "drone")]) is None, "classe non cible -> None"
    # 5) Plus rien -> None.
    assert sel.select([]) is None
    print("TargetSelector OK")
