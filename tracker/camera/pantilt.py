"""Orientation de la caméra (pan/tilt) pour suivre la cible verrouillée.

Deux servos SG90 montés en pan (gauche/droite) et tilt (haut/bas) gardent la cible au
centre de l'image par asservissement proportionnel sur l'erreur angulaire mesurée dans la
frame. Deux backends, **même interface** :

  - ``servo`` (prod, Raspberry Pi) : pilote 2 servos via gpiozero (PWM logiciel). Import
    paresseux ; si gpiozero est absent ou l'init échoue, repli automatique sur ``sim``
    (même esprit best-effort que le contrôleur SCF4 dans ``capture.py``).
  - ``sim``  (dev) : ne bouge aucun moteur, ne fait que mémoriser les angles. C'est ce qui
    fait **tourner le cône caméra dans la vue 3D** sans matériel.

Repères / conventions
---------------------
* Angles **servo** : ce que le matériel accepte, bornés à ``[servo_min_deg, servo_max_deg]``
  (SG90 : 0–180). C'est l'état persistant. ``start()`` les place sur ``home_*_deg``.
* Réduction mécanique **propre à chaque moteur** : ``angle_caméra = gear_ratio * angle_servo``.
  Pour tourner la caméra de ``Δ°`` il faut donc tourner le servo de ``Δ/gear_ratio °``.
* ``state()`` expose les angles **caméra-monde** (``pan_cam``/``tilt_cam``, en degrés relatifs
  au home) consommés par la vue 3D ; le signe tient compte de ``invert_*`` (servo câblé à l'envers).
"""

from __future__ import annotations

import math

from loguru import logger


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def aim_error(bbox: tuple[int, int, int, int], frame_w: int, frame_h: int,
              hfov_deg: float) -> tuple[float, float]:
    """Erreur angulaire (degrés) du centre de la bbox vs axe optique (modèle sténopé).

    Renvoie ``(az, el)`` :
      * ``az`` > 0  -> cible à **droite** du centre,
      * ``el`` > 0  -> cible **au-dessus** du centre.
    Mêmes intrinsèques que la vue 3D : focale = (W/2)/tan(hfov/2), pixels carrés (focale
    identique en x et y), donc pas besoin du VFOV.
    """
    focal = (frame_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    az = math.degrees(math.atan2(cx - frame_w / 2.0, focal))
    el = math.degrees(math.atan2(frame_h / 2.0 - cy, focal))   # haut de l'image -> +
    return az, el


class PanTiltController:
    """Asservit deux servos (ou les simule) pour centrer la cible dans l'image."""

    def __init__(self, config: dict) -> None:
        cfg = config or {}
        self.driver = str(cfg.get("driver", "sim")).strip().lower()
        self.servo_min = float(cfg.get("servo_min_deg", 0))
        self.servo_max = float(cfg.get("servo_max_deg", 180))
        self.home_pan = float(cfg.get("home_pan_deg", 90))
        self.home_tilt = float(cfg.get("home_tilt_deg", 90))
        self.gain = float(cfg.get("gain", 0.5))
        self.deadband = float(cfg.get("deadband_deg", 1.5))
        self.max_step = float(cfg.get("max_step_deg", 8))

        pan = cfg.get("pan", {}) or {}
        tilt = cfg.get("tilt", {}) or {}
        self.pan_cfg, self.tilt_cfg = pan, tilt
        self.pan_gear = float(pan.get("gear_ratio", 1.0))
        self.tilt_gear = float(tilt.get("gear_ratio", 1.0))
        # Sens physique : -1 si le servo est câblé à l'envers (l'asservissement ET la vue 3D
        # utilisent ce même signe, donc ils restent cohérents).
        self._pan_dir = -1.0 if cfg.get("invert_pan", False) else 1.0
        self._tilt_dir = -1.0 if cfg.get("invert_tilt", False) else 1.0

        # État persistant : angles servo courants (initialisés au home).
        self.pan_servo = _clamp(self.home_pan, self.servo_min, self.servo_max)
        self.tilt_servo = _clamp(self.home_tilt, self.servo_min, self.servo_max)
        self._pan_servo_obj = None
        self._tilt_servo_obj = None

    # -- cycle de vie ------------------------------------------------------

    def start(self) -> None:
        """Initialise le backend et place la caméra en position de repos (home)."""
        if self.driver == "servo" and not self._init_servos():
            self.driver = "sim"  # repli : pas de gpiozero / init KO -> simulation
        self.pan_servo = _clamp(self.home_pan, self.servo_min, self.servo_max)
        self.tilt_servo = _clamp(self.home_tilt, self.servo_min, self.servo_max)
        self._apply()
        logger.info("Pan/tilt : backend '{}', home=({:.0f}°,{:.0f}°).",
                    self.driver, self.home_pan, self.home_tilt)

    def _init_servos(self) -> bool:
        """Crée les 2 servos gpiozero. Renvoie False (et log) si indisponible -> repli sim."""
        try:
            from gpiozero import AngularServo  # import différé : absent en dev
        except Exception as exc:  # noqa: BLE001 - gpiozero absent (machine de dev)
            logger.warning("gpiozero indisponible ({}) : pan/tilt simulé.", exc)
            return False
        try:
            self._pan_servo_obj = self._make_servo(AngularServo, self.pan_cfg)
            self._tilt_servo_obj = self._make_servo(AngularServo, self.tilt_cfg)
            return True
        except Exception as exc:  # noqa: BLE001 - pin occupée / pas de GPIO
            logger.warning("Init servos échouée ({}) : pan/tilt simulé.", exc)
            self._pan_servo_obj = self._tilt_servo_obj = None
            return False

    def _make_servo(self, AngularServo, axis_cfg: dict):
        """Instancie un AngularServo gpiozero depuis la config d'un axe (impulsions en ms)."""
        return AngularServo(
            axis_cfg["pin"],
            min_angle=self.servo_min, max_angle=self.servo_max,
            min_pulse_width=float(axis_cfg.get("min_pulse_ms", 0.5)) / 1000.0,
            max_pulse_width=float(axis_cfg.get("max_pulse_ms", 2.5)) / 1000.0,
        )

    def close(self) -> None:
        """Relâche les servos (no-op en sim)."""
        for s in (self._pan_servo_obj, self._tilt_servo_obj):
            try:
                if s is not None:
                    s.close()
            except Exception:  # noqa: BLE001 - fermeture best-effort
                pass
        self._pan_servo_obj = self._tilt_servo_obj = None

    # -- asservissement ----------------------------------------------------

    def track(self, bbox: tuple[int, int, int, int], frame_w: int, frame_h: int,
              hfov_deg: float) -> None:
        """Rapproche la caméra de la cible (un pas par frame), bornée et lissée."""
        az, el = aim_error(bbox, frame_w, frame_h, hfov_deg)
        # az > 0 (cible à droite) -> tourner à droite ; el > 0 (cible en haut) -> tourner en haut.
        self.pan_servo = self._step(self.pan_servo, az, self.pan_gear, self._pan_dir)
        self.tilt_servo = self._step(self.tilt_servo, el, self.tilt_gear, self._tilt_dir)
        self._apply()

    def _step(self, servo: float, err_cam_deg: float, gear: float, direction: float) -> float:
        """Nouvel angle servo après un pas proportionnel vers la cible (zone morte + vitesse max)."""
        if abs(err_cam_deg) < self.deadband:
            return servo
        delta_cam = self.gain * err_cam_deg
        delta_servo = direction * delta_cam / gear           # réduction propre à l'axe
        delta_servo = _clamp(delta_servo, -self.max_step, self.max_step)
        return _clamp(servo + delta_servo, self.servo_min, self.servo_max)

    def _apply(self) -> None:
        """Pousse les angles courants vers le matériel (no-op en sim)."""
        if self._pan_servo_obj is not None:
            self._pan_servo_obj.angle = self.pan_servo
        if self._tilt_servo_obj is not None:
            self._tilt_servo_obj.angle = self.tilt_servo

    # -- état pour la vue 3D ----------------------------------------------

    @property
    def pan_cam(self) -> float:
        """Cap caméra (°) relatif au home : + = vers la droite."""
        return self._pan_dir * (self.pan_servo - self.home_pan) * self.pan_gear

    @property
    def tilt_cam(self) -> float:
        """Élévation caméra (°) relative au home : + = vers le haut."""
        return self._tilt_dir * (self.tilt_servo - self.home_tilt) * self.tilt_gear

    def state(self) -> dict:
        return {
            "driver": self.driver,
            "pan_cam": round(self.pan_cam, 2),
            "tilt_cam": round(self.tilt_cam, 2),
            "pan_servo": round(self.pan_servo, 1),
            "tilt_servo": round(self.tilt_servo, 1),
        }


if __name__ == "__main__":
    # Test standalone (sans matériel) : géométrie + asservissement.
    W, H, HFOV = 640, 480, 66.0

    az, el = aim_error((310, 230, 330, 250), W, H, HFOV)
    assert abs(az) < 1 and abs(el) < 1, f"cible centrée -> erreur ~0 (az={az:.2f}, el={el:.2f})"
    az_r, _ = aim_error((600, 230, 640, 250), W, H, HFOV)
    assert az_r > 5, f"cible bord droit -> az nettement > 0 (az={az_r:.2f})"

    # gear_ratio = 2 : l'angle caméra vaut 2x l'angle servo.
    pt = PanTiltController({"driver": "sim", "home_pan_deg": 90, "home_tilt_deg": 90,
                            "gain": 1.0, "deadband_deg": 0.0, "max_step_deg": 999,
                            "pan": {"gear_ratio": 2.0}, "tilt": {"gear_ratio": 1.0}})
    pt.start()
    assert pt.pan_cam == 0.0, "au home, cap caméra = 0"
    # Cible loin à droite : le servo bouge, le cap caméra suit (x2 via le gear).
    for _ in range(40):
        pt.track((600, 230, 640, 250), W, H, HFOV)
    assert pt.pan_cam > 0, f"doit avoir tourné à droite (pan_cam={pt.pan_cam})"
    assert pt.pan_servo <= pt.servo_max, "servo borné à la butée"
    print("pan/tilt OK :", pt.state())
