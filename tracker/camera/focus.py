"""LensController : pilotage du focus de la Kurokesu C3 18X (lentille SCF4).

Reprend l'algorithme d'autofocus prouvé de ``test_kurokesu/main.py`` (recherche
discrète en 2 passes — grossière puis fine —, interpolation parabolique sub-pas,
arrêt anticipé, compensation de backlash), mais **paramétré par la config** et
**découplé de toute fenêtre OpenCV** : les frames viennent d'un callable injecté
(le cache de ``CameraCapture``) et l'autofocus tourne dans un **thread** pour ne
jamais figer le pipeline de détection ni le flux MJPEG.

Politique de focus : **pas de refocus automatique**. L'autofocus ne tourne qu'au
démarrage (si ``autofocus_on_start``) et sur demande explicite (bouton du
dashboard). Une fois net, la position reste figée jusqu'au prochain déclenchement.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import cv2
import numpy as np
from loguru import logger


class LensController:
    """Focus motorisé SCF4 (axe B) : autofocus one-shot + réglage manuel.

    Paramètres lus dans ``focus_cfg`` (section ``camera.focus`` du YAML) :
    focus_min/max, coarse_step, fine_step, fine_window, speed, backlash_steps,
    approach_dir, early_stop_ratio, roi_ratio, manual_step, settle_sec,
    home_on_start.
    """

    def __init__(self, dev, focus_cfg: dict, get_latest_frame: Callable[[], "np.ndarray | None"],
                 zoom_cfg: dict | None = None):
        self.dev = dev                       # instance scf4.SCF4 déjà connectée
        self._get_frame = get_latest_frame   # renvoie la dernière frame BGR (ou None)
        cfg = focus_cfg or {}
        zcfg = zoom_cfg or {}

        # Zoom (axe A) : on dézoome au maximum au démarrage. Le homing de A amène le
        # moteur à sa butée de référence (PI), qui est ici le côté TÉLÉOBJECTIF ; on
        # se déplace donc ensuite vers `wide_position` (extrémité grand-angle = champ
        # max). Le homing fixe l'origine A à 32000 ; le grand-angle est à l'opposé du
        # PI (par défaut côté 0 ; passer à 64000 si le zoom part quand même au télé).
        self.dezoom_on_start = bool(zcfg.get("dezoom_on_start", True))
        self.zoom_axis = str(zcfg.get("axis", "A"))
        self.zoom_wide_position = int(zcfg.get("wide_position", 0))
        self.zoom_min = int(zcfg.get("min", 0))
        self.zoom_max = int(zcfg.get("max", 64000))
        self.zoom_step = int(zcfg.get("manual_step", 2000))
        self._zoom_position = self.zoom_wide_position

        self.focus_min = int(cfg.get("focus_min", 2000))
        self.focus_max = int(cfg.get("focus_max", 15000))
        self.coarse_step = int(cfg.get("coarse_step", 300))
        self.fine_step = int(cfg.get("fine_step", 30))
        self.fine_window = int(cfg.get("fine_window", 400))
        self.speed = int(cfg.get("speed", 600))
        self.backlash_steps = int(cfg.get("backlash_steps", 20))
        self.approach_dir = int(cfg.get("approach_dir", 1))
        self.early_stop_ratio = float(cfg.get("early_stop_ratio", 0.30))
        self.roi_ratio = float(cfg.get("roi_ratio", 0.40))
        self.manual_step = int(cfg.get("manual_step", 50))
        self.settle_sec = float(cfg.get("settle_sec", 0.15))
        self.home_on_start = bool(cfg.get("home_on_start", True))

        self._position = self._clamp((self.focus_min + self.focus_max) // 2)
        self._busy = False                   # un autofocus est en cours
        self._stop = threading.Event()       # demande d'arrêt (fermeture)
        self._thread: threading.Thread | None = None
        self._serial_lock = threading.Lock()  # une seule opération moteur à la fois
        self._lock = threading.Lock()         # protège l'état publié
        self._state = {"enabled": True, "status": "idle", "position": None, "sharpness": 0.0}

    # ------------------------------------------------------------------ cycle de vie

    def start(self) -> None:
        """Initialise le contrôleur et fait le homing focus. Peut lever (le
        backend caméra attrape l'erreur pour basculer en focus désactivé)."""
        self.dev.init_controller()
        # Zoom AVANT le focus : changer le zoom modifie la mise au point, donc on
        # dézoome d'abord, puis on home/autofocus le focus sur le champ final.
        if self.dezoom_on_start:
            self._drive_to_wide_stop()
        if self.home_on_start:
            logger.info("Focus C3 : homing de l'axe B...")
            self.dev.home_axis("B")
        self.dev.set_speed("B", self.speed)
        self._position = self._clamp((self.focus_min + self.focus_max) // 2)
        # Position zoom réelle lue sur le contrôleur (sert de base au réglage manuel).
        try:
            self._zoom_position = self.dev.position(self.zoom_axis)
        except Exception:  # noqa: BLE001
            self._zoom_position = self.zoom_wide_position
        self._set_state(status="idle", position=self._position)
        logger.info("Focus C3 prêt (plage {}..{}, zoom={}).",
                    self.focus_min, self.focus_max, self._zoom_position)

    def _drive_to_wide_stop(self) -> None:
        """Pousse le zoom jusqu'à sa butée mécanique grand-angle (dézoom max).

        On ne se fie PAS au homing (non répétable sur ce SCF4) ni à une position
        absolue : on commande un déplacement RELATIF plus long que toute la course,
        dans le sens du grand-angle (déduit de `wide_position`). Le moteur, en
        boucle ouverte, cale sur la butée physique — référence répétable du « plus
        grand champ possible ». On y fixe ensuite l'origine (`G92` = `wide_position`)
        pour que les boutons de zoom manuels repartent d'un zéro connu.
        """
        axis = self.zoom_axis
        midpoint = (self.zoom_min + self.zoom_max) / 2
        # Sens physique vers le grand-angle : côté de `wide_position` (configurable).
        sign = 1 if self.zoom_wide_position >= midpoint else -1
        overtravel = sign * (int((self.zoom_max - self.zoom_min) * 1.2) + 1000)
        logger.info("Zoom : dézoom max — course relative {:+d} (axe {}) jusqu'à la butée...",
                    overtravel, axis)
        self.dev.set_speed(axis, self.speed)
        self.dev.send("G91")                                   # mode relatif
        self.dev.send(f"G0 {axis}{overtravel:+d}")             # course longue -> butée mécanique
        try:
            self.dev.wait_stop(axis, timeout=90)
        except TimeoutError:
            logger.warning("Zoom : butée non confirmée (timeout) — supposé grand-angle.")
        self.dev.send("G90")                                   # retour en absolu
        self.dev.send(f"G92 {axis}{self.zoom_wide_position}")  # la butée devient wide_position
        self._zoom_position = self.zoom_wide_position
        logger.info("Zoom : butée grand-angle atteinte (origine fixée à {}).",
                    self.zoom_wide_position)

    def close(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        try:
            self.dev.close()
        except Exception:  # noqa: BLE001 - fermeture best-effort
            pass

    # ------------------------------------------------------------------ API publique

    def autofocus(self) -> bool:
        """Lance un cycle d'autofocus one-shot dans un thread. Non bloquant.

        Renvoie False si un autofocus est déjà en cours (clic ignoré)."""
        with self._lock:
            if self._busy:
                logger.info("Autofocus déjà en cours, demande ignorée.")
                return False
            self._busy = True
            self._state.update(status="focusing")
        self._thread = threading.Thread(target=self._af_worker, daemon=True)
        self._thread.start()
        return True

    def adjust_focus(self, delta: int) -> bool:
        """Décale le focus de `delta` pas (signé). Ignoré si autofocus en cours."""
        if self._busy:
            logger.info("Réglage manuel ignoré : autofocus en cours.")
            return False
        if not self._serial_lock.acquire(blocking=False):
            return False
        try:
            target = self._clamp(self._position + int(delta))
            self.dev.move_to_backlash("B", target, self.backlash_steps, self.approach_dir, wait=True)
            self._position = target
            frame = self._get_frame()
            s = self._sharpness(frame) if frame is not None else 0.0
            self._set_state(status="done", position=target, sharpness=s)
            logger.info("Focus manuel -> {} (net={:.0f})", target, s)
            return True
        except Exception as exc:  # noqa: BLE001 - perte série / timeout moteur
            logger.warning("Réglage focus manuel échoué : {}", exc)
            self._set_state(status="error")
            return False
        finally:
            self._serial_lock.release()

    def adjust_zoom(self, delta: int) -> bool:
        """Décale le zoom (axe A) de `delta` pas. delta<0 = dézoome (grand-angle).
        Ignoré si un autofocus est en cours."""
        if self._busy:
            return False
        if not self._serial_lock.acquire(blocking=False):
            return False
        try:
            target = max(self.zoom_min, min(self.zoom_max, self._zoom_position + int(delta)))
            self.dev.move_abs(self.zoom_axis, target, wait=True)
            self._zoom_position = target
            logger.info("Zoom manuel -> {}", target)
            return True
        except Exception as exc:  # noqa: BLE001 - perte série / timeout moteur
            logger.warning("Réglage zoom manuel échoué : {}", exc)
            return False
        finally:
            self._serial_lock.release()

    def dezoom_max(self) -> bool:
        """Va directement à la position grand-angle max (sans re-homing). Bouton
        de secours si le dézoom au démarrage n'a pas suffi."""
        if self._busy:
            return False
        if not self._serial_lock.acquire(blocking=False):
            return False
        try:
            self.dev.move_abs(self.zoom_axis, self.zoom_wide_position, wait=False)
            try:
                self.dev.wait_stop(self.zoom_axis, timeout=60)
            except TimeoutError:
                logger.warning("Zoom : fin de course non confirmée (timeout) — supposé grand-angle.")
            self._zoom_position = self.zoom_wide_position
            logger.info("Zoom : grand-angle max (pos={}).", self.zoom_wide_position)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dézoom max échoué : {}", exc)
            return False
        finally:
            self._serial_lock.release()

    def state(self) -> dict:
        with self._lock:
            return dict(self._state)

    # ------------------------------------------------------------------ autofocus (interne)

    def _af_worker(self) -> None:
        try:
            with self._serial_lock:
                self._run_autofocus()
        finally:
            self._busy = False

    def _run_autofocus(self) -> None:
        """Recherche 2 passes (grossière + fine) + placement final avec backlash."""
        t0 = time.time()
        try:
            self.dev.set_speed("B", self.speed)

            coarse = self._scan_range(self.focus_min, self.focus_max,
                                      self.coarse_step, self.early_stop_ratio)
            if not coarse or self._stop.is_set():
                self._set_state(status="idle")
                return
            peak_c, _ = max(coarse, key=lambda t: t[1])

            lo = self._clamp(peak_c - self.fine_window)
            hi = self._clamp(peak_c + self.fine_window)
            fine = self._scan_range(lo, hi, self.fine_step)
            best = self._parabolic_peak(fine) if fine else peak_c

            final = self._move_and_measure(best, backlash=True)
            self._position = best
            self._set_state(status="done", position=best, sharpness=final)
            logger.info("Autofocus C3 : pos={} net={:.0f} en {:.1f}s",
                        best, final, time.time() - t0)
        except Exception as exc:  # noqa: BLE001 - perte série / timeout moteur
            logger.warning("Autofocus C3 échoué : {}", exc)
            self._set_state(status="error")

    def _scan_range(self, lo, hi, step, early_stop_ratio=None):
        """Balaye lo->hi par pas `step`. Arrêt anticipé si la netteté retombe
        sous (1 - ratio) du pic vu (le pic est passé)."""
        table = []
        peak = 0.0
        for pos in range(int(lo), int(hi) + 1, int(step)):
            if self._stop.is_set():
                break
            s = self._move_and_measure(pos)
            table.append((pos, s))
            self._set_state(status="focusing", position=pos, sharpness=s)
            if early_stop_ratio is not None:
                peak = max(peak, s)
                if peak > 0 and len(table) >= 3 and s < peak * (1 - early_stop_ratio):
                    break
        return table

    def _move_and_measure(self, pos, settle=None, backlash=False):
        """Va à `pos`, attend l'arrêt moteur, renvoie le MAX de netteté sur une
        courte fenêtre (les frames de transition sont plus floues -> le max =
        image stabilisée, robuste à la latence du pipeline caméra)."""
        settle = self.settle_sec if settle is None else settle
        pos = self._clamp(pos)
        if backlash:
            self.dev.move_to_backlash("B", pos, self.backlash_steps, self.approach_dir, wait=True)
        else:
            self.dev.move_abs("B", pos, wait=True)

        best = 0.0
        t0 = time.time()
        while time.time() - t0 < settle:
            frame = self._get_frame()
            if frame is not None:
                s = self._sharpness(frame)
                if s > best:
                    best = s
            time.sleep(0.02)
        return best

    def _parabolic_peak(self, table):
        """Position du score max + interpolation parabolique sub-pas."""
        i = max(range(len(table)), key=lambda k: table[k][1])
        pos = table[i][0]
        if 0 < i < len(table) - 1:
            (x0, y0), (x1, y1), (x2, y2) = table[i - 1], table[i], table[i + 1]
            denom = y0 - 2 * y1 + y2
            if denom != 0:
                pos = x1 + 0.5 * (y0 - y2) / denom * (x1 - x0)
        return self._clamp(pos)

    def _sharpness(self, frame):
        """Variance du Laplacien sur la ROI centrale (roi_ratio) après flou anti-bruit."""
        h, w = frame.shape[:2]
        rw, rh = int(w * self.roi_ratio), int(h * self.roi_ratio)
        x0, y0 = (w - rw) // 2, (h - rh) // 2
        roi = frame[y0:y0 + rh, x0:x0 + rw]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return float(cv2.Laplacian(blurred, cv2.CV_64F).var())

    # ------------------------------------------------------------------ utilitaires

    def _clamp(self, v):
        return max(self.focus_min, min(self.focus_max, int(v)))

    def _set_state(self, **kw):
        with self._lock:
            self._state.update(kw)
