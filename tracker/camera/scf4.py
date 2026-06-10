"""Driver série pour le contrôleur de lentille Kurokesu SCF4 (caméra C3 18X).

Le SCF4 pilote les moteurs pas-à-pas de la lentille via des commandes G-code
envoyées sur un port série USB (CDC, 115200 bauds) :
  - axe A = zoom
  - axe B = focus
  - axe C = iris / filtre IR (non utilisé ici)

Protocole : https://wiki.kurokesu.com/books/scf4
SDK officiel : https://github.com/Kurokesu/SCF4-SDK

NOTE : copie vendorée de ``test_kurokesu/scf4.py`` (source de vérité). Gardée
autonome (ne dépend que de ``pyserial``) pour que ``tracker/`` reste déployable
seul. Si tu corriges le driver, reporte le changement dans les deux copies.
"""

import sys
import time

import serial
from serial.tools import list_ports

# Index des champs renvoyés par la commande de statut "!1"
# (position A, B, C, photo-interrupteur A, B, C, en-mouvement A, B, C)
A_POS, B_POS, C_POS = 0, 1, 2
A_PI, B_PI, C_PI = 3, 4, 5
A_MOVE, B_MOVE, C_MOVE = 6, 7, 8

_PI_IDX = {"A": A_PI, "B": B_PI, "C": C_PI}
_MOVE_IDX = {"A": A_MOVE, "B": B_MOVE, "C": C_MOVE}
_POS_IDX = {"A": A_POS, "B": B_POS, "C": C_POS}

# Le SCF4 est un STM32F103 en USB CDC (VID/PID ST Microelectronics)
_USB_IDS = {(0x0483, 0x5740)}
_NAME_HINTS = ("scf4", "kurokesu", "stm32", "usbmodem")


def find_port():
    """Cherche le port série du SCF4. Lève RuntimeError s'il est introuvable."""
    ports = list(list_ports.comports())
    for p in ports:
        if (p.vid, p.pid) in _USB_IDS:
            return p.device
    for p in ports:
        text = " ".join(filter(None, (p.device, p.description, p.manufacturer, p.product))).lower()
        if any(h in text for h in _NAME_HINTS) and "bluetooth" not in text:
            return p.device
    available = "\n".join(f"  {p.device}  ({p.description})" for p in ports) or "  (aucun)"
    raise RuntimeError(
        "Contrôleur SCF4 introuvable. La caméra est-elle branchée en USB ?\n"
        f"Ports série détectés :\n{available}\n"
        "Spécifie le port manuellement avec --port /dev/cu.usbmodemXXXX"
    )


class SCF4:
    """Connexion au contrôleur SCF4 : init, homing, mouvements, statut."""

    def __init__(self, port="auto", baudrate=115200, timeout=5, verbose=False, ser=None):
        self.verbose = verbose
        if ser is not None:
            # réutilise un port déjà ouvert (ex. après autodétection du contrôleur)
            self.ser = ser
            port = getattr(ser, "port", port)
        else:
            if port in (None, "auto"):
                port = find_port()
            self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        self.port = port

    # ------------------------------------------------------------------ bas niveau

    def send(self, cmd):
        """Envoie une commande et retourne la réponse (une ligne)."""
        self.ser.write((cmd + "\n").encode("utf8"))
        resp = self.ser.readline().decode("utf8", "replace").strip()
        if self.verbose:
            print(f"  > {cmd}\n  < {resp}")
        return resp

    def status(self):
        """Statut '!1' : [posA, posB, posC, piA, piB, piC, moveA, moveB, moveC]."""
        raw = self.send("!1")
        try:
            return [int(v.strip()) for v in raw.split(",")]
        except ValueError:
            raise IOError(f"Réponse de statut invalide : {raw!r}")

    def position(self, axis):
        return self.status()[_POS_IDX[axis]]

    def is_moving(self, axis):
        return self.status()[_MOVE_IDX[axis]] == 1

    # ------------------------------------------------------------------ initialisation

    def init_controller(self):
        """Séquence d'init reprise du démo autofocus officiel X18 de Kurokesu."""
        version = self.send("$S")
        self.send("$B2")                                  # reset + init du driver moteur
        self.send("M243 C6")                              # micro-stepping
        self.send("M230")                                 # mode mouvement normal
        self.send("G91")                                  # coordonnées relatives (pour le homing)
        self.send("M238")                                 # allume les LEDs des photo-interrupteurs
        self.send("M234 A190 B190 C190 D90")              # courant moteurs
        self.send("M235 A120 B120 C120")                  # courant au repos
        self.send("M240 A600 B600 C600")                  # vitesse moteurs
        self.send("M232 A400 B400 C400 E700 F700 G700")   # seuils de détection PI
        self.send("M7")                                   # filtre IR : position VIS
        return version

    # ------------------------------------------------------------------ attentes

    def wait_stop(self, axis, timeout=30.0, on_poll=None, settle_polls=2):
        """Attend la fin du mouvement. Exige `settle_polls` statuts 'arrêté'
        consécutifs : le SCF4 répond à G0 AVANT de démarrer le moteur, donc un
        seul statut 'arrêté' peut être lu à tort juste après l'envoi."""
        deadline = time.time() + timeout
        stopped = 0
        st = None
        while time.time() < deadline:
            st = self.status()
            if on_poll:
                on_poll(st)
            if st[_MOVE_IDX[axis]] != 1:
                stopped += 1
                if stopped >= settle_polls:
                    return st
            else:
                stopped = 0
            time.sleep(0.01)
        raise TimeoutError(f"L'axe {axis} bouge toujours après {timeout}s")

    def _wait_pi_change(self, axis, initial, timeout=30.0):
        """Attend le basculement du photo-interrupteur de l'axe (utilisé par le homing)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            st = self.status()
            if st[_PI_IDX[axis]] != initial:
                return st
            time.sleep(0.01)
        raise TimeoutError(f"Photo-interrupteur de l'axe {axis} inchangé après {timeout}s")

    # ------------------------------------------------------------------ mouvements

    def move_abs(self, axis, pos, wait=False, on_poll=None):
        """Va à une position absolue (suppose le mode G90 actif, cf. home())."""
        self.send(f"G0 {axis}{int(pos)}")
        if wait:
            self.wait_stop(axis, on_poll=on_poll)

    def move_multi(self, positions, wait_axis=None, on_poll=None):
        """Bouge plusieurs axes dans une seule commande G0 (mouvement combiné fluide).

        positions : dict, ex. {"A": 30000, "B": 35000}. C'est la technique du démo
        parfocal officiel : zoom et focus avancent ensemble pour rester nets.
        """
        parts = " ".join(f"{ax}{int(p)}" for ax, p in positions.items())
        self.send(f"G0 {parts}")
        if wait_axis:
            self.wait_stop(wait_axis, on_poll=on_poll)

    def move_to_backlash(self, axis, pos, backlash=20, approach_dir=1, wait=True, on_poll=None):
        """Atteint `pos` toujours par le même sens pour absorber le jeu mécanique.

        Les lentilles varifocales ont du backlash : selon le sens d'arrivée, le
        moteur s'arrête à quelques pas près de la consigne. On pré-positionne donc
        `backlash` pas au-delà de la cible dans le sens OPPOSÉ à `approach_dir`,
        puis on ramène sur `pos` — la prise de jeu se fait toujours du même côté,
        ce qui rend la position répétable.
        """
        pos = int(pos)
        pre = pos - int(approach_dir) * abs(int(backlash))
        if pre != pos:
            self.move_abs(axis, pre, wait=wait)
        self.move_abs(axis, pos, wait=wait, on_poll=on_poll)

    def set_speed(self, axis, value):
        self.send(f"M240 {axis}{int(value)}")

    def stop(self):
        self.send("M0")

    # ------------------------------------------------------------------ homing

    def home_axis(self, axis):
        """Homing d'un axe sur son photo-interrupteur, puis origine posée à 32000.

        Séquence reprise du démo officiel : approche en mode forcé jusqu'au
        basculement du PI, recul, ré-approche lente, puis G92 <axis>32000.
        """
        st = self.status()
        pi0 = st[_PI_IDX[axis]]

        self.send("G91")
        self.send(f"M231 {axis}")                    # mode forcé : tourne jusqu'au changement du PI
        self.send(f"G0 {axis}{'+100' if pi0 == 0 else '-100'}")
        self._wait_pi_change(axis, pi0)

        self.send(f"M230 {axis}")                    # retour en mode normal
        self.send(f"G0 {axis}-200")                  # recule un peu
        self.wait_stop(axis)

        self.send("G91")
        self.send(f"M231 {axis}")                    # ré-approche pour une référence précise
        self.send(f"G0 {axis}+100")
        self._wait_pi_change(axis, pi0)

        self.send(f"G92 {axis}32000")                # cette position devient 32000
        self.send(f"M230 {axis}")
        self.send("G90")                             # repasse en coordonnées absolues

    def home(self):
        """Homing zoom (A) puis focus (B), et passage en mode absolu."""
        self.home_axis("A")
        self.home_axis("B")
        self.send("G90")

    def close(self):
        try:
            self.send("M239")                        # éteint les LEDs PI
        except Exception:
            pass
        self.ser.close()


if __name__ == "__main__":
    # Petit test de connexion : python scf4.py [port]
    port = sys.argv[1] if len(sys.argv) > 1 else "auto"
    ctrl = SCF4(port=port, verbose=True)
    print(f"Connecté sur {ctrl.port}")
    print("Version :", ctrl.init_controller())
    print("Statut  :", ctrl.status())
    ctrl.close()
