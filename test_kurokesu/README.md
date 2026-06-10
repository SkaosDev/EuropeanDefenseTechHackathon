# Kurokesu C3 18X — vidéo nette + focus

Script unique et simple pour afficher le flux de la caméra Kurokesu C3 18X
**net**, en pilotant la mise au point. Le zoom est supposé **fixe** (dézoomé au
grand-angle) ; on ne contrôle que le **focus** (axe B du contrôleur SCF4).

Tout est **mono-thread et synchrone** : la netteté (variance du Laplacien sur la
zone centrale) est mesurée en lisant une frame *fraîche* juste après l'arrêt du
moteur. Simple et fiable.

Deux câbles côté matériel : micro-USB pour le contrôleur moteur (port série
SCF4) et le câble du retour vidéo (UVC).

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python main.py --list-cams     # repérer l'index de la caméra
python main.py --cam 1         # vidéo + autofocus au démarrage
python main.py --cam 1 --no-home
python main.py --cam 1 --scan  # diagnostic : courbe netteté vs position
```

Touches dans la fenêtre vidéo :

| Touche | Action |
|---|---|
| `a` / espace | (re)faire l'autofocus |
| `+` / `-` (ou `.` / `,`) | focus manuel |
| `h` | refaire le homing puis autofocus |
| `q` / `Esc` | quitter |

## Réglages

En tête de `main.py` (pas de fichier de config) :
- `FOCUS_MIN / FOCUS_MAX` (2000 / 15000) : plage de recherche du focus, mesurée
  sur ce montage. **À revérifier avec `--scan`** si le montage/distance change.
- `COARSE_STEP` (300) / `FINE_STEP` (30) : pas des passes grossière/fine. Le pic
  de focus est étroit (~±150 pas), donc `COARSE_STEP` doit rester < ~300.
- `MANUAL_STEP` (50) : pas du focus manuel.
- `BACKLASH_STEPS` (20) / `APPROACH_DIR` (+1) : compensation du jeu mécanique. La
  position finale est toujours abordée par le même sens (le bas), ce qui rend la
  mise au point répétable malgré le backlash de la lentille varifocale.
- `EARLY_STOP_RATIO` (0.30) : arrêt anticipé de la passe grossière dès que la
  netteté retombe sous 70 % du pic vu (le pic est passé) → balayage plus rapide.

## Diagnostic

`python main.py --cam N --scan --min 2000 --max 15000 --step 100` balaie la plage
et imprime la courbe netteté-vs-position + le pic. Vise une cible texturée et
éclairée. Un pic « net » donne une netteté de plusieurs dizaines ; si tout reste
< 20, élargis `--min/--max`.

## Notes matériel (issues du débogage)

- Le **homing SCF4 n'est pas répétable** (finit vers ~62000-64500, pas 32000) →
  la plage de recherche est volontairement large pour absorber la dérive.
- Le **pic de focus est étroit** → la passe fine est indispensable.
- Si seules la caméra Mac et l'iPhone apparaissent dans `--list-cams`, c'est en
  général un câble USB-C charge-only : utiliser un vrai câble de données.

## Archive

L'ancienne architecture (autofocus continu threadé, `FocusController`,
`AutofocusEngine`, `SharpnessEstimator`, config YAML, autodétection SCF4/SCE2)
est conservée dans `_archive/` — jugée trop complexe et remplacée par ce script
unique. Le driver moteur `scf4.py` reste partagé.
