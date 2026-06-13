#!/usr/bin/env python3
"""
run.py — lanceur multiplateforme (Windows / macOS / Linux) du système AEGIS counter-UAS.

Sans argument : menu interactif. Avec argument : commande directe.

  python run.py                 # menu
  python run.py setup           # crée .venv + installe deps Python & frontend (yarn)
  python run.py regen [N] [S]   # régénère le dataset (N drones, graine S ; défaut 10000 1)
  python run.py train [EPOCHS]  # entraîne le modèle (défaut 60)
  python run.py eval            # accuracy par fraction d'observation
  python run.py backend         # API FastAPI sur :8000
  python run.py frontend        # interface Vite sur :5173
  python run.py demo            # backend + frontend ensemble (Ctrl+C pour arrêter)
"""
import os
import platform
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
IS_WIN = platform.system() == "Windows"


def venv_python():
    p = os.path.join(REPO, ".venv", "Scripts" if IS_WIN else "bin", "python.exe" if IS_WIN else "python")
    return p if os.path.exists(p) else sys.executable


def have_venv():
    return os.path.exists(os.path.join(REPO, ".venv", "Scripts" if IS_WIN else "bin"))


def yarn_cmd():
    return shutil.which("yarn") or ("yarn.cmd" if IS_WIN else "yarn")


def run(cmd, cwd=REPO, **kw):
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, cwd=cwd, **kw)


def need_venv():
    if not have_venv():
        print("⚠  .venv introuvable. Lance d'abord :  python run.py setup")
        sys.exit(1)


# --------------------------------------------------------------------------- #
def cmd_setup():
    print("== Setup ==")
    if not have_venv():
        run([sys.executable, "-m", "venv", ".venv"])
    py = venv_python()
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    run([py, "-m", "pip", "install", "-r", "requirements-ml.txt"])
    yarn = yarn_cmd()
    if shutil.which(yarn) or os.path.exists(os.path.join(REPO, "frontend")):
        run([yarn, "install"], cwd=os.path.join(REPO, "frontend"))
    else:
        print("⚠  yarn introuvable — installe Node.js puis 'corepack enable' (ou brew install yarn).")
    print("\n✅ Setup terminé. Ensuite :  python run.py regen  →  python run.py train  →  python run.py demo")


def cmd_regen(args):
    need_venv()
    n = args[0] if len(args) > 0 else "10000"
    seed = args[1] if len(args) > 1 else "1"
    run([venv_python(), "-m", "dataset_generator.main", "--n-drones", n, "--seed", seed,
         "--out", "dataset_generator/out", "--no-viz"])


def cmd_train(args):
    need_venv()
    epochs = args[0] if len(args) > 0 else "60"
    run([venv_python(), "-m", "model.train", "--epochs", epochs])


def cmd_eval():
    need_venv()
    run([venv_python(), "-m", "model.eval_fractions"])


def cmd_backend():
    need_venv()
    run([venv_python(), "-m", "uvicorn", "backend.main:app", "--port", "8000"])


def cmd_frontend():
    run([yarn_cmd(), "dev"], cwd=os.path.join(REPO, "frontend"))


def cmd_demo():
    need_venv()
    print("== Démo : backend :8000 + frontend :5173 ==  (Ctrl+C pour arrêter)\n")
    backend = subprocess.Popen([venv_python(), "-m", "uvicorn", "backend.main:app", "--port", "8000"], cwd=REPO)
    frontend = subprocess.Popen([yarn_cmd(), "dev"], cwd=os.path.join(REPO, "frontend"))
    print("\n→ Ouvre http://localhost:5173 dans ton navigateur.\n")
    try:
        backend.wait()
    except KeyboardInterrupt:
        print("\nArrêt…")
    finally:
        for p in (frontend, backend):
            try:
                p.terminate()
            except Exception:
                pass


MENU = [
    ("setup", "Installer l'environnement (.venv + deps + yarn install)", cmd_setup),
    ("regen", "Régénérer le dataset (10000 drones)", lambda: cmd_regen([])),
    ("train", "Entraîner le modèle (60 epochs)", lambda: cmd_train([])),
    ("eval", "Évaluer (accuracy par fraction d'observation)", cmd_eval),
    ("backend", "Lancer le backend (API :8000)", cmd_backend),
    ("frontend", "Lancer le frontend (Vite :5173)", cmd_frontend),
    ("demo", "Tout lancer (backend + frontend)", cmd_demo),
]


def menu():
    print("\n=== AEGIS — Counter-UAS · lanceur ===")
    for i, (key, desc, _) in enumerate(MENU, 1):
        print(f"  {i}. {key:9s} — {desc}")
    print("  0. quitter")
    try:
        choice = input("\nChoix : ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if choice in ("0", "", "q"):
        return
    if choice.isdigit() and 1 <= int(choice) <= len(MENU):
        MENU[int(choice) - 1][2]()
    else:
        print("Choix invalide.")


def main():
    if len(sys.argv) < 2:
        menu()
        return
    cmd, args = sys.argv[1], sys.argv[2:]
    dispatch = {
        "setup": lambda: cmd_setup(), "regen": lambda: cmd_regen(args),
        "train": lambda: cmd_train(args), "eval": lambda: cmd_eval(),
        "backend": lambda: cmd_backend(), "frontend": lambda: cmd_frontend(),
        "demo": lambda: cmd_demo(), "all": lambda: cmd_demo(),
    }
    if cmd in dispatch:
        dispatch[cmd]()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
