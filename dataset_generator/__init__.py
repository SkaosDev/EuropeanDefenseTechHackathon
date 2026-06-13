"""
dataset_generator — Simulateur de données d'entraînement counter-UAS.

Génère des trajectoires synthétiques de munitions rôdeuses (Shahed-136, Gerbera,
FPV fibre, Lancet) depuis des sites de lancement réels vers des cibles ukrainiennes
réelles, puis simule un réseau de capteurs (optique / acoustique / vibration / fibre
DAS / RF) qui transforment la trajectoire en un FLUX D'ÉVÉNEMENTS bruités.

Ce flux d'événements est l'ENTRÉE du modèle ; la vraie trajectoire reste une
vérité-terrain (labels : cible, classe, trajectoire future).

Usage défensif / éducatif uniquement (hackathon défense).
"""

__version__ = "0.1.0"
