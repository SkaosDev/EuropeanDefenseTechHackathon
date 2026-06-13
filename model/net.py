"""
net.py — LSTM multi-têtes pour la prédiction temps réel.

Entrée : séquence d'événements de détection [B, L, 17] + masque [B, L] (1 = événement réel,
0 = padding). La représentation concatène trois vues complémentaires de la fenêtre :
  * le DERNIER état caché valide (récence : les événements les plus récents = les plus
    proches de la cible, déterminants en fin de course) ;
  * un POOLING MOYEN masqué (géométrie globale de l'essaim de capteurs) ;
  * un POOLING MAX masqué (détections saillantes).
Empiriquement, "dernier état" excelle en observation complète et le pooling aide tôt :
les combiner donne le meilleur des deux.

  features -> Linear+ReLU -> LSTM -> [last ⊕ mean ⊕ max] -> 3 têtes
    (a) target  : 65 cibles    -> CrossEntropy (priorité, label smoothing)
    (c) class   : 4 classes    -> CrossEntropy pondérée
    (b) future  : 12 (lat,lon) -> MSE masquée

Le modèle ne voit JAMAIS la vérité-terrain ; le clutter est mélangé volontairement.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DroneNet(nn.Module):
    def __init__(self, n_feat=17, proj=64, hidden=128, n_layers=2, bidir=False,
                 n_targets=65, n_classes=4, n_future=12, dropout=0.3, lstm_dropout=0.2):
        super().__init__()
        self.n_future = n_future
        self.input_proj = nn.Sequential(nn.Linear(n_feat, proj), nn.ReLU())
        self.lstm = nn.LSTM(
            proj, hidden, num_layers=n_layers, batch_first=True, bidirectional=bidir,
            dropout=lstm_dropout if n_layers > 1 else 0.0,
        )
        dirs = 2 if bidir else 1
        d = hidden * dirs
        rep = d * 3                                   # concat(last, mean, max)
        self.drop = nn.Dropout(dropout)
        self.head_target = nn.Sequential(
            nn.Linear(rep, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_targets),
        )
        self.head_class = nn.Linear(rep, n_classes)
        self.head_future = nn.Linear(rep, n_future * 2)

    def forward(self, x, mask):
        # x: [B, L, F]   mask: [B, L] (1 = réel)
        h = self.input_proj(x)
        out, _ = self.lstm(h)                         # [B, L, D]
        m = mask.unsqueeze(-1)                        # [B, L, 1]
        lengths = mask.sum(dim=1).long()
        idx = (lengths - 1).clamp(min=0)
        b = torch.arange(x.size(0), device=x.device)
        last = out[b, idx]                            # dernier état valide
        denom = m.sum(dim=1).clamp(min=1.0)
        mean_pool = (out * m).sum(dim=1) / denom
        max_pool = torch.nan_to_num(
            out.masked_fill(m == 0, float("-inf")).max(dim=1).values, neginf=0.0)
        rep = self.drop(torch.cat([last, mean_pool, max_pool], dim=1))
        target_logits = self.head_target(rep)
        class_logits = self.head_class(rep)
        future = self.head_future(rep).view(-1, self.n_future, 2)
        return target_logits, class_logits, future
