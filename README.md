# 🏭 Transformer Autoencoder — Réduction de dimension & Visualisation (PFE)

**Sujet PFE :** *Autoencodeurs profonds pour la réduction de dimension et la visualisation de données complexes*  
**Application :** Maintenance prédictive sur signaux vibratoires multi-capteurs (NASA/IMS Bearing Dataset)

Ce projet propose un pipeline **non supervisé** basé sur :
- **Transformer Autoencoder** pour apprendre un **espace latent** (compression / réduction de dimension),
- **PCA 2D sur le latent** pour une **visualisation interprétable**,
- **Erreur de reconstruction (MAE)** + **seuil (train + option MAD)** pour repérer les comportements atypiques,
- Un **dashboard Streamlit** pour tester des fichiers et visualiser les résultats.

---

## ✨ Fonctionnalités
- Upload d’un fichier **TAB (4 colonnes)** (capteurs IMS)
- Prétraitement : **segmentation** + extraction **MAV**
- Création de fenêtres temporelles (**TIME_STEPS**)
- Détection d’anomalies : **score MAE** + seuil
- Visualisation :
  - courbe du score MAE + seuil,
  - PCA 2D du latent (si couche `latent` disponible)
- Mode debug (infos internes : ratio, MAD, etc.)

---

## 📂 Structure du dépôt (actuelle)

├── app.py
├── requirements.txt
├── README.md
├── notebooks/
├── scaler_bearing.gz
├── threshold_transformer.json
└── transformer_bearing_anomaly_detection.keras
## Rôle des fichiers :**
- `app.py` : application Streamlit (dashboard)
- `notebooks/` : entraînement / exploration (notebook)
- `transformer_bearing_anomaly_detection.keras` : modèle entraîné (Transformer AE)
- `scaler_bearing.gz` : scaler (normalisation identique à l’entraînement)
- `threshold_transformer.json` : configuration (seuil τ, TIME_STEPS, segment_size, etc.)
## 🚀 Installation
```bash
pip install -r requirements.txt
streamlit run app.py