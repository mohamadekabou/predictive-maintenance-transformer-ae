# 🏭 Transformer Autoencoder — Réduction de dimension & Visualisation (PFE)

## 📌 Contexte
**Sujet du PFE (module Calcul Scientifique)** :  
Autoencodeurs profonds pour la réduction de dimension et la visualisation de données complexes.

**Application** :  
Maintenance prédictive sur signaux vibratoires multi-capteurs  
(NASA / IMS Bearing Dataset)

Ce projet illustre l’utilisation des **autoencodeurs profonds**, et en particulier des
**Transformer Autoencoders**, pour analyser des données industrielles complexes de grande dimension
dans un cadre non supervisé.

---

## 🎯 Objectifs
- Apprendre une **représentation latente compacte**
- Réduire la dimension des séries temporelles vibratoires
- Visualiser l’espace latent de manière interprétable
- Détecter automatiquement des comportements anormaux
- Proposer une application interactive d’analyse

---

## 🧠 Approche
Pipeline non supervisé basé sur :

- **Transformer Autoencoder**
  - Encoder–Decoder avec Self-Attention
  - Apprentissage d’un espace latent non linéaire

- **Visualisation**
  - Projection PCA 2D appliquée au latent

- **Détection d’anomalies**
  - Erreur de reconstruction (MAE)
  - Seuil d’entraînement
  - Option seuil adaptatif robuste (MAD)

- **Application Streamlit**
  - Analyse et visualisation interactive des résultats

---

## ✨ Fonctionnalités
- Upload de fichiers TAB (4 capteurs IMS)
- Prétraitement : segmentation + extraction MAV
- Fenêtrage temporel (`TIME_STEPS`)
- Calcul du score MAE
- Visualisation :
  - score MAE + seuil
  - PCA 2D du latent
- Mode debug (ratio, MAD, paramètres internes)

---

## 📂 Structure du dépôt
├── app.py
├── requirements.txt
├── README.md
├── notebooks/
├── report/
│ └── Rapport_PFE_Calcul_Scientifique_KABOURI_Mohamed.pdf
├── scaler_bearing.gz
├── threshold_transformer.json
└── transformer_bearing_anomaly_detection.keras

---

## 🚀 Installation & exécution
```bash
pip install -r requirements.txt
streamlit run app.py
📄 Rapport

Rapport du projet (Calcul Scientifique) disponible ici :

report/Rapport_PFE_Calcul_Scientifique_KABOURI_Mohamed.pdf
