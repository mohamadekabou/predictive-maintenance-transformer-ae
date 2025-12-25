# 🏭 Maintenance Prédictive - NASA Bearing Dataset (PFE)

**Sujet :** Autoencodeurs profonds pour la réduction de dimension et la visualisation de données complexes.

## 📌 Description
Ce projet utilise le Deep Learning (Transformer Autoencoder) pour prédire les pannes de roulements industriels. Il inclut une réduction de dimension (ACP/PCA) pour visualiser la séparation entre l'état sain et défaillant.

## 📂 Contenu
* `app.py` : Dashboard interactif Streamlit.
* `notebooks/` : Code d'entraînement complet (Google Colab).
* `models/` : Modèles IA entraînés (.keras) et Scalers.

## 🚀 Installation
```bash
pip install -r requirements.txt
streamlit run app.py