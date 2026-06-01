# 🏭 Predictive Maintenance — Transformer Autoencoder

> Unsupervised anomaly detection on multi-sensor vibration signals using a **Transformer Autoencoder**, with dimensionality reduction, latent-space visualization, and an interactive **Streamlit** app.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/Keras%2FTensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-completed-success)

---

## 📌 Context

Final-year project (Master 2 — *Calcul Scientifique* module): **deep autoencoders for dimensionality reduction and visualization of complex data**.

**Application:** predictive maintenance on multi-sensor vibration signals (**NASA / IMS Bearing Dataset**). The project demonstrates how deep autoencoders — and Transformer Autoencoders in particular — can analyze high-dimensional industrial data in a fully **unsupervised** setting.

## 🎯 Objectives

- Learn a compact latent representation of vibration signals
- Reduce the dimensionality of vibratory time series
- Visualize the latent space in an interpretable way
- Automatically detect abnormal behavior (early failure signs)
- Deliver an interactive analysis app

## 🧠 Approach

Unsupervised pipeline built around:

- **Transformer Autoencoder** — Encoder–Decoder with Self-Attention, learning a non-linear latent space
- **Visualization** — 2D PCA projection of the latent representation
- **Anomaly detection** — reconstruction error (MAE), training threshold, with an optional robust adaptive threshold (MAD)
- **Streamlit app** — interactive analysis and visualization of results

## ✨ Features

- Upload of TAB files (4 IMS sensors)
- Preprocessing: segmentation + MAV (Mean Absolute Value) extraction
- Temporal windowing (`TIME_STEPS`)
- MAE reconstruction-score computation
- Visualization: MAE score + threshold, and 2D PCA of the latent space
- Debug mode (ratio, MAD, internal parameters)

## 📊 Results

<!-- 👉 Ajoute ici une capture d'écran de ton app Streamlit : glisse l'image dans l'éditeur GitHub, ça génère le lien tout seul -->
<!-- ![Streamlit demo](docs/demo.png) -->

- The Transformer Autoencoder reconstructs normal behavior with low error; abnormal windows produce a clear spike in reconstruction error.
- The 2D PCA of the latent space separates normal vs. degraded states visually.
- *(Add your concrete numbers here: detection threshold, run-to-failure point detected, etc.)*

## 🗂️ Repository structure

```
.
├── app.py                                  # Streamlit application
├── requirements.txt
├── notebooks/                              # exploration & training
├── report/
│   └── Rapport_PFE_Calcul_Scientifique_KABOURI_Mohamed.pdf
├── scaler_bearing.gz                       # fitted scaler
├── threshold_transformer.json             # anomaly threshold
└── transformer_bearing_anomaly_detection.keras   # trained model
```

## 🚀 Installation & usage

```bash
git clone https://github.com/mohamadekabou/predictive-maintenance-transformer-ae.git
cd predictive-maintenance-transformer-ae
pip install -r requirements.txt
streamlit run app.py
```

## 🧰 Tech stack

`Python` · `Keras / TensorFlow` · `Transformer Autoencoder` · `Self-Attention` · `PCA` · `Streamlit` · `NumPy` · `scikit-learn`

## 📄 Report

Full project report (Calcul Scientifique) available here:
[`report/Rapport_PFE_Calcul_Scientifique_KABOURI_Mohamed.pdf`](report/Rapport_PFE_Calcul_Scientifique_KABOURI_Mohamed.pdf)

---

*Realized as part of my Master 2 final project (PFE).*
