# app.py — Streamlit Dashboard (Transformer Autoencoder : MAE + PCA latent)
# --------------------------------------------------------------
# Place ce fichier dans le même dossier que :
#  - transformer_bearing_anomaly_detection.keras
#  - scaler_bearing.gz
#  - threshold_transformer.json
#
# Lancement :
#   streamlit run app.py
# --------------------------------------------------------------

import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib
import plotly.graph_objects as go
from sklearn.decomposition import PCA


# ==============================================================
# 1) Custom Layer (IMPORTANT pour éviter l'erreur de désérialisation Keras)
# ==============================================================

@tf.keras.utils.register_keras_serializable(package="PFE")
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
            }
        )
        return cfg


# ==============================================================
# 2) UI config
# ==============================================================

st.set_page_config(page_title="Transformer Autoencoder (MAE + PCA latent)", layout="wide")

st.title("🏭 Dashboard — Transformer Autoencoder (MAE + PCA latent)")
st.markdown(
    """
**Sujet PFE :** Autoencodeurs profonds pour la réduction de dimension et la visualisation de données complexes.  
**Approche :** Détection d’anomalies non supervisée par **erreur de reconstruction (MAE)** + **PCA sur espace latent**.
"""
)

MODEL_PATH = "transformer_bearing_anomaly_detection.keras"
SCALER_PATH = "scaler_bearing.gz"
THRESH_PATH = "threshold_transformer.json"

# Baseline pour le seuil adaptatif (Option B) : on calcule MAD sur le début du fichier
BASELINE_FRAC = 0.20


# ==============================================================
# 3) Helpers
# ==============================================================

def create_sequences(values: np.ndarray, time_steps: int) -> np.ndarray:
    """Sliding window -> (N_seq, time_steps, d)"""
    out = []
    for i in range(len(values) - time_steps):
        out.append(values[i : (i + time_steps)])
    return np.asarray(out, dtype=np.float32)


def mav_by_segments(values_2d: np.ndarray, segment_size: int) -> np.ndarray:
    """
    values_2d: (N,4) brut
    retourne MAV segments: (m,4)
    """
    X = values_2d.astype(np.float32)
    n = (len(X) // segment_size) * segment_size
    if n <= 0:
        return np.empty((0, X.shape[1]), dtype=np.float32)

    X = X[:n]
    X = X.reshape(-1, segment_size, X.shape[1])   # (m, seg, 4)
    return np.mean(np.abs(X), axis=1)             # (m, 4)


def robust_mad(x: np.ndarray) -> float:
    """MAD robuste (median absolute deviation)."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)) + 1e-12)
    return mad


# ==============================================================
# 4) Load assets (cache)
# ==============================================================

@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"TransformerBlock": TransformerBlock},
            compile=False
        )
        scaler = joblib.load(SCALER_PATH)

        with open(THRESH_PATH, "r") as f:
            cfg = json.load(f)

        t_train = float(cfg.get("threshold"))
        time_steps = int(cfg.get("time_steps", 10))
        segment_size_train = int(cfg.get("segment_size", 512))

        # Encoder depuis la couche "latent" si elle existe
        encoder = None
        try:
            latent_layer = model.get_layer("latent")
            encoder = tf.keras.Model(inputs=model.input, outputs=latent_layer.output)
        except Exception:
            encoder = None

        return model, scaler, cfg, t_train, time_steps, segment_size_train, encoder

    except Exception as e:
        return None, None, None, None, None, None, None, str(e)


assets = load_assets()

# assets peut contenir une erreur texte en 8ème position si échec
if len(assets) == 8 and isinstance(assets[7], str):
    model, scaler, cfg, T_TRAIN, TIME_STEPS, SEGMENT_SIZE_TRAIN, encoder, err = assets
else:
    model, scaler, cfg, T_TRAIN, TIME_STEPS, SEGMENT_SIZE_TRAIN, encoder = assets
    err = None


# ==============================================================
# 5) Sidebar
# ==============================================================

st.sidebar.header("✅ Statut assets")

if err is None and model is not None and scaler is not None and cfg is not None:
    st.sidebar.success("Modèle + Scaler + Seuil chargés")
    st.sidebar.write(f"τ (seuil train) = **{T_TRAIN:.6f}**")
    st.sidebar.write(f"TIME_STEPS = **{TIME_STEPS}**")
else:
    st.sidebar.error("Erreur chargement assets")
    st.sidebar.error(err if err else "Vérifie la présence des fichiers .keras / .gz / .json")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("📥 Données d'entrée")

uploaded_file = st.sidebar.file_uploader("Charger un fichier NASA / IMS (tab, 4 colonnes)")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Paramètres")

segment_size = st.sidebar.slider(
    "Taille segment (samples) → MAV",
    min_value=512,
    max_value=8192,
    value=int(SEGMENT_SIZE_TRAIN),
    step=256
)

decision_ratio = st.sidebar.slider(
    "Seuil décision (% fenêtres > seuil)",
    min_value=0.00,
    max_value=0.20,
    value=0.05,
    step=0.01
)

use_mad = st.sidebar.checkbox("Seuil adaptatif (MAD) en plus du seuil train", value=True)

k_mad = st.sidebar.slider(
    "k (MAD) pour seuil adaptatif",
    min_value=0.0,
    max_value=10.0,
    value=6.0,
    step=0.5
)

debug = st.sidebar.checkbox("Mode debug", value=False)


# ==============================================================
# 6) Main
# ==============================================================

if uploaded_file is None:
    st.info("👈 Charge un fichier (tab, 4 colonnes) pour commencer.")
    st.stop()

# Lecture du fichier
try:
    df = pd.read_csv(uploaded_file, sep="\t", header=None)
except Exception:
    st.error("Erreur lecture fichier. Assure-toi que c'est bien un fichier TAB (4 colonnes).")
    st.stop()

# Vérification colonnes
if df.shape[1] != 4:
    st.error(f"Le modèle attend 4 colonnes (4 capteurs). Fichier détecté: {df.shape[1]} colonnes.")
    st.stop()

st.subheader("📊 Aperçu (signal brut)")
st.dataframe(df.head())

# Bouton d'analyse
if st.button("Lancer le Diagnostic IA"):
    with st.spinner("Analyse en cours..."):
        try:
            raw_values = df.values.astype(np.float32)

            # MAV par segments (réduit fortement la taille)
            feats = mav_by_segments(raw_values, segment_size=segment_size)  # (m,4)

            if debug:
                st.write(f"DEBUG — nb segments MAV = {len(feats)}")

            if len(feats) <= TIME_STEPS:
                st.error(f"Pas assez de segments MAV ({len(feats)}) pour TIME_STEPS={TIME_STEPS}.")
                st.stop()

            # Scaling (même scaler que training)
            feats_scaled = scaler.transform(feats)

            # Séquences
            X_seq = create_sequences(feats_scaled, TIME_STEPS)  # (N, T, 4)
            if debug:
                st.write(f"DEBUG — nb fenêtres = {len(X_seq)}")

            # Reconstruction
            X_pred = model.predict(X_seq, verbose=0)

            # Score MAE par fenêtre
            score = np.mean(np.abs(X_pred - X_seq), axis=(1, 2))  # (N,)
            score_min = float(np.min(score))
            score_mean = float(np.mean(score))
            score_max = float(np.max(score))

            # Seuil utilisé
            t_train = float(T_TRAIN)
            t_used = t_train

            # ---- Seuil adaptatif robuste sur la BASELINE (début fichier) ----
            if use_mad:
                n_base = max(50, int(BASELINE_FRAC * len(score)))
                base = score[:n_base]
                med = float(np.median(base))
                mad = robust_mad(base)
                t_adapt = med + float(k_mad) * mad
                t_used = max(t_train, t_adapt)
            else:
                med, mad, t_adapt = None, None, None

            # Règle décision (ratio)
            ratio = float(np.mean(score > t_used))
            alert = ratio > float(decision_ratio)

            # ==========================================================
            # ✅ OPTION B : si 0% > seuil => NOMINAL
            #    - PANNE si ratio >= decision_ratio
            #    - DÉGRADATION si 0% < ratio < decision_ratio
            #    - NOMINAL si ratio == 0
            #    + sécurité : si score_max > seuil, force PANNE
            # ==========================================================
            if ratio >= float(decision_ratio):
                status = "PANNE"
            elif ratio > 0.0:
                status = "DÉGRADATION"
            else:
                status = "NOMINAL"

            # sécurité : un pic au-dessus du seuil => PANNE
            if score_max > t_used:
                status = "PANNE"

            # UI metrics
            col1, col2 = st.columns(2)
            col1.metric("Score d'Anomalie Max (MAE)", f"{score_max:.6f}")

            if status == "PANNE":
                col2.error("🚨 PANNE DÉTECTÉE")
            elif status == "DÉGRADATION":
                col2.warning("🟠 DÉGRADATION DÉTECTÉE")
            else:
                col2.success("✅ SYSTÈME NOMINAL")

            st.caption(
                f"Seuil utilisé = {t_used:.6f} | % fenêtres > seuil = {ratio*100:.2f}% | statut = {status}"
            )

            if use_mad and debug:
                st.write(
                    f"DEBUG — t_train={t_train:.6f} | baseline_median={med:.6f} | "
                    f"baseline_MAD={mad:.6f} | t_adapt={t_adapt:.6f}"
                )

            if debug:
                st.write(f"DEBUG — score min/mean/max = {score_min:.6f} {score_mean:.6f} {score_max:.6f}")
                st.write(f"DEBUG — ratio(score>seuil) = {ratio:.6f}")
                st.write(f"DEBUG — decision_ratio = {float(decision_ratio):.6f}")

            # Plot score
            st.subheader("📈 Évolution du score d’anomalie (MAE)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=score, mode="lines", name="Score anomalie (MAE)"))
            fig.add_hline(y=t_used, line_dash="dash", annotation_text="Seuil utilisé", annotation_position="top left")

            # Optionnel: afficher aussi le seuil train si seuil adaptatif activé
            if use_mad and t_used != t_train:
                fig.add_hline(y=t_train, line_dash="dot", annotation_text="Seuil train (τ)", annotation_position="bottom left")

            fig.update_layout(
                title="Score MAE par fenêtres",
                xaxis_title="Fenêtres (séquences)",
                yaxis_title="Score MAE",
                template="plotly_white",
                height=420
            )
            st.plotly_chart(fig, use_container_width=True)

            # PCA latent
            st.markdown("---")
            st.subheader("🧬 PCA sur l’espace latent (réduction de dimension apprise)")

            if encoder is None:
                st.info("Encoder latent indisponible (couche 'latent' non trouvée). Sauvegarde le modèle avec une couche nommée 'latent'.")
            else:
                Z = encoder.predict(X_seq, verbose=0)  # (N, latent_dim)
                pca = PCA(n_components=2, random_state=42)
                Z2 = pca.fit_transform(Z)

                labels = (score > t_used).astype(int)  # 1 anomalie
                fig_pca = go.Figure()

                # Normal
                idx0 = labels == 0
                fig_pca.add_trace(go.Scatter(
                    x=Z2[idx0, 0],
                    y=Z2[idx0, 1],
                    mode="markers",
                    name="Normal",
                    marker=dict(size=6, opacity=0.6)
                ))

                # Anomalie
                idx1 = labels == 1
                fig_pca.add_trace(go.Scatter(
                    x=Z2[idx1, 0],
                    y=Z2[idx1, 1],
                    mode="markers",
                    name="Anomalie",
                    marker=dict(size=7, opacity=0.7)
                ))

                var = pca.explained_variance_ratio_
                fig_pca.update_layout(
                    title=f"PCA 2D de l’espace latent — variance expliquée: PC1={var[0]:.3f}, PC2={var[1]:.3f}",
                    xaxis_title="PC1",
                    yaxis_title="PC2",
                    template="plotly_white",
                    height=520
                )

                st.plotly_chart(fig_pca, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors de l'analyse : {e}")
else:
    st.info("Clique sur **Lancer le Diagnostic IA** pour analyser le fichier.")
