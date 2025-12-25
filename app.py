import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go 
from sklearn.decomposition import PCA  # Indispensable pour la visualisation 2D

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="NASA Predictive Maintenance", layout="wide")

st.title("🏭 Dashboard de Maintenance Prédictive (IA Transformer)")
st.markdown("""
**Sujet PFE :** Autoencodeurs profonds pour la réduction de dimension et la visualisation de données complexes.
*Modèle utilisé : Transformer Encoder (Self-Attention)*
""")

# --- 2. CHARGEMENT DU MODÈLE ET DU SCALER ---
@st.cache_resource 
def load_assets():
    try:
        # Charge le modèle .keras et le scaler .gz
        model = tf.keras.models.load_model('transformer_bearing_anomaly_detection.keras')
        scaler = joblib.load('scaler_bearing.gz')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_assets()

# Vérification du chargement
if model is not None and scaler is not None:
    st.sidebar.success("✅ Modèle IA & Scaler chargés")
else:
    st.sidebar.error("⚠️ Erreur : Fichiers manquants. Vérifiez que .keras et .gz sont dans le dossier.")

# --- 3. FONCTIONS UTILITAIRES ---
def create_sequences(values, time_steps=10):
    """Transforme les données en séquences pour le Transformer"""
    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

# --- 4. INTERFACE UTILISATEUR ---
st.sidebar.header("📥 Simulation de Données")
uploaded_file = st.sidebar.file_uploader("Charger un fichier NASA")

if uploaded_file is not None:
    # A. Lecture du fichier
    try:
        df = pd.read_csv(uploaded_file, sep='\t', header=None)
        st.write("### 📊 Aperçu des données brutes")
        st.dataframe(df.head())
    except Exception as e:
        st.error("Erreur de lecture du fichier. Assurez-vous que c'est bien un fichier NASA.")

    # B. Bouton pour lancer l'analyse
    if st.button("Lancer le Diagnostic IA"):
        with st.spinner('Analyse des vibrations en cours...'):
            try:
                # --- PRÉPARATION DES DONNÉES ---
                data_values = df.values 
                
                # Échantillonnage si le fichier est trop gros (pour aller vite)
                if len(data_values) > 5000:
                    data_values = data_values[:5000]

                # Normalisation (Réduction de dimension : Scaling)
                data_scaled = scaler.transform(data_values)
                
                # Séquençage (Time Steps)
                X_input = create_sequences(data_scaled, time_steps=10)
                
                # --- PRÉDICTION (AUTOENCODEUR) ---
                # Le modèle essaie de reconstruire l'entrée
                X_pred = model.predict(X_input)
                
                # Calcul de l'erreur de reconstruction (MAE)
                mae_loss = np.mean(np.abs(X_pred - X_input), axis=1)
                score = np.mean(mae_loss, axis=1) 
                
                # --- CALIBRAGE DU SEUIL (PFE) ---
                # Basé sur tes tests : Sain ~27, Panne ~80. Seuil fixé à 45.
                THRESHOLD = 45.0 
                
                # --- RÉSULTATS VISUELS ---
                max_score = np.max(score)
                col1, col2 = st.columns(2)
                
                col1.metric("Score d'Anomalie Max", f"{max_score:.4f}", delta_color="inverse")
                
                if max_score > THRESHOLD:
                    col2.error("🚨 ALERTE CRITIQUE : DÉFAUT DÉTECTÉ")
                else:
                    col2.success("✅ SYSTÈME SAIN")
                
                # 1. Graphique Temporel (Ligne)
                st.subheader("📈 Évolution du Score d'Anomalie")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=score, mode='lines', name='Erreur (Transformer)', line=dict(color='blue')))
                fig.add_hline(y=THRESHOLD, line_dash="dash", line_color="red", annotation_text="Seuil Alerte (45.0)")
                
                fig.update_layout(
                    title="Détection d'Anomalies en Temps Réel",
                    xaxis_title="Temps (Séquences)",
                    yaxis_title="Erreur de Reconstruction",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)

                # --- 2. VISUALISATION PCA (Correctement indenté ici) ---
                st.markdown("---")
                st.subheader("🧬 Visualisation de la Réduction de Dimension (PCA)")
                st.write("Projection des données complexes (4 capteurs) sur un plan 2D pour visualiser la séparation Sain/Défaut.")

                # Préparation PCA
                data_pca = data_scaled  # On utilise les données normalisées existantes
                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(data_pca)

                # Création Graphique PCA
                fig_pca = go.Figure()

                # Couleur dynamique selon le diagnostic
                color_pca = 'red' if max_score > THRESHOLD else 'green'
                label_pca = 'Défaut Détecté' if max_score > THRESHOLD else 'Données Saines'

                # Points du fichier actuel
                fig_pca.add_trace(go.Scatter(
                    x=principal_components[:, 0],
                    y=principal_components[:, 1],
                    mode='markers',
                    marker=dict(size=8, color=color_pca, opacity=0.7),
                    name=f"Fichier Actuel ({label_pca})"
                ))

                # Nuage de référence (Simulation zone saine pour comparaison visuelle)
                random_healthy_x = np.random.normal(0, 0.5, 100)
                random_healthy_y = np.random.normal(0, 0.5, 100)

                fig_pca.add_trace(go.Scatter(
                    x=random_healthy_x,
                    y=random_healthy_y,
                    mode='markers',
                    marker=dict(size=5, color='green', opacity=0.1),
                    name="Zone de Référence (Sain)"
                ))

                fig_pca.update_layout(
                    title="Projection 2D de l'Espace Latent (PCA)",
                    xaxis_title="Composante Principale 1",
                    yaxis_title="Composante Principale 2",
                    template="plotly_white"
                )

                st.plotly_chart(fig_pca)

            except Exception as e:
                st.error(f"Erreur lors de l'analyse : {e}")

else:
    st.info("👈 Veuillez charger un fichier NASA (ex: 2004.02.12...) pour commencer.")