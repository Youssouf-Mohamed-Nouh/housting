import pandas as pd
import streamlit as st
import joblib
from datetime import datetime, date
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
st.set_page_config(initial_sidebar_state='expanded',page_title='Prédicteur la prix maison - Youssouf',page_icon='🏦',layout='wide')

# source
@st.cache_resource
def charger_model():
    try:
        model = joblib.load('best_model.pkl')
        features = joblib.load('features.pkl')
        return model,features
    except FileNotFoundError as e:
        st.error(f'Erreur: Fichier manquant - {e}')
        st.stop()
    except Exception as e:
        st.error(f'Erreur lors de chargement :{e}')
        st.stop()
        
model,features = charger_model()

# en tete
st.markdown('''
            <style>
            .main-header{
               background: linear-gradient(135deg, #91BDF2 0%, #91BDF2 100%);
               padding:2.2rem;
               border-radius:50px;
               margin-bottom:2rem;
               text-align:center;
               border-shadow: 0 20px 50px rgba(0,0,0,0.1);
                }
            </style>
            ''',unsafe_allow_html=True)
st.markdown('''
            <div class='main-header'>
            <h1>🏦 Prédicteur le prix immobilier</h1>
            <p style='font-size:20px;'>Développé par - <strong>Youssouf</strong> Assistant Intelligent</p>
            </div>
            
            ''',unsafe_allow_html=True)
# sidbar
st.markdown('''
            <style>
            .friendly-info {
                background: #e3f2fd;
                padding: 2rem;
                border-radius: 15px;
                border-left: 5px solid #2196F3;
                margin: 1.5rem 0;
            }
            .encouragement {
            background: linear-gradient(135deg, #fff3e0, #ffecb3);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            border-left: 5px solid #ff9800;
        }
            </style>
            ''',unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## 🤖 À propos de votre assistant")
    st.markdown("""
    <div class="friendly-info">
        <h4>Comment je fonctionne ?</h4>
        <p>• J'utilise un modèle d'IA entraîné sur des milliers de cas</p>
        <p>• Ma précision est d'environ 84%</p>
        <p>• Je respecte votre vie privée</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 💡 Rappel important")
    st.markdown("""
    <div class="encouragement">
        <p><strong>Gardez en tête :</strong></p>
        <p>✨ Je suis un outil d'aide, pas un agent de immobilier</p>
    </div>
    """, unsafe_allow_html=True)

# formulaire
st.markdown('''
            <h2 style='color:#343a40;text-align:center;margin-bottom:25px'> 📋 Informations du clients</h2>
            
            ''',unsafe_allow_html=True)
# --- Formulaire client ---
with st.form(key='formulaire_client'):
    def user_input_features():
        data = {}
        data['superficie'] = st.number_input("Superficie (m²)", min_value=1000, max_value=10500, value=1000)
        data['chambre'] = st.number_input("Nombre de chambres", min_value=1, max_value=5, value=4)
        data['salle_de_pain'] = st.number_input("Nombre de salles de bain", min_value=0, max_value=10, value=1)
        data['etage'] = st.number_input("Nombre d'étages", min_value=0, max_value=5, value=1)
        data['stationnement'] = st.number_input("Places de stationnement", min_value=0, max_value=10, value=1)
    
        # Variables catégorielles binaires (yes/no)
        data['route_principale_yes'] = 1 if st.checkbox("Route principale à proximité") else 0
        data['chambre_invite_yes'] = 1 if st.checkbox("Chambre d'invité") else 0
        data['sous_sol_yes'] = 1 if st.checkbox("Sous-sol") else 0
        data['chauffe_eau_yes'] = 1 if st.checkbox("Chauffe-eau") else 0
        data['climatisation_yes'] = 1 if st.checkbox("Climatisation") else 0
        data['zone_privilegie_yes'] = 1 if st.checkbox("Zone privilégiée") else 0

        # --- Meublé : utilisation de selectbox ---
        meuble = st.selectbox("Meublé", options=['furnished', 'semi-furnished', 'unfurnished'], index=0)

        # Initialisation des colonnes liées au meuble
        data['meuble_semi-furnished'] = 1 if meuble == 'semi-furnished' else 0
        data['meuble_unfurnished'] = 1 if meuble == 'unfurnished' else 0
        # Remarque : 'furnished' est le cas de référence (drop_first=True), donc pas besoin de colonne

        return pd.DataFrame([data])

    input_df = user_input_features()
    submitted = st.form_submit_button("Prédire le prix")

# Aligner les colonnes avec celles du modèle
df = pd.DataFrame(np.zeros((1, len(features))), columns=features)
for col in input_df.columns:
    if col in df.columns:
        df[col] = input_df[col].values

# --- Prédiction ---
if submitted:
    with st.spinner("Calcul en cours..."):
        try:
            prediction = model.predict(df)[0]
            st.success(f"🏷️ Prix prédit : {prediction:,.0f} (monnaie locale)")

            # Importance des variables
            importances = model.named_steps['model'].feature_importances_
            feat_importance = pd.Series(importances, index=features).sort_values(ascending=True)
            st.subheader("Importance des variables")
            fig, ax = plt.subplots()
            feat_importance.tail(10).plot(kind='barh', ax=ax)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

         
# Message de conclusion plus chaleureux
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 20px; margin-top: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h4 style="color: #495057; margin-bottom: 1rem;">🏦 Votre Assistant Intelligente</h4>
    <p style="font-size: 1em; color: #6c757d; margin-bottom: 0.5rem;">
        Créé avec passion par <strong>Youssouf</strong> pour vous accompagner dans votre parcours santé
    </p>
    <p style="font-size: 0.9em; color: #6c757d; margin-bottom: 1rem;">
        Version 2024 - Mis à jour régulièrement pour votre bien-être
    </p>
    <div style="border-top: 1px solid #dee2e6; padding-top: 1rem;">
        <p style="font-size: 0.85em; color: #6c757d; font-style: italic;">
            ⚠️ Rappel important : Cet outil d'aide à la décision complète mais ne remplace jamais 
            l'expertise de votre agent
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

















    