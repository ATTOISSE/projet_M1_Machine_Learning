import streamlit as st
import joblib
import pandas as pd

best_knn = joblib.load('best_knn_model.pkl')

st.title("Prédiction de la Satisfaction Client")

st.write("Veuillez entrer les caractéristiques du client :")
input_data = {}
for col in best_knn.feature_names_in_:
    input_data[col] = st.number_input(f"{col}", min_value=0.0, step=0.01)

input_df = pd.DataFrame([input_data])

if st.button("Prédire la Satisfaction"):
    prediction = best_knn.predict(input_df)
    st.write("Prédiction : Satisfait" if prediction[0] == 'satisfied' else "Insatisfait")

def preprocess_data(df):
     """
    Prétraitement d'un DataFrame pour la modélisation.

    Cette fonction effectue les étapes suivantes :
    1. Gestion des valeurs aberrantes :
        - Identification des valeurs aberrantes à l'aide de l'intervalle interquartile.
        - Remplacement des valeurs aberrantes par les bornes de l'intervalle.
    2. Transformation des variables numériques :
        - Imputation des valeurs manquantes par la médiane.
        - Standardisation des variables pour une moyenne nulle et une variance unitaire.
    3. Transformation des variables catégorielles :
        - Encodage one-hot pour représenter les catégories sous forme de variables binaires.
    4. Création d'un nouveau DataFrame avec les données prétraitées.

    Args:
        df (pd.DataFrame): DataFrame à prétraiter.

    Returns:
        pd.DataFrame: DataFrame prétraité.
    """