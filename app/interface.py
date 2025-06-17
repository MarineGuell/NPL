"""
Interface utilisateur du chatbot développée avec Streamlit.
Ce script gère l'interface graphique, les interactions utilisateur et l'affichage des statistiques.
"""

import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from chatbot import Chatbot
import time
from utils import (
    clean_text, extract_patterns, validate_regex, get_regex_examples,
    text_to_tfidf, text_to_count, text_to_bert_embeddings, encode_categorical,
    text_to_word_embeddings, get_vectorization_info, save_vectors, apply_regex, encode_text, transform_text
)
import numpy as np
import tempfile
import os
from models import SupervisedClassifier, DeepLearningClassifier, RNNTextClassifier, KerasTextClassifier

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Kaeru Chatbot",
    page_icon="🐸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement du CSS externe
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), "static", "style.css")
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Chargement du CSS
load_css()

# Initialisation des variables de session
# Stocke l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Stocke la fonction sélectionnée
if "selected_function" not in st.session_state:
    st.session_state.selected_function = "chat"

# Fonction pour envoyer un message au chatbot
def send_message(message, function_type="chat"):
    """
    Envoie un message au serveur et récupère la réponse.
    Args:
        message (str): Le message à envoyer
        function_type (str): Le type de fonction à utiliser (chat, wiki, summarize)
    Returns:
        dict: La réponse du serveur ou None en cas d'erreur
    """
    try:
        response = requests.post(
            f"http://localhost:8000/{function_type}",
            json={"text": message}
        )
        return response.json()
    except Exception as e:
        st.error(f"Erreur de connexion au serveur: {str(e)}")
        return None

def main():
    st.title("🐸 Kaeru Chatbot")
    
    # Initialisation des sessions
    if "chatbot" not in st.session_state:
        with st.spinner("Je nage dans ta direction, kero..."):
            st.session_state.chatbot = Chatbot()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "vectors" not in st.session_state:
        st.session_state.vectors = None
    if "vectorizer" not in st.session_state:
        st.session_state.vectorizer = None
    if "classifier" not in st.session_state:
        st.session_state.classifier = None

    # Sidebar pour les options
    with st.sidebar:
        st.markdown("### ⚙️ Options")
        st.markdown("---")
        
        # Sélection de la fonction
        function = st.radio(
            "Choisissez une fonction, kero 🐸",
            ["Classification", "Résumé de texte", "Recherche Wikipedia"],
            key="function_choice"
        )

        # Options spécifiques à chaque fonction
        if function == "Classification":
            model_type = st.radio(
                "Type de modèle pour la prédiction",
                ["Machine Learning", "Deep Learning"],
                key="classif_model"
            )
            use_dl = model_type != "ML Classique (Naive Bayes)"
        
        elif function == "Résumé de texte":
            model_type = st.radio(
                "Type de modèle pour le résumé",
                ["Machine Learning", "Deep Learning"],
                key="summarize_model"
            )
            use_dl = model_type == "Deep Learning"

        elif function == "Recherche Wikipedia":
            st.markdown("### Options de recherche")
            search_query = st.text_input("Entrez votre requête de recherche")

    # Zone principale pour le chat et les résultats
    st.markdown("""
    <div style='text-align: center'>
        <p>Chatbot réalisé dans le cadre du cours de Natural Langage Processing Enseigné par Nicolas Miotto à l'école Ynov Toulouse Campus</p>
    </div>
    """, unsafe_allow_html=True) 
    
    st.markdown("### 💬 Chat")
    
    # Affichage des messages
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🐸"):
                st.markdown(message["content"])
        else:
            with st.chat_message("user"):
                st.markdown(message["content"])
    
    # Zone de saisie
    if prompt := st.chat_input("Entrez votre message ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Traitement du message selon la fonction sélectionnée
        with st.chat_message("assistant", avatar="🐸"):
            with st.spinner("Je réfléchis, kero..."):
                if function == "Classification":
                    if model_type == "Machine Learning":
                        response = st.session_state.chatbot.classify_text(prompt, use_dl=False)
                    else:
                        response = st.session_state.chatbot.classify_text(prompt, use_dl=True)
                elif function == "Résumé de texte":
                    if model_type == "Machine Learning":
                        response = st.session_state.chatbot.classify_text(prompt, use_dl=False)
                    else:
                        response = st.session_state.chatbot.classify_text(prompt, use_dl=True)
                elif function == "Recherche Wikipedia":
                    response = st.session_state.chatbot.search_wikipedia(prompt)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)

if __name__ == "__main__":
    main()


