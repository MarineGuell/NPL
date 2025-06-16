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
    page_title="Kero Chatbot",
    page_icon="🐸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé pour l'interface
st.markdown("""
    <style>
    /* Thème sombre global */
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    
    /* Style pour la sidebar */
    [data-testid="stSidebar"] {
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
    
    [data-testid="stSidebar"] .stRadio > div {
        color: #FFFFFF;
    }
    
    [data-testid="stSidebar"] .stCheckbox > div {
        color: #FFFFFF;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div {
        color: #FFFFFF;
    }
    
    /* Style pour les messages de chat */
    .stChatMessage {
        background-color: #2D2D2D;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Style pour les messages de l'utilisateur */
    .stChatMessage[data-testid="stChatMessage"] {
        background-color: #3D3D3D;
    }
    
    /* Style pour les messages du bot */
    .stChatMessage[data-testid="stChatMessage"] [data-testid="chatAvatarIcon"] {
        background-color: #4CAF50;
    }
    
    /* Style pour le texte */
    .stMarkdown {
        color: #FFFFFF;
    }
    
    /* Style pour les titres */
    h1, h2, h3 {
        color: #FFFFFF;
    }
    
    /* Style pour la zone de saisie */
    .stTextInput>div>div>input {
        background-color: #3D3D3D;
        color: #FFFFFF;
        border: 1px solid #4CAF50;
    }
    
    /* Style pour les boutons */
    .stButton>button {
        background-color: #4CAF50;
        color: #FFFFFF;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
    }
    
    /* Style pour les spinners */
    .stSpinner {
        color: #4CAF50;
    }
    
    /* Style pour les alertes */
    .stAlert {
        background-color: #3D3D3D;
        color: #FFFFFF;
    }
    
    /* Ajustement de la marge pour le contenu principal */
    .main .block-container {
        padding-top: 1rem;
        max-width: 100%;
    }
    
    /* Style pour l'avatar du bot */
    .bot-avatar {
        background-color: #4CAF50;
        color: #FFFFFF;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

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
    st.title("🐸 Kero Chatbot")
    
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
            ["Classification", "Résumé de texte", "Recherche Wikipedia", 
             "Nettoyage de texte", "Expressions régulières", "Transformation numérique",
             "Classification supervisée"],
            key="function_choice"
        )

        # Options spécifiques à chaque fonction
        if function == "Classification":
            model_type = st.radio(
                "Type de modèle pour la prédiction",
                ["ML Classique (Naive Bayes)", "Deep Learning (BERT)", "RNN", "Keras"],
                key="classif_model"
            )
            use_dl = model_type != "ML Classique (Naive Bayes)"
        elif function == "Résumé de texte":
            model_type = st.radio(
                "Type de modèle pour le résumé",
                ["ML Classique (TF-IDF)", "Deep Learning (BART)"],
                key="summarize_model"
            )
            use_dl = model_type == "Deep Learning (BART)"
        elif function == "Nettoyage de texte":
            st.markdown("### Options de nettoyage")
            remove_html = st.checkbox("Supprimer le HTML", value=True)
            remove_emojis = st.checkbox("Supprimer les emojis", value=True)
            remove_urls = st.checkbox("Supprimer les URLs", value=True)
            remove_emails = st.checkbox("Supprimer les emails", value=True)
            remove_numbers = st.checkbox("Supprimer les nombres", value=False)
            remove_punctuation = st.checkbox("Supprimer la ponctuation", value=False)
        elif function == "Expressions régulières":
            st.markdown("### Options de recherche")
            regex_examples = get_regex_examples()
            selected_pattern = st.selectbox(
                "Choisir un motif prédéfini",
                list(regex_examples.keys())
            )
            custom_pattern = st.text_input(
                "Ou entrer votre propre expression régulière",
                regex_examples[selected_pattern]
            )
        elif function == "Transformation numérique":
            st.markdown("### Options de transformation")
            transform_method = st.selectbox(
                "Méthode de transformation",
                ["TF-IDF", "Count Vectorizer", "BERT", "Word2Vec"]
            )
        elif function == "Classification supervisée":
            st.markdown("### Options de classification")
            classifier_type = st.selectbox(
                "Type de classificateur",
                ["SupervisedClassifier", "DeepLearningClassifier", "RNNTextClassifier", "KerasTextClassifier"]
            )

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
                    if model_type == "ML Classique (Naive Bayes)":
                        response = st.session_state.chatbot.classify_text(prompt, use_dl=False)
                    else:
                        response = st.session_state.chatbot.classify_text(prompt, use_dl=True)
                elif function == "Résumé de texte":
                    response = st.session_state.chatbot.summarize_text(prompt, use_dl=use_dl)
                elif function == "Recherche Wikipedia":
                    response = st.session_state.chatbot.search_wikipedia(prompt)
                elif function == "Nettoyage de texte":
                    response = clean_text(
                        prompt,
                        remove_html=remove_html,
                        remove_emojis=remove_emojis,
                        remove_urls=remove_urls,
                        remove_emails=remove_emails,
                        remove_numbers=remove_numbers,
                        remove_punctuation=remove_punctuation
                    )
                elif function == "Expressions régulières":
                    if validate_regex(custom_pattern):
                        matches = apply_regex(prompt, custom_pattern)
                        response = f"Correspondances trouvées : {matches}"
                    else:
                        response = "Expression régulière invalide"
                elif function == "Transformation numérique":
                    if transform_method == "TF-IDF":
                        vectors, vectorizer = text_to_tfidf([prompt])
                    elif transform_method == "Count Vectorizer":
                        vectors, vectorizer = text_to_count([prompt])
                    elif transform_method == "BERT":
                        vectors = text_to_bert_embeddings([prompt])
                        vectorizer = None
                    else:  # Word2Vec
                        vectors = text_to_word_embeddings([prompt])
                        vectorizer = None
                    
                    st.session_state.vectors = vectors
                    st.session_state.vectorizer = vectorizer
                    
                    vector_info = get_vectorization_info(vectors, vectorizer)
                    response = f"Vecteurs créés avec {transform_method}:\n{vector_info}"
                elif function == "Classification supervisée":
                    if classifier_type == "SupervisedClassifier":
                        classifier = SupervisedClassifier()
                    elif classifier_type == "DeepLearningClassifier":
                        classifier = DeepLearningClassifier()
                    elif classifier_type == "RNNTextClassifier":
                        classifier = RNNTextClassifier()
                    else:  # KerasTextClassifier
                        classifier = KerasTextClassifier()
                    
                    st.session_state.classifier = classifier
                    response = f"Classificateur {classifier_type} initialisé"
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)

if __name__ == "__main__":
    main()


