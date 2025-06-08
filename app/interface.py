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

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Kero Chatbot",
    page_icon="🐸",
    layout="wide"
)

# Style CSS personnalisé pour l'interface
# Définit l'apparence des messages, boutons et autres éléments
st.markdown("""
    <style>
    /* Style pour le champ de saisie */
    .stTextInput>div>div>input {
        font-size: 18px;
    }
    /* Style pour les messages de chat */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    /* Style pour les messages de l'utilisateur */
    .chat-message.user {
        background-color: #2b313e;
    }
    /* Style pour les messages du bot */
    .chat-message.bot {
        background-color: #475063;
    }
    /* Style pour le contenu des messages */
    .chat-message .content {
        display: flex;
        flex-direction: row;
        align-items: center;
    }
    /* Style pour les avatars */
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 1rem;
    }
    /* Style pour le texte des messages */
    .chat-message .message {
        flex: 1;
    }
    /* Style pour les boutons de fonction */
    .function-button {
        margin: 5px;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        background-color: #4CAF50;
        color: white;
        cursor: pointer;
    }
    /* Style pour les boutons actifs */
    .function-button.active {
        background-color: #45a049;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .stChatMessage[data-testid="stChatMessage"] {
        background-color: #f0f2f6;
    }
    .stChatMessage[data-testid="stChatMessage"] [data-testid="chatAvatarIcon"] {
        background-color: #4CAF50;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .bot-message {
        background-color: #f5f5f5;
    }
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1rem 0;
        height: 50px;
    }
    .loading-text {
        color: #4CAF50;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    @keyframes jump {
        0% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
        100% { transform: translateY(0); }
    }
    .jumping-frog {
        font-size: 2em;
        animation: jump 1s infinite;
        display: inline-block;
    }
    .loading-message {
        display: flex;
        align-items: center;
        gap: 10px;
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
        st.session_state.chatbot = Chatbot()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    # Sélection de la fonction
    function = st.radio(
        "Choisissez une fonction, kero 🐸",
        ["Classification", "Résumé de texte", "Recherche Wikipedia"],
        horizontal=True
    )

    # Sélection du mode selon la fonction
    if function == "Classification":
        model_type = st.radio(
            "Choisisissez votre type de modèle pour réaliser la prédiction, kero 🐸",
            ["ML Classique (Naive Bayes)", "Deep Learning (BERT)"],
            horizontal=True
        )
        use_dl = model_type == "Deep Learning (BERT)"
    elif function == "Résumé de texte":
        model_type = st.radio(
            "Choisisissez votre type de modèle pour le résumé, kero 🐸",
            ["ML Classique (TF-IDF)", "Deep Learning (BART)"],
            horizontal=True
        )
        use_dl = model_type == "Deep Learning (BART)"

    # Zone de chat
    st.subheader("💬🐸Chat, kero")
    
    # Affichage des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🐸" if message["role"] == "assistant" else "👤"):
            # Affichage du contenu avec support LaTeX
            st.markdown(message["content"], unsafe_allow_html=True)
            if "category" in message and message["category"]:
                st.info(f"Catégorie: {message['category']} (Confiance: {message['confidence']:.2%})")
            if "embeddings" in message and message["embeddings"]:
                st.caption("Embeddings BERT disponibles")

    # Animation de chargement
    if st.session_state.is_processing:
        st.markdown("""
        <div class="loading-container">
            <div class="loading-message">
                <div class="jumping-frog">🐸</div>
                <div class="loading-text">Kero réfléchit...</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Zone de saisie
    user_input = st.chat_input("Écrivez votre message ici...")

    if user_input:
        # Démarrage du traitement
        st.session_state.is_processing = True
        st.experimental_rerun()

        # Ajout du message de l'utilisateur
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Traitement selon la fonction sélectionnée
        if function == "Classification":
            response = st.session_state.chatbot.generate_response(user_input, use_dl=use_dl)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["text"],
                "category": response["category"],
                "confidence": response["confidence"],
                "embeddings": response["embeddings"]
            })
        elif function == "Résumé de texte":
            summary = st.session_state.chatbot.summarize_text(user_input, use_dl=use_dl)
            st.session_state.messages.append({
                "role": "assistant",
                "content": summary
            })
        elif function == "Recherche Wikipedia":
            wiki_response = st.session_state.chatbot.search_wikipedia(user_input)
            st.session_state.messages.append({
                "role": "assistant",
                "content": wiki_response
            })

        # Fin du traitement
        st.session_state.is_processing = False
        st.experimental_rerun()

if __name__ == "__main__":
    main()

# Pied de page
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Chatbot réalisé dans le cadre du cours de Natural Langage Processing Enseigné par Nicolas Miotto à l'école Ynov Toulouse Campus</p>
    </div>
""", unsafe_allow_html=True) 