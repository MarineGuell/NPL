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
from monitoring import Monitoring

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Chatbot Intelligent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
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
    </style>
    """, unsafe_allow_html=True)

# Initialisation des variables de session
# Stocke l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Stocke les métriques du système
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "total_requests": 0,
        "success_rate": 0,
        "average_response_time": 0
    }

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

# Fonction pour mettre à jour les métriques
def update_metrics():
    """
    Récupère et met à jour les métriques depuis le serveur.
    """
    try:
        metrics = requests.get("http://localhost:8000/metrics").json()
        st.session_state.metrics = metrics
    except:
        pass

def main():
    st.title("🤖 Chatbot Avancé")
    
    # Initialisation des sessions
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = Chatbot()
    if "monitoring" not in st.session_state:
        st.session_state.monitoring = Monitoring()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "stats" not in st.session_state:
        st.session_state.stats = {
            "total_requests": 0,
            "avg_response_time": 0,
            "success_rate": 0
        }

    # Sidebar pour les statistiques
    with st.sidebar:
        st.header("📊 Statistiques")
        st.metric("Total Requêtes", st.session_state.stats["total_requests"])
        st.metric("Temps de réponse moyen", f"{st.session_state.stats['avg_response_time']:.2f}s")
        st.metric("Taux de succès", f"{st.session_state.stats['success_rate']:.1f}%")

    # Sélection du mode de classification
    classification_mode = st.radio(
        "Mode de classification",
        ["ML Classique (Naive Bayes)", "Deep Learning (BERT)"],
        horizontal=True
    )
    use_dl = classification_mode == "Deep Learning (BERT)"

    # Sélection de la fonction
    function = st.radio(
        "Choisissez une fonction",
        ["Classification", "Résumé de texte", "Recherche Wikipedia"],
        horizontal=True
    )

    # Zone de chat
    st.subheader("💬 Chat")
    
    # Affichage des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "category" in message and message["category"]:
                st.info(f"Catégorie: {message['category']} (Confiance: {message['confidence']:.2%})")
            if "embeddings" in message and message["embeddings"]:
                st.caption("Embeddings BERT disponibles")

    # Zone de saisie
    user_input = st.chat_input("Écrivez votre message ici...")

    if user_input:
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
            summary = st.session_state.chatbot.summarize_text(user_input)
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

        # Mise à jour des statistiques
        st.session_state.stats["total_requests"] += 1
        st.session_state.stats["avg_response_time"] = st.session_state.monitoring.get_average_response_time()
        st.session_state.stats["success_rate"] = st.session_state.monitoring.get_success_rate()

        # Rafraîchir l'affichage
        st.experimental_rerun()

if __name__ == "__main__":
    main()

# Pied de page
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Chatbot réalisé dans le cadre du cours de *Natural Langage Processing* Ensegné par Miotto à l'école Ynov</p>
    </div>
""", unsafe_allow_html=True) 