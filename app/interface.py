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

# Interface principale
st.title("🤖 Chatbot Intelligent")

# Section de sélection des fonctionnalités
st.subheader("Choisissez une fonction :")
col1, col2, col3 = st.columns(3)

# Bouton pour la classification de texte
with col1:
    if st.button("🗂️ Classification catégorielle de texte", key="classify_btn", use_container_width=True):
        st.session_state.selected_function = "classify"
        st.experimental_rerun()

# Bouton pour le résumé de texte
with col2:
    if st.button("📝 Résumé de Texte", key="summarize_btn", use_container_width=True):
        st.session_state.selected_function = "summarize"
        st.experimental_rerun()

# Bouton pour la recherche Wikipedia
with col3:
    if st.button("📚 Recherche Wikipedia", key="wiki_btn", use_container_width=True):
        st.session_state.selected_function = "wiki"
        st.experimental_rerun()

# Affichage de la fonction actuellement sélectionnée
st.info(f"Fonction actuelle : {st.session_state.selected_function.upper()}")

# Champ de saisie pour les messages
user_input = st.chat_input("Écrivez votre message ici...")

# Traitement des messages
if user_input:
    # Enregistrement du message de l'utilisateur
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
    
    # Envoi au chatbot et récupération de la réponse
    response = send_message(user_input, st.session_state.selected_function)
    
    if response:
        # Enregistrement de la réponse du chatbot
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.get("text", "Désolé, je n'ai pas pu générer une réponse."),
            "category": response.get("category"),
            "confidence": response.get("confidence"),
            "timestamp": datetime.now().isoformat()
        })
        update_metrics()
        st.experimental_rerun()

# Mise en page en deux colonnes
col1, col2 = st.columns([2, 1])

# Colonne principale : Affichage des messages
with col1:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "category" in message and message["category"]:
                st.info(f"Catégorie: {message['category']} (Confiance: {message.get('confidence', 0):.2f})")

# Colonne latérale : Statistiques
with col2:
    st.title("📊 Statistiques")
    
    # Bouton de rafraîchissement des métriques
    if st.button("🔄 Rafraîchir les statistiques"):
        update_metrics()
    
    # Affichage des métriques principales
    st.metric("Total Requêtes", st.session_state.metrics.get("total_requests", 0))
    st.metric("Taux de Succès", f"{st.session_state.metrics.get('success_rate', 0)*100:.1f}%")
    st.metric("Temps de Réponse Moyen", f"{st.session_state.metrics.get('average_response_time', 0):.2f}s")
    
    # Graphique de distribution des catégories
    if "category_distribution" in st.session_state.metrics:
        categories = st.session_state.metrics["category_distribution"]
        if categories:
            df = pd.DataFrame(list(categories.items()), columns=["Catégorie", "Nombre"])
            fig = px.pie(df, values="Nombre", names="Catégorie", title="Distribution des Catégories")
            st.plotly_chart(fig, use_container_width=True)

# Pied de page
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Chatbot réalisé dans le cadre du cours de *Natural Langage Processing* Ensegné par Miotto à l'école Ynov</p>
    </div>
""", unsafe_allow_html=True) 