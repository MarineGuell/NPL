import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Chatbot Intelligent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .stTextInput>div>div>input {
        font-size: 18px;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.bot {
        background-color: #475063;
    }
    .chat-message .content {
        display: flex;
        flex-direction: row;
        align-items: center;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex: 1;
    }
    .function-button {
        margin: 5px;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        background-color: #4CAF50;
        color: white;
        cursor: pointer;
    }
    .function-button.active {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialisation de la session
if "messages" not in st.session_state:
    st.session_state.messages = []

if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "total_requests": 0,
        "success_rate": 0,
        "average_response_time": 0
    }

if "selected_function" not in st.session_state:
    st.session_state.selected_function = "chat"

# Fonction pour envoyer un message au chatbot
def send_message(message, function_type="chat"):
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
    try:
        metrics = requests.get("http://localhost:8000/metrics").json()
        st.session_state.metrics = metrics
    except:
        pass

# Interface principale
st.title("🤖 Chatbot Intelligent")

# Sélection de la fonction
st.subheader("Choisissez une fonction :")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💬 Chat Simple", key="chat_btn", use_container_width=True):
        st.session_state.selected_function = "chat"
        st.experimental_rerun()

with col2:
    if st.button("📚 Recherche Wikipedia", key="wiki_btn", use_container_width=True):
        st.session_state.selected_function = "wiki"
        st.experimental_rerun()

with col3:
    if st.button("📝 Résumé de Texte", key="summarize_btn", use_container_width=True):
        st.session_state.selected_function = "summarize"
        st.experimental_rerun()

# Affichage de la fonction sélectionnée
st.info(f"Fonction actuelle : {st.session_state.selected_function.upper()}")

# Champ de saisie (doit être à la racine)
user_input = st.chat_input("Écrivez votre message ici...")

if user_input:
    # Ajout du message de l'utilisateur
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
    
    # Envoi au chatbot avec la fonction sélectionnée
    response = send_message(user_input, st.session_state.selected_function)
    
    if response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.get("text", "Désolé, je n'ai pas pu générer une réponse."),
            "category": response.get("category"),
            "confidence": response.get("confidence"),
            "timestamp": datetime.now().isoformat()
        })
        update_metrics()
        st.experimental_rerun()

# Colonnes pour affichage
col1, col2 = st.columns([2, 1])

with col1:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "category" in message and message["category"]:
                st.info(f"Catégorie: {message['category']} (Confiance: {message.get('confidence', 0):.2f})")

# Colonne latérale pour les statistiques
with col2:
    st.title("📊 Statistiques")
    
    # Bouton pour rafraîchir les métriques
    if st.button("🔄 Rafraîchir les statistiques"):
        update_metrics()
    
    # Affichage des métriques
    st.metric("Total Requêtes", st.session_state.metrics.get("total_requests", 0))
    st.metric("Taux de Succès", f"{st.session_state.metrics.get('success_rate', 0)*100:.1f}%")
    st.metric("Temps de Réponse Moyen", f"{st.session_state.metrics.get('average_response_time', 0):.2f}s")
    
    # Distribution des catégories
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
        <p>Chatbot Intelligent - Développé avec ❤️</p>
    </div>
""", unsafe_allow_html=True) 