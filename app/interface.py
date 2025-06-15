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
        background-color: #e3f2fd;
        color: #000000;
    }
    /* Style pour les messages du bot */
    .chat-message.bot {
        background-color: #f5f5f5;
        color: #000000;
    }
    /* Style pour le texte des messages */
    .chat-message .message {
        flex: 1;
        color: #000000 !important;
    }
    /* Style pour les avatars */
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 1rem;
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
        color: #000000 !important;
    }
    .bot-message {
        background-color: #f5f5f5;
        color: #000000 !important;
    }
    /* Style pour le texte dans les messages Streamlit */
    .stChatMessage [data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }
    .stChatMessage [data-testid="stMarkdownContainer"] {
        color: #000000 !important;
    }
    /* Style pour les messages d'erreur et d'info */
    .stAlert {
        color: #000000 !important;
    }
    .stAlert [data-testid="stMarkdownContainer"] {
        color: #000000 !important;
    }
    /* Style pour les captions */
    .stCaption {
        color: #000000 !important;
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
    .loading-message img {
        width: 50px;
        height: 50px;
        object-fit: contain;
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
        with st.spinner("Chargement des modèles..."):
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

    # Création de deux colonnes
    col1, col2 = st.columns([1, 3])

    # Colonne de gauche pour les options
    with col1:
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
                "Ou entrer une expression régulière personnalisée",
                regex_examples[selected_pattern]
            )
            if not validate_regex(custom_pattern):
                st.error("Expression régulière invalide !")
        elif function == "Transformation numérique":
            st.markdown("### Options de transformation")
            transform_type = st.selectbox(
                "Type de transformation",
                ["TF-IDF", "Bag of Words", "BERT Embeddings", "Word Embeddings", "Encodage catégoriel"]
            )
            max_features = st.number_input("Nombre maximum de features", min_value=100, max_value=10000, value=5000)
            if transform_type in ["TF-IDF", "Bag of Words"]:
                st.markdown("### Options de sauvegarde")
                save_format = st.selectbox("Format de sauvegarde", ["numpy", "csv", "json"])
                if st.button("Sauvegarder les vecteurs"):
                    if st.session_state.vectors is not None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{save_format}") as tmp:
                            save_vectors(st.session_state.vectors, tmp.name, save_format)
                            st.success(f"Vecteurs sauvegardés dans {tmp.name}")
        elif function == "Classification supervisée":
            st.markdown("### Options de classification")
            classifier_type = st.selectbox(
                "Type de classifieur",
                ["ML Classique", "Deep Learning", "RNN", "Keras"]
            )
            
            if classifier_type == "ML Classique":
                model_type = st.selectbox(
                    "Modèle",
                    ["Régression logistique", "SVM", "Random Forest", "Gradient Boosting", "Réseau de neurones"]
                )
                model_mapping = {
                    "Régression logistique": "logistic",
                    "SVM": "svm",
                    "Random Forest": "random_forest",
                    "Gradient Boosting": "gradient_boosting",
                    "Réseau de neurones": "neural_network"
                }
                st.session_state.classifier = SupervisedClassifier(model_type=model_mapping[model_type])
            elif classifier_type == "Deep Learning":
                model_name = st.selectbox(
                    "Modèle BERT",
                    ["bert-base-uncased", "bert-base-multilingual-cased"]
                )
                num_labels = st.number_input("Nombre de classes", min_value=2, value=2)
                st.session_state.classifier = DeepLearningClassifier(model_name=model_name, num_labels=num_labels)
            elif classifier_type == "RNN":
                st.session_state.classifier = RNNTextClassifier()
            else:
                st.session_state.classifier = KerasTextClassifier()
            
            # Options d'entraînement
            st.markdown("### Options d'entraînement")
            train_size = st.slider("Taille de l'ensemble d'entraînement (%)", 50, 90, 80)
            if classifier_type == "Deep Learning":
                epochs = st.number_input("Nombre d'époques", min_value=1, max_value=10, value=3)
                batch_size = st.number_input("Taille du batch", min_value=8, max_value=64, value=16)
                learning_rate = st.number_input("Taux d'apprentissage", min_value=1e-6, max_value=1e-3, value=2e-5, format="%.6f")

    # Colonne de droite pour le chat
    with col2:
        st.subheader("💬🐸Chat, kero")
        
        # Affichage des messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🐸" if message["role"] == "assistant" else "👤"):
                st.markdown(message["content"], unsafe_allow_html=True)
                if "category" in message and message["category"]:
                    st.info(f"Catégorie: {message['category']} (Confiance: {message['confidence']:.2%})")
                if "embeddings" in message and message["embeddings"]:
                    st.caption("Embeddings BERT disponibles")
                if "vector_info" in message:
                    st.json(message["vector_info"])
                if "training_info" in message:
                    st.json(message["training_info"])

        # Animation de chargement
        if st.session_state.is_processing:
            st.markdown("""
            <div class="loading-container">
                <div class="loading-message">
                    <img src="app/img/frog.gif" alt="Loading..." style="width: 50px; height: 50px;">
                    <div class="loading-text">Kero réfléchit...</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Zone de saisie
        user_input = st.text_input("Écrivez votre message ici...", key="chat_input")
        send_button = st.button("Envoyer", key="send_button")

        if send_button and user_input:
            try:
                # Ajout du message de l'utilisateur
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Démarrage du traitement
                st.session_state.is_processing = True
                
                # Traitement selon la fonction sélectionnée
                if function == "Classification":
                    if use_dl:
                        if model_type == "Deep Learning (BERT)":
                            response = st.session_state.chatbot.generate_response(user_input, use_dl=True)
                        elif model_type == "RNN":
                            response = st.session_state.chatbot.generate_response(user_input, use_dl=True, model_type="rnn")
                        else:  # Keras
                            response = st.session_state.chatbot.generate_response(user_input, use_dl=True, model_type="keras")
                    else:
                        response = st.session_state.chatbot.generate_response(user_input, use_dl=False)
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
                elif function == "Nettoyage de texte":
                    cleaned_text = clean_text(
                        user_input,
                        remove_html=remove_html,
                        remove_emojis=remove_emojis,
                        remove_urls=remove_urls,
                        remove_emails=remove_emails,
                        remove_numbers=remove_numbers,
                        remove_punctuation=remove_punctuation
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Texte nettoyé :\n\n{cleaned_text}"
                    })
                elif function == "Expressions régulières":
                    if validate_regex(custom_pattern):
                        matches = extract_patterns(user_input, custom_pattern)
                        if matches:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"Motifs trouvés :\n\n" + "\n".join(f"- {match}" for match in matches)
                            })
                        else:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": "Aucun motif correspondant trouvé dans le texte."
                            })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "L'expression régulière n'est pas valide. Veuillez la corriger."
                        })
                elif function == "Transformation numérique":
                    texts = [line.strip() for line in user_input.split('\n') if line.strip()]
                    if model_type in ["TF-IDF", "Bag of Words"]:
                        vectors, vectorizer = text_to_tfidf(texts, max_features)
                        st.session_state.vectors = vectors
                        st.session_state.vectorizer = vectorizer
                    elif model_type == "BERT Embeddings":
                        vectors = text_to_bert_embeddings(texts)
                        st.session_state.vectors = vectors
                    elif model_type == "Word Embeddings":
                        vectors = text_to_word_embeddings(texts)
                        st.session_state.vectors = vectors
                    elif model_type == "Encodage catégoriel":
                        vectors, encoder = encode_categorical(texts)
                        st.session_state.vectors = vectors
                        st.session_state.vectorizer = encoder

                    vector_info = get_vectorization_info(st.session_state.vectors, st.session_state.vectorizer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Transformation numérique effectuée avec succès !",
                        "vector_info": vector_info
                    })
                elif function == "Classification supervisée":
                    # Format attendu : texte,label (une ligne par exemple)
                    lines = [line.strip() for line in user_input.split('\n') if line.strip()]
                    texts = []
                    labels = []
                    
                    for line in lines:
                        try:
                            text, label = line.split(',', 1)
                            texts.append(text.strip())
                            labels.append(label.strip())
                        except ValueError:
                            st.error(f"Format invalide pour la ligne : {line}")
                            continue
                    
                    if texts and labels:
                        # Division train/test
                        train_size = int(len(texts) * train_size / 100)
                        train_texts = texts[:train_size]
                        train_labels = labels[:train_size]
                        val_texts = texts[train_size:]
                        val_labels = labels[train_size:]
                        
                        if classifier_type == "ML Classique":
                            # Transformation en vecteurs TF-IDF
                            vectors, vectorizer = text_to_tfidf(train_texts)
                            val_vectors = vectorizer.transform(val_texts)
                            
                            # Entraînement
                            accuracy, report = st.session_state.classifier.train(
                                vectors.toarray(), train_labels,
                                val_vectors.toarray(), val_labels
                            )
                            
                            training_info = {
                                "accuracy": accuracy,
                                "classification_report": report
                            }
                        elif classifier_type == "Deep Learning":
                            # Entraînement du modèle BERT
                            st.session_state.classifier.train(
                                train_texts, train_labels,
                                val_texts, val_labels,
                                batch_size=batch_size,
                                epochs=epochs,
                                learning_rate=learning_rate
                            )
                            
                            # Évaluation
                            accuracy = st.session_state.classifier.evaluate(val_texts, val_labels)
                            training_info = {
                                "accuracy": accuracy
                            }
                        elif classifier_type == "RNN":
                            # Entraînement du RNN
                            st.session_state.classifier.train(
                                train_texts, train_labels,
                                val_texts, val_labels,
                                batch_size=batch_size,
                                epochs=epochs,
                                learning_rate=learning_rate
                            )
                            
                            # Évaluation
                            accuracy = st.session_state.classifier.evaluate(val_texts, val_labels)
                            training_info = {
                                "accuracy": accuracy
                            }
                        else:
                            # Entraînement du Keras
                            st.session_state.classifier.train(
                                train_texts, train_labels,
                                val_texts, val_labels,
                                batch_size=batch_size,
                                epochs=epochs,
                                learning_rate=learning_rate
                            )
                            
                            # Évaluation
                            accuracy = st.session_state.classifier.evaluate(val_texts, val_labels)
                            training_info = {
                                "accuracy": accuracy
                            }
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "Entraînement terminé avec succès !",
                            "training_info": training_info
                        })

            except Exception as e:
                st.error(f"Une erreur s'est produite : {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Désolé, une erreur s'est produite. Veuillez réessayer."
                })
            finally:
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