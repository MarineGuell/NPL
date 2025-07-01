"""
Interface Utilisateur Streamlit - FrogBot Kaeru

Interface graphique du FrogBot Kaeru développée avec Streamlit.
Personnalité : Grenouille japonaise qui ponctue ses phrases par "kero".

5 Fonctions Principales Disponibles :
1. Classification (Machine Learning) : TF-IDF + Naive Bayes optimisé
2. Classification (Deep Learning) : LSTM bidirectionnel avec BatchNormalization  
3. Summarization (Machine Learning) : Résumé extractif par similarité cosinus
4. Summarization (Deep Learning) : Résumé extractif par autoencodeur
5. Wikipedia Search : Recherche intelligente avec extraction de mots-clés et modèles ML/DL

Fonctionnalités de l'Interface :
- Conversation persistante avec historique des messages
- Sélection de fonction via sidebar radio buttons
- Recherche Wikipedia intelligente avec extraction de mots-clés via modèles entraînés
- Gestion des suggestions Wikipedia avec boutons interactifs
- Messages personnalisés selon le niveau de confiance des modèles
- Actions descriptives de la grenouille (*hops excitedly*, *tilts head*, etc.)
- Chargement automatique des modèles via l'orchestrateur

Pipeline Utilisateur :
1. Sélection de la fonction dans la sidebar
2. Saisie du texte dans le chat input
3. Prétraitement automatique par l'orchestrateur
4. Prédiction avec le modèle approprié
5. Affichage de la réponse formatée avec personnalité

Tous les textes passent automatiquement par le pipeline de prétraitement
et de transformation numérique avant d'être traités par les modèles.
"""

import streamlit as st
from chatbot import TextProcessor
from wikipedia_search import WikipediaIntelligentSearch
import os
import random

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Kaeru Frogbot",
    page_icon="🐸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement du CSS (si le fichier existe)
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), "static", "style.css")
    if os.path.exists(css_file):
        with open(css_file) as f:
         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Mise en cache des modèles a utiliser
@st.cache_resource
def get_orchestrator():
    return TextProcessor()
orchestrator = get_orchestrator()
# Mise en cache de wiki
@st.cache_resource
def get_wikipedia_search():
    return WikipediaIntelligentSearch()
wiki_search = get_wikipedia_search()

# vider le cache
def clear_cache():
    st.cache_resource.clear()
    st.rerun()

def main():
    st.title("🐸 Kaeru Frogbot")
    
    # Texte d'introduction sous le titre
    st.markdown(
        """
        <div style='margin-bottom: 1.5em; font-size: 1.1em;'>
        Welcome!<br>
        I am Kaeru, your frog assistant to navigate the mysterious pond of knowledge.<br>
        I can help you understand the academic field of an unknown and intriguing text, kero! 🐸<br>
        If you don't have time to read all those words, I can also cut it short for you!<br>
        Finally, I can dive deep into the pond of knowledge to fetch more information about a specific topic, kero! 🐸<br>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # === GESTION DE L'ÉTAT DE L'APPLICATION ===
    # Stockage de l'historique des messages pour maintenir la conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Stockage des suggestions Wikipedia en cas d'ambiguïté
    if "wiki_suggestions" not in st.session_state:
        st.session_state.wiki_suggestions = None
    
    # Stockage des mots-clés extraits pour affichage
    if "extracted_keywords" not in st.session_state:
        st.session_state.extracted_keywords = None
    
    # Stockage des options ambiguës pour les pages Wikipedia
    if "ambiguous_options" not in st.session_state:
        st.session_state.ambiguous_options = None

    # === BARRE LATÉRALE ===
    with st.sidebar:
        task = st.radio(
            # options
            "### What can I do for you, kero? 🐸",
            [
                "Classification (Machine Learning)",
                "Classification (Deep Learning)",
                "Summarization (Machine Learning)",
                "Summarization (Deep Learning)",
                "Wikipedia Search"
            ]
        )
        
        # bouton clear cache
        st.markdown("---")
        st.markdown("### 🦟 Debug")
        if st.button('''🧠 Brain fog (Clear Cache) : \n\n"What where we saying again, kero ?"'''):
            clear_cache()

    
    # === AFFICHAGE DE L'HISTORIQUE GLOBAL DES MESSAGES ===
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar", "")):
            st.markdown(message["content"])

    # === ZONE DE SAISIE ET TRAITEMENT DES REQUÊTES ===
    if prompt := st.chat_input("Drop your text here, kero..."):
        # Ajout du message utilisateur à l'historique
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "🧙‍♀️"})
        with st.chat_message("user", avatar="🧙‍♀️"):
            st.markdown(prompt)
        
        # Traitement de la demande selon la fonction sélectionnée
        with st.chat_message("assistant", avatar="🐸"):
            with st.spinner("Let me see, kero..."):
                response = ""
                
                # classification Machine Learning
                if task == "Classification (Machine Learning)":
                    response_str = orchestrator.classify(prompt, model_type='ml')
                    # Extraction du label et de la confiance
                    try:
                        label_line, conf_line = response_str.split('\n')
                        label = label_line.split(':', 1)[1].strip()
                        confidence = float(conf_line.split(':', 1)[1].strip())
                        if confidence >= 0.8:
                            response = f"I'm **{confidence*100:.0f}%** sure this is about **{label}**, kero! 🐸 *The frog puffs up its chest proudly.*"
                        elif confidence >= 0.5:
                            response = f"I think this is about **{label}** (confidence: {confidence*100:.0f}%), kero... *The frog tilts its head, a bit unsure.*"
                        else:
                            response = f"Hmm... I'm not sure, but maybe **{label}** ? (confidence: {confidence*100:.0f}%), kero... *The frog looks around, uncertain.*"
                    except Exception:
                        response = response_str

                # Classification Deep Learning
                elif task == "Classification (Deep Learning)":
                    response_str = orchestrator.classify(prompt, model_type='dl')
                    try:
                        label_line, conf_line = response_str.split('\n')
                        label = label_line.split(':', 1)[1].strip()
                        confidence = float(conf_line.split(':', 1)[1].strip())
                        if confidence >= 0.8:
                            response = f"I'm **{confidence*100:.0f}%** sure this is about **{label}**, kero! 🐸 *The frog puffs up its chest proudly.*"
                        elif confidence >= 0.5:
                            response = f"I think this isabout **{label}** (confidence: {confidence*100:.0f}%), kero... *The frog tilts its head, a bit unsure.*"
                        else:
                            response = f"Hmm... I'm not sure, but maybe **{label}**? (confidence: {confidence*100:.0f}%), kero... *The frog looks around, uncertain.*"
                    except Exception:
                        response = response_str

                # resumé Machine Learning
                elif task == "Summarization (Machine Learning)":
                    response_str = orchestrator.summarize(prompt, model_type='ml')
                    # Séparation résumé / mots-clés
                    if "Mots-clés importants :" in response_str:
                        summary_part, keywords_part = response_str.split("Mots-clés importants :", 1)
                        summary = summary_part.replace("Résumé :", "").strip()
                        keywords = keywords_part.strip()
                    else:
                        summary = response_str.strip()
                        keywords = None
                    intro_choices = [
                        f"In short, this text says: {summary}, kero! 🐸",
                        f"Here's the gist: {summary}, kero! 🐸",
                        f"To summarize: {summary}, kero! 🐸"
                    ]
                    response = random.choice(intro_choices)
                    if keywords:
                        kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
                        if kw_list:
                            response += f"\n\nI got {len(kw_list)} big idea{'s' if len(kw_list)>1 else ''} from it: **{', '.join(kw_list)}**, kero!"

                # resumé Deep Learning
                elif task == "Summarization (Deep Learning)":
                    summary = orchestrator.summarize(prompt, model_type='dl')
                    intro_choices = [
                        f"In short, this text says: {summary}, kero! 🐸",
                        f"Here's the gist: {summary}, kero! 🐸",
                        f"To summarize: {summary}, kero! 🐸"
                    ]
                    response = random.choice(intro_choices)
                
                # Recherche Wikipedia     
                elif task == "Wikipedia Search":
                    search_result = wiki_search.get_page_summary(prompt, sentences=4)
                    if search_result['status'] == 'success':
                        # Vérifie si le titre retourné est différent de la requête utilisateur (non strictement égal, insensible à la casse et espaces)
                        if search_result['title'].strip().lower() != prompt.strip().lower():
                            response = f'''I'm not sure about '{prompt}', I couldn't find something about it specifically.\n\n 
                            But here is what I can tell you about {search_result['title']}, kero :\n\n{search_result['summary']}'''
                        else:
                            response = f"Here's what I found about **{search_result['title']}**, kero! 🐸\n\n{search_result['summary']}"
                    else:
                        response = f"Sorry, I couldn't find a Wikipedia page for your query, kero! 🐸 Please try another word or phrase."

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "🐸"})


if __name__ == "__main__":
    main()


