"""
Interface Utilisateur Streamlit - Chatbot Kaeru

Interface graphique du chatbot Kaeru développée avec Streamlit.
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
from chatbot import ChatbotOrchestrator
from wikipedia_search import WikipediaIntelligentSearch
import os

# Fonction utilitaire pour lister les datasets disponibles
def list_datasets():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    return [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Kaeru Chatbot",
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

# Initialisation de l'orchestrateur et du système de recherche Wikipedia en session
@st.cache_resource
def get_orchestrator():
    return ChatbotOrchestrator()

@st.cache_resource
def get_wikipedia_search():
    return WikipediaIntelligentSearch()

orchestrator = get_orchestrator()
wiki_search = get_wikipedia_search()

def main():
    st.title("🐸 Kaeru Chatbot")
    
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

    # === CONFIGURATION DE LA BARRE LATÉRALE ===
    with st.sidebar:
        st.markdown("### ⚙️ Functions")
        st.markdown("---")
        task = st.radio(
            "What would you like to do, kero? 🐸",
            [
                "Classification (Machine Learning)",
                "Classification (Deep Learning)",
                "Summarization (Machine Learning)",
                "Summarization (Deep Learning)",
                "Wikipedia Search"
            ]
        )

    # === AFFICHAGE DE L'HISTORIQUE DE CONVERSATION ===
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])
    
    # === AFFICHAGE DES MOTS-CLÉS EXTRACTÉS (Wikipedia Search) ===
    if st.session_state.extracted_keywords and task == "Wikipedia Search":
        st.markdown("### 🔍 Mots-clés extraits par Kaeru:")
        keywords_display = []
        for keyword, score in st.session_state.extracted_keywords:
            keywords_display.append(f"**{keyword}** (confiance: {score:.2f})")
        st.markdown(" • ".join(keywords_display))
        st.markdown("---")
    
    # === GESTION DES SUGGESTIONS WIKIPEDIA ===
    if st.session_state.wiki_suggestions:
        st.markdown("### 🤔 Which Wikipedia page would you like to explore, kero? 🐸")
        
        # Affichage des suggestions avec scores de confiance
        for i, suggestion in enumerate(st.session_state.wiki_suggestions):
            confidence = suggestion['confidence']
            title = suggestion['title']
            keyword = suggestion['keyword']
            
            # Bouton avec score de confiance
            if st.button(f"📖 {title} (via '{keyword}', confiance: {confidence})", 
                        key=f"wiki_suggestion_{i}"):
                # Récupération du résumé de la page sélectionnée
                try:
                    summary_result = wiki_search.get_page_summary(title, sentences=4)
                    
                    if summary_result['status'] == 'success':
                        # Affichage du résumé avec autoencodeur si disponible
                        if 'autoencoder_summary' in summary_result:
                            response = f"""*hops to the knowledge pond* 🐸 Here's what I found about **{title}**, kero!

**Résumé Wikipedia:**
{summary_result['summary']}

**Résumé Kaeru (IA):**
{summary_result['autoencoder_summary']}"""
                        else:
                            response = f"*hops to the knowledge pond* 🐸 Here's what I found about **{title}**, kero!\n\n{summary_result['summary']}"
                        
                        # Mise à jour de l'historique
                        st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "🐸"})
                        st.session_state.wiki_suggestions = None
                        st.session_state.extracted_keywords = None
                        st.rerun()
                        
                    elif summary_result['status'] == 'ambiguous':
                        # Page ambiguë - proposer les options
                        options = summary_result['options']
                        st.markdown(f"**Multiple pages found for '{title}':**")
                        for j, option in enumerate(options):
                            if st.button(f"📄 {option}", key=f"ambiguous_{i}_{j}"):
                                # Récupérer le résumé de l'option choisie
                                option_summary = wiki_search.get_page_summary(option, sentences=4)
                                if option_summary['status'] == 'success':
                                    response = f"*hops to the knowledge pond* 🐸 Here's what I found about **{option}**, kero!\n\n{option_summary['summary']}"
                                    st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "🐸"})
                                    st.session_state.wiki_suggestions = None
                                    st.session_state.extracted_keywords = None
                                    st.rerun()
                    else:
                        st.error(summary_result['message'])
                        
                except Exception as e:
                    st.error(f"Error accessing Wikipedia: {str(e)}")
        
        # Bouton d'annulation
        if st.button("❌ Cancel", key="cancel_wiki"):
            st.session_state.wiki_suggestions = None
            st.session_state.extracted_keywords = None
            st.rerun()
            
    # === ZONE DE SAISIE ET TRAITEMENT DES REQUÊTES ===
    elif prompt := st.chat_input("Drop your text here, kero..."):
        # Ajout du message utilisateur à l'historique
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "🌸"})
        with st.chat_message("user", avatar="🦟"):
            st.markdown(prompt)
        
        # Traitement de la demande selon la fonction sélectionnée
        with st.chat_message("assistant", avatar="🐸"):
            with st.spinner("I'm thinking, kero..."):
                response = ""
                
                if task == "Classification (Machine Learning)":
                    response = orchestrator.classify(prompt, model_type='ml')
                elif task == "Classification (Deep Learning)":
                    response = orchestrator.classify(prompt, model_type='dl')
                elif task == "Summarization (Machine Learning)":
                    response = orchestrator.summarize(prompt, model_type='ml')
                elif task == "Summarization (Deep Learning)":
                    response = orchestrator.summarize(prompt, model_type='dl')
                elif task == "Wikipedia Search":
                    # === RECHERCHE WIKIPEDIA INTELLIGENTE ===
                    # Utilisation de la nouvelle fonction de recherche intelligente
                    search_result = wiki_search.intelligent_search(prompt, max_suggestions=8)
                    
                    if search_result['status'] == 'success':
                        # Stockage des mots-clés extraits pour affichage
                        st.session_state.extracted_keywords = search_result['keywords']
                        
                        # Stockage des suggestions pour affichage avec boutons
                        st.session_state.wiki_suggestions = search_result['suggestions']
                        
                        # Message informatif
                        response = f"""*tilts head thoughtfully* 🐸 {search_result['message']}

I extracted these keywords from your text using my AI models:
{', '.join([f"'{k[0]}' (score: {k[1]:.2f})" for k in search_result['keywords'][:5]])}

Please choose a page below, kero!"""
                        
                    else:  # error
                        response = search_result['message']
                
                # Affichage et enregistrement de la réponse
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "🐸"})

if __name__ == "__main__":
    main()


