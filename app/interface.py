"""
Interface Utilisateur Streamlit - Chatbot Kaeru

Interface graphique du chatbot Kaeru développée avec Streamlit.
Personnalité : Grenouille japonaise qui ponctue ses phrases par "kero".

5 Fonctions Principales Disponibles :
1. Classification (Machine Learning) : TF-IDF + Naive Bayes optimisé
2. Classification (Deep Learning) : LSTM bidirectionnel avec BatchNormalization  
3. Summarization (Machine Learning) : Résumé extractif par similarité cosinus
4. Summarization (Deep Learning) : Résumé extractif par autoencodeur
5. Wikipedia Search : Recherche intelligente avec gestion d'ambiguïté

Fonctionnalités de l'Interface :
- Conversation persistante avec historique des messages
- Sélection de fonction via sidebar radio buttons
- Gestion intelligente de l'ambiguïté Wikipedia (boutons interactifs)
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
from utils import search_wikipedia_smart, DataLoader
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

# Initialisation de l'orchestrateur en session
@st.cache_resource
def get_orchestrator():
    return ChatbotOrchestrator()

orchestrator = get_orchestrator()

def main():
    st.title("🐸 Kaeru Chatbot")
    
    # === GESTION DE L'ÉTAT DE L'APPLICATION ===
    # Stockage de l'historique des messages pour maintenir la conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Stockage des suggestions Wikipedia en cas d'ambiguïté
    # - Cette variable permet de "mémoriser" les options proposées
    # - Elle persiste entre les interactions jusqu'à ce que l'utilisateur fasse un choix
    if "wiki_suggestions" not in st.session_state:
        st.session_state.wiki_suggestions = None

    # === CONFIGURATION DE LA BARRE LATÉRALE ===
    # Interface pour sélectionner la fonctionnalité souhaitée
    with st.sidebar:
        st.markdown("### ⚙️ Functions")
        st.markdown("---")
        # Sélection de la fonction parmi les 5 options disponibles
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
    # Restitution de tous les messages précédents pour maintenir le contexte
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])
    
    # === GESTION DES SUGGESTIONS WIKIPEDIA EN CAS D'AMBIGUÏTÉ ===
    # Cette section s'affiche uniquement quand il y a des suggestions en attente
    if st.session_state.wiki_suggestions:
        st.markdown("### 🤔 Which Wikipedia page would you like to explore, kero? 🐸")
        
        # === AFFICHAGE ORGANISÉ DES SUGGESTIONS ===
        # - Groupement par mot-clé pour une organisation logique
        # - Chaque mot-clé peut avoir plusieurs pages Wikipedia associées
        for keyword, pages in st.session_state.wiki_suggestions.items():
            st.markdown(f"**For '{keyword}':**")
            
            # === CRÉATION DE BOUTONS INTERACTIFS ===
            # - Un bouton pour chaque page Wikipedia trouvée
            # - Clés uniques pour éviter les conflits d'interface
            for i, page in enumerate(pages):
                if st.button(f"📖 {page}", key=f"wiki_{keyword}_{i}"):
                    # === TRAITEMENT DU CHOIX UTILISATEUR ===
                    # Récupération du résumé de la page sélectionnée
                    try:
                        import wikipedia
                        wikipedia.set_lang("en")
                        summary = wikipedia.summary(page, sentences=3)
                        
                        # === CRÉATION DE LA RÉPONSE PERSONNALISÉE ===
                        # - Message de la grenouille avec action appropriée
                        # - Affichage du résumé Wikipedia formaté
                        response = f"*hops to the knowledge pond* 🐸 Here's what I found about **{page}**, kero:\n\n{summary}"
                        
                        # === MISE À JOUR DE L'HISTORIQUE ===
                        # - Ajout de la réponse à l'historique de conversation
                        # - Réinitialisation des suggestions (fin de l'ambiguïté)
                        # - Rafraîchissement de l'interface
                        st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "🐸"})
                        st.session_state.wiki_suggestions = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error accessing Wikipedia: {str(e)}")
        
        # === BOUTON D'ANNULATION ===
        # - Permet à l'utilisateur d'annuler la recherche
        # - Nettoie l'état et permet une nouvelle interaction
        if st.button("❌ Cancel", key="cancel_wiki"):
            st.session_state.wiki_suggestions = None
            st.rerun()
            
    # === ZONE DE SAISIE ET TRAITEMENT DES REQUÊTES ===
    # Cette section s'affiche quand il n'y a pas de suggestions en attente
    elif prompt := st.chat_input("Drop your text here, kero..."):
        # === AJOUT DU MESSAGE UTILISATEUR À L'HISTORIQUE ===
        # - Enregistrement du message pour maintenir la conversation
        # - Affichage immédiat dans l'interface
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "🌸"})
        with st.chat_message("user", avatar="🦟"):
            st.markdown(prompt)
        
        # === TRAITEMENT DE LA DEMANDE SELON LA FONCTION SÉLECTIONNÉE ===
        # Traiter la demande et afficher la réponse de l'assistant
        with st.chat_message("assistant", avatar="🐸"):
            with st.spinner("I'm thinking, kero..."):
                response = ""
                
                # === ROUTAGE VERS LA FONCTIONNALITÉ APPROPRIÉE ===
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
                    # Utilisation de la fonction search_wikipedia_smart pour analyse intelligente
                    result = search_wikipedia_smart(prompt)
                    
                    # === GESTION DES DIFFÉRENTS CAS DE RÉPONSE ===
                    if result['status'] == 'success':
                        # CAS A: Succès direct - une seule page trouvée
                        # - Affichage immédiat du résumé Wikipedia
                        # - Message personnalisé de la grenouille
                        response = f"*dives into the knowledge pond* 🐸 Here's what I found about **{result['page']}**, kero:\n\n{result['summary']}"
                    elif result['status'] == 'ambiguous':
                        # CAS B: Ambiguïté - plusieurs pages disponibles
                        # - Stockage des suggestions pour affichage avec boutons
                        # - Message demandant à l'utilisateur de choisir
                        st.session_state.wiki_suggestions = result['suggestions']
                        response = f"*tilts head thoughtfully* 🐸 I found several Wikipedia pages that might interest you! Please choose one below, kero!"
                    else:  # error
                        # CAS C: Erreur - aucun résultat trouvé
                        # - Affichage du message d'erreur personnalisé
                        response = result['message']
                
                # === AFFICHAGE ET ENREGISTREMENT DE LA RÉPONSE ===
                # - Affichage de la réponse dans l'interface
                # - Ajout à l'historique de conversation pour persistance
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "🐸"})

if __name__ == "__main__":
    main()


