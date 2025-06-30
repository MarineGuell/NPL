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

# Fonction utilitaire pour lister les datasets disponibles
def list_datasets():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    return [f for f in os.listdir(data_dir) if f.endswith('.csv')]

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
        Welcome !<br>
        I am <b>Kaeru</b>, your frog assistant to navigate the mysterious pond of knowledge.<br>
        I can help you understand the academic field of an unknown and intriguing text, kero ! 🐸<br>
        If you don't have time to read all those words, I can also cut it short for you !<br>
        Finally, I can dive deep into the pond of knowledge too fetch more information about a specific topic, kero ! 🐸<br>
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

    # === CONFIGURATION DE LA BARRE LATÉRALE ===
    with st.sidebar:
        task = st.radio(
            "### What can I do for you, kero? 🐸",
            [
                "Classification (Machine Learning)",
                "Classification (Deep Learning)",
                "Summarization (Machine Learning)",
                "Summarization (Deep Learning)",
                "Wikipedia Search"
            ]
        )
        
        st.markdown("---")
        st.markdown("### 🦟 Debug")
        if st.button('''🧠 Brain fog (Clear Cache) : \n\n"What where we saying again, kero ?"'''):
            clear_cache()

##########################################################################
     
#     # === AFFICHAGE DES MOTS-CLÉS EXTRACTÉS (Wikipedia Search) ===
#     if st.session_state.extracted_keywords and task == "Wikipedia Search":
#         st.markdown("I understood those words :")
#         keywords_display = []
#         for keyword, score in st.session_state.extracted_keywords:
#             keywords_display.append(f"**{keyword}** (confiance: {score:.2f})")
#         st.markdown(" • ".join(keywords_display))
#         # st.markdown("---")
    
#     # === GESTION DES SUGGESTIONS WIKIPEDIA ===
#     if st.session_state.wiki_suggestions and not st.session_state.ambiguous_options:
#         st.markdown("So I should search about... ?")
        
#         # Affichage des suggestions avec scores de confiance
#         for i, suggestion in enumerate(st.session_state.wiki_suggestions):
#             confidence = suggestion['confidence']
#             title = suggestion['title']
#             keyword = suggestion['keyword']
            
#             # Bouton avec score de confiance
#             if st.button(f"{title}", #  (via '{keyword}', confiance: {confidence})
#                         key=f"wiki_suggestion_{i}"):
#                 # Récupération du résumé de la page sélectionnée
#                 try:
#                     summary_result = wiki_search.get_page_summary(title, sentences=4)
                    
#                     if summary_result['status'] == 'success':
#                         # Affichage du résumé avec autoencodeur si disponible
#                         if 'autoencoder_summary' in summary_result:
#                             response = f"""Here's what I found about **{title}**, kero! 🐸

# {summary_result['summary']} Kero 🐸

# **But if you want me to be short : **

# {summary_result['autoencoder_summary']} Kero 🐸"""
#                         else:
#                             response = f"Here's what I found about **{title}**, kero! 🐸 \n\n{summary_result['summary']} Kero 🐸"
                        
#                         # Mise à jour de l'historique
#                         st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "🐸"})
#                         st.session_state.wiki_suggestions = None
#                         st.session_state.extracted_keywords = None
#                         st.rerun()
                        
#                     elif summary_result['status'] == 'ambiguous':
#                         # Page ambiguë - stocker les options pour affichage séparé
#                         options = summary_result['options']
#                         st.session_state.ambiguous_options = options
#                         st.session_state.wiki_suggestions = None
#                         st.rerun()
#                     else:
#                         st.error(summary_result['message'])
                        
#                 except Exception as e:
#                     st.error(f"Error accessing Wikipedia: {str(e)}")
        
#         # Bouton d'annulation
#         if st.button("❌ Cancel", key="cancel_wiki"):
#             st.session_state.wiki_suggestions = None
#             st.session_state.extracted_keywords = None
#             st.session_state.ambiguous_options = None
#             st.rerun()
    
#     # === GESTION DES OPTIONS AMBIGUËS WIKIPEDIA ===
#     elif st.session_state.ambiguous_options:
#         st.markdown("**Multiple pages found, which one do you want?**")
        
#         # Affichage des options avec boutons
#         for j, option in enumerate(st.session_state.ambiguous_options):
#             if st.button(f"📄 {option}", key=f"ambiguous_option_{j}"):
#                 # Récupérer le résumé de l'option choisie
#                 try:
#                     option_summary = wiki_search.get_page_summary(option, sentences=4)
#                     if option_summary['status'] == 'success':
#                         # Affichage du résumé avec autoencodeur si disponible
#                         if 'autoencoder_summary' in option_summary:
#                             response = f"""Here's what I found about **{option}**, kero! 🐸

# {option_summary['summary']} Kero 🐸

# **But if you want me to be short : **

# {option_summary['autoencoder_summary']} Kero 🐸"""
#                         else:
#                             response = f"Here's what I found about **{option}**, kero! 🐸 \n\n{option_summary['summary']} Kero 🐸"
                        
#                         # Mise à jour de l'historique
#                         st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "🐸"})
#                         st.session_state.wiki_suggestions = None
#                         st.session_state.extracted_keywords = None
#                         st.session_state.ambiguous_options = None
#                         st.rerun()
#                     else:
#                         st.error(option_summary['message'])
#                 except Exception as e:
#                     st.error(f"Error accessing Wikipedia: {str(e)}")
        
#         # Bouton d'annulation pour les options ambiguës
#         if st.button("❌ Cancel", key="cancel_ambiguous"):
#             st.session_state.wiki_suggestions = None
#             st.session_state.extracted_keywords = None
#             st.session_state.ambiguous_options = None
#             st.rerun()
    
#     # === ZONE DE SAISIE SPÉCIFIQUE POUR WIKIPEDIA ===
#     elif task == "Wikipedia Search":
#         # Zone de saisie dédiée pour Wikipedia
#         wiki_prompt = st.text_input("*What knowladge do you seek, kero ?🐸*", placeholder="Tell Kaeru...")
        
#         if wiki_prompt:
#             with st.spinner("Searching Wikipedia, kero..."):
#                 # === RECHERCHE WIKIPEDIA INTELLIGENTE ===
#                 search_result = wiki_search.smart_search_by_combinations(wiki_prompt, max_suggestions=8)
                
#                 if search_result['status'] == 'success':
#                     # Stockage des mots-clés extraits pour affichage
#                     st.session_state.extracted_keywords = search_result['keywords']
                    
#                     # Stockage des suggestions pour affichage avec boutons
#                     st.session_state.wiki_suggestions = search_result['suggestions']
                    
#                     # Pas besoin d'ajouter de message ici, les suggestions seront affichées
#                     st.rerun()
                    
#                 else:  # error
#                     st.error(search_result['message'])
    
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
                
                if task == "Classification (Machine Learning)":
                    response = orchestrator.classify(prompt, model_type='ml')
                    print('Tu en fais quoi ?')

                elif task == "Classification (Deep Learning)":
                    response = orchestrator.classify(prompt, model_type='dl')
                    print('Tu en fais quoi ?')

                elif task == "Summarization (Machine Learning)":
                    response = orchestrator.summarize(prompt, model_type='ml')
                    print('Tu en fais quoi ?')

                elif task == "Summarization (Deep Learning)":
                    response = orchestrator.summarize(prompt, model_type='dl')
                    print('Tu en fais quoi ?')

                    
                elif task == "Wikipedia Search":
                    # === RECHERCHE WIKIPEDIA INTELLIGENTE ===
                    # Utilisation de la nouvelle fonction de recherche intelligente
                    search_result = wiki_search.smart_search_by_combinations(prompt, max_suggestions=8)
                    
                    if search_result['status'] == 'success':
                        # Stockage des mots-clés extraits pour affichage
                        st.session_state.extracted_keywords = search_result['keywords']
                        
                        # Stockage des suggestions pour affichage avec boutons
                        st.session_state.wiki_suggestions = search_result['suggestions']
                        
                        # Pas besoin d'ajouter de message ici, les suggestions seront affichées
                        st.rerun()
                        
                    else:  # error
                        response = search_result['message']


if __name__ == "__main__":
    main()


