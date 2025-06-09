"""
Module d'utilitaires pour le chatbot.
Ce script contient des fonctions utilitaires pour le prétraitement du texte,
la recherche Wikipedia et le résumé de texte.
"""

import re
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import wikipedia
from transformers import pipeline
from functools import lru_cache

# Téléchargement des ressources NLTK nécessaires pour le traitement du texte
nltk.download('punkt')  # Pour la tokenization
nltk.download('stopwords')  # Pour la suppression des mots vides
nltk.download('wordnet')  # Pour la lemmatization

# Initialisation du modèle de résumé en variable globale
summarizer = None

def get_summarizer():
    """
    Retourne l'instance du modèle de résumé, en le créant si nécessaire.
    """
    global summarizer
    if summarizer is None:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

def cleaning(sentence: str) -> str:
    """
    Nettoie et prétraite un texte en plusieurs étapes.
    
    Args:
        sentence (str): Le texte à nettoyer
        
    Returns:
        str: Le texte nettoyé et prétraité
    """
    # Nettoyage de base
    sentence = sentence.strip()  # Suppression des espaces en début et fin
    sentence = sentence.lower()  # Conversion en minuscules
    sentence = ''.join(char for char in sentence if not char.isdigit())  # Suppression des chiffres
    
    # Suppression des adresses email
    sentence = re.sub(r'From:.*?Subject:', '', sentence, flags=re.DOTALL)
    sentence = re.sub(r'\S+@\S+', '', sentence)

    # Suppression des mots avec 3 lettres consécutives identiques
    sentence = re.sub(r'\b\w*(\w)\1{2,}\w*\b', '', sentence)

    # Suppression des URLs
    sentence = re.sub(r'http\S+|www\S+|https\S+', '', sentence, flags=re.MULTILINE)
    
    # Nettoyage avancé : suppression de la ponctuation
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '')

    # Tokenization et suppression des mots vides
    tokenized_sentence = word_tokenize(sentence)
    tokenized_sentence_cleaned = [
        w for w in tokenized_sentence if not w in set(stopwords.words('english'))
    ]

    # Lemmatization des mots
    lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos = "v")
        for word in tokenized_sentence_cleaned
    ]

    # Reconstruction du texte
    cleaned_sentence = ' '.join(word for word in lemmatized)

    return cleaned_sentence

def preprocess_text(text: str) -> str:
    """
    Prétraite un texte en le nettoyant et en le normalisant.
    Args:
        text (str): Le texte à prétraiter
    Returns:
        str: Le texte prétraité
    """
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression de la ponctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Suppression des mots vides
    stop_words = set(stopwords.words('french'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Reconstruction du texte
    return " ".join(tokens)

def format_math_formulas(text: str) -> str:
    """
    Formate les formules mathématiques en LaTeX pour l'affichage.
    Args:
        text (str): Le texte contenant des formules mathématiques
    Returns:
        str: Le texte avec les formules formatées en LaTeX
    """
    # Remplace les formules inline (entre $...$)
    text = re.sub(r'\$(.*?)\$', r'$\1$', text)
    
    # Remplace les formules en bloc (entre $$...$$)
    text = re.sub(r'\$\$(.*?)\$\$', r'$$\1$$', text)
    
    # Remplace les formules LaTeX (entre \(...\))
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
    
    # Remplace les formules LaTeX en bloc (entre \[...\])
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text)
    
    return text

@lru_cache(maxsize=100)
def summarize_text(text: str) -> str:
    """
    Résume un texte en utilisant BART.
    Args:
        text (str): Le texte à résumer
    Returns:
        str: Le résumé du texte
    """
    try:
        # Utilisation du modèle de résumé singleton
        summarizer = get_summarizer()
        
        # Découpage du texte en morceaux si nécessaire (BART a une limite de 1024 tokens)
        max_chunk_length = 1000
        chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        # Résumé de chaque morceau
        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        
        # Combinaison des résumés
        return " ".join(summaries)
    except Exception as e:
        print(f"Erreur lors du résumé du texte : {str(e)}")
        return "Désolé, je n'ai pas pu résumer ce texte."

def search_wikipedia(query: str) -> str:
    """
    Effectue une recherche Wikipedia et retourne un résumé.
    Args:
        query (str): La requête de recherche
    Returns:
        str: Le résumé de l'article Wikipedia
    """
    try:
        # Recherche de l'article
        wikipedia.set_lang("fr")
        search_results = wikipedia.search(query, results=1)
        
        if not search_results:
            return "Aucun article trouvé sur ce sujet."
        
        # Récupération du contenu
        page = wikipedia.page(search_results[0])
        
        # Formatage du contenu
        content = page.content
        
        # Extraction des sections pertinentes
        sections = content.split('\n\n')
        relevant_sections = []
        
        for section in sections:
            # Ignore les sections vides ou trop courtes
            if len(section.strip()) < 50:
                continue
            # Ignore les sections de références, notes, etc.
            if any(keyword in section.lower() for keyword in ['références', 'notes', 'bibliographie', 'liens externes']):
                continue
            relevant_sections.append(section)
        
        # Limite à 3 sections maximum
        relevant_sections = relevant_sections[:3]
        
        # Formatage du texte avec les formules mathématiques
        formatted_content = format_math_formulas('\n\n'.join(relevant_sections))
        
        return f"Voici ce que j'ai trouvé sur {page.title} :\n\n{formatted_content}"
    except Exception as e:
        print(f"Erreur lors de la recherche Wikipedia : {str(e)}")
        return "Désolé, je n'ai pas pu trouver d'informations sur ce sujet dans Wikipedia."


