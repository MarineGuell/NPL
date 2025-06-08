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

# Téléchargement des ressources NLTK nécessaires pour le traitement du texte
nltk.download('punkt')  # Pour la tokenization
nltk.download('stopwords')  # Pour la suppression des mots vides
nltk.download('wordnet')  # Pour la lemmatization

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
    Effectue la tokenization, la suppression des mots vides et la lemmatization.
    
    Args:
        text (str): Le texte à prétraiter
        
    Returns:
        str: Le texte prétraité
    """
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression de la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Suppression des mots vides
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    # Lemmatization
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens]
    
    # Reconstruction du texte
    return " ".join(tokens)

def summarize_text(text: str, max_length: int = 150, use_advanced: bool = True) -> str:
    """
    Génère un résumé d'un texte en utilisant soit BART (avancé) soit TF-IDF (basique).
    
    Args:
        text (str): Le texte à résumer
        max_length (int): Longueur maximale du résumé
        use_advanced (bool): Utiliser BART si True, TF-IDF si False
        
    Returns:
        str: Le texte résumé
    """
    if use_advanced:
        try:
            # Utilisation du modèle BART pour un résumé avancé
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Erreur lors du résumé avancé, utilisation du résumé basique: {str(e)}")
            use_advanced = False
    
    if not use_advanced:
        try:
            # Méthode basique utilisant TF-IDF
            # Découpage du texte en phrases
            sentences = sent_tokenize(text)
            
            # Application de TF-IDF pour évaluer l'importance des phrases
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(sentences)

            # Calcul des scores d'importance
            scores = np.sum(X.toarray(), axis=1)

            # Sélection des phrases les plus importantes
            num_sentences = min(2, len(sentences))
            top_indices = scores.argsort()[-num_sentences:][::-1]

            # Reconstruction du résumé dans l'ordre original
            top_indices.sort()
            summary = [sentences[i] for i in top_indices]

            return " ".join(summary)
        except Exception as e:
            print(f"Erreur lors du résumé basique: {str(e)}")
            return text

def search_wikipedia(query: str, sentences: int = 2) -> str:
    """
    Recherche des informations sur Wikipedia et retourne un extrait.
    
    Args:
        query (str): La requête de recherche
        sentences (int): Nombre de phrases à retourner
        
    Returns:
        str: Un extrait du contenu Wikipedia
    """
    try:
        # Recherche de la page la plus pertinente
        search_results = wikipedia.search(query, results=1)
        if not search_results:
            return "Désolé, je n'ai pas trouvé d'informations sur ce sujet."
        
        # Récupération du contenu de la page
        page = wikipedia.page(search_results[0])
        content = page.content
        
        # Découpage en phrases
        sentences_list = sent_tokenize(content)
        
        # Retour des premières phrases
        return " ".join(sentences_list[:sentences])
    except Exception as e:
        print(f"Erreur lors de la recherche Wikipedia: {str(e)}")
        return "Désolé, une erreur s'est produite lors de la recherche."


