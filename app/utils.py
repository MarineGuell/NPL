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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import wikipedia
from transformers import pipeline
from functools import lru_cache
import unicodedata
import emoji
import contractions
from bs4 import BeautifulSoup
import html
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

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
def summarize_text(text: str, use_dl: bool = True) -> str:
    """
    Résume un texte en utilisant soit BART (DL) soit TF-IDF (ML).
    Args:
        text (str): Le texte à résumer
        use_dl (bool): Si True, utilise BART (DL), sinon utilise TF-IDF (ML)
    Returns:
        str: Le résumé du texte
    """
    try:
        if use_dl:
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
        else:
            # Méthode ML avec TF-IDF
            from nltk.tokenize import sent_tokenize
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Découpage du texte en phrases
            sentences = sent_tokenize(text)
            
            if len(sentences) <= 3:
                return text
            
            # Création du vectoriseur TF-IDF
            vectorizer = TfidfVectorizer(stop_words='french')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calcul des scores pour chaque phrase
            sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
            
            # Sélection des phrases les plus importantes
            num_sentences = min(3, len(sentences))  # Limite à 3 phrases
            top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
            top_indices.sort()  # Garde l'ordre original
            
            # Construction du résumé
            summary = [sentences[i] for i in top_indices]
            return " ".join(summary)
            
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

def clean_text(text: str, remove_html: bool = True, remove_emojis: bool = True, 
               remove_urls: bool = True, remove_emails: bool = True, 
               remove_numbers: bool = False, remove_punctuation: bool = False) -> str:
    """
    Nettoie un texte en appliquant plusieurs transformations.
    
    Args:
        text (str): Texte à nettoyer
        remove_html (bool): Supprime les balises HTML
        remove_emojis (bool): Supprime les emojis
        remove_urls (bool): Supprime les URLs
        remove_emails (bool): Supprime les adresses email
        remove_numbers (bool): Supprime les nombres
        remove_punctuation (bool): Supprime la ponctuation
    
    Returns:
        str: Texte nettoyé
    """
    if not text:
        return ""
    
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des balises HTML
    if remove_html:
        text = BeautifulSoup(text, "html.parser").get_text()
        text = html.unescape(text)
    
    # Suppression des URLs
    if remove_urls:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Suppression des emails
    if remove_emails:
        text = re.sub(r'\S+@\S+', '', text)
    
    # Suppression des emojis
    if remove_emojis:
        text = emoji.replace_emoji(text, replace='')
    
    # Expansion des contractions (ex: "don't" -> "do not")
    text = contractions.fix(text)
    
    # Normalisation des caractères spéciaux
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    
    # Suppression des nombres
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Suppression de la ponctuation
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    
    # Suppression des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_patterns(text: str, pattern: str) -> list:
    """
    Extrait des motifs spécifiques d'un texte en utilisant des expressions régulières.
    
    Args:
        text (str): Texte à analyser
        pattern (str): Expression régulière à rechercher
    
    Returns:
        list: Liste des motifs trouvés
    """
    try:
        return re.findall(pattern, text)
    except re.error as e:
        print(f"Erreur dans l'expression régulière : {str(e)}")
        return []

def validate_regex(pattern: str) -> bool:
    """
    Vérifie si une expression régulière est valide.
    
    Args:
        pattern (str): Expression régulière à valider
    
    Returns:
        bool: True si l'expression est valide, False sinon
    """
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False

def get_regex_examples() -> dict:
    """
    Retourne des exemples d'expressions régulières courantes.
    
    Returns:
        dict: Dictionnaire d'exemples d'expressions régulières
    """
    return {
        "Email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "URL": r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
        "Date (JJ/MM/AAAA)": r'\d{2}/\d{2}/\d{4}',
        "Numéro de téléphone français": r'(?:(?:\+|00)33|0)\s*[1-9](?:[\s.-]*\d{2}){4}',
        "Code postal français": r'\b[0-9]{5}\b',
        "Hashtag": r'#\w+',
        "Mention Twitter": r'@\w+',
        "Heure (HH:MM)": r'\b([01]?[0-9]|2[0-3]):[0-5][0-9]\b',
        "IPv4": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        "Code hexadécimal": r'#[0-9a-fA-F]{6}'
    }

def text_to_tfidf(texts: list, max_features: int = 5000) -> tuple:
    """
    Transforme une liste de textes en vecteurs TF-IDF.
    
    Args:
        texts (list): Liste de textes à transformer
        max_features (int): Nombre maximum de features à extraire
    
    Returns:
        tuple: (vecteurs TF-IDF, vectoriseur)
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectors = vectorizer.fit_transform(texts)
    return vectors, vectorizer

def text_to_count(texts: list, max_features: int = 5000) -> tuple:
    """
    Transforme une liste de textes en vecteurs de comptage (Bag of Words).
    
    Args:
        texts (list): Liste de textes à transformer
        max_features (int): Nombre maximum de features à extraire
    
    Returns:
        tuple: (vecteurs de comptage, vectoriseur)
    """
    vectorizer = CountVectorizer(max_features=max_features)
    vectors = vectorizer.fit_transform(texts)
    return vectors, vectorizer

def text_to_bert_embeddings(texts: list, model_name: str = 'bert-base-uncased') -> np.ndarray:
    """
    Transforme une liste de textes en embeddings BERT.
    
    Args:
        texts (list): Liste de textes à transformer
        model_name (str): Nom du modèle BERT à utiliser
    
    Returns:
        np.ndarray: Matrice d'embeddings
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings

def encode_categorical(texts: list) -> tuple:
    """
    Encode des textes catégoriels en valeurs numériques.
    
    Args:
        texts (list): Liste de textes catégoriels
    
    Returns:
        tuple: (valeurs encodées, encodeur)
    """
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(texts)
    return encoded, encoder

def text_to_word_embeddings(texts: list, word2vec_model=None) -> np.ndarray:
    """
    Transforme une liste de textes en embeddings de mots moyens.
    
    Args:
        texts (list): Liste de textes à transformer
        word2vec_model: Modèle Word2Vec pré-entraîné (optionnel)
    
    Returns:
        np.ndarray: Matrice d'embeddings de mots moyens
    """
    if word2vec_model is None:
        # Utiliser un modèle BERT comme fallback
        return text_to_bert_embeddings(texts)
    
    # Tokenization
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    
    # Calcul des embeddings moyens
    embeddings = []
    for tokens in tokenized_texts:
        word_vectors = []
        for token in tokens:
            try:
                word_vectors.append(word2vec_model[token])
            except KeyError:
                continue
        if word_vectors:
            embeddings.append(np.mean(word_vectors, axis=0))
        else:
            embeddings.append(np.zeros(word2vec_model.vector_size))
    
    return np.array(embeddings)

def get_vectorization_info(vectors, vectorizer=None) -> dict:
    """
    Retourne des informations sur la vectorisation.
    
    Args:
        vectors: Vecteurs générés
        vectorizer: Vectoriseur utilisé (optionnel)
    
    Returns:
        dict: Informations sur la vectorisation
    """
    info = {
        "shape": vectors.shape,
        "sparsity": (1.0 - np.count_nonzero(vectors) / vectors.size) * 100 if hasattr(vectors, "size") else None,
        "feature_names": vectorizer.get_feature_names_out().tolist() if vectorizer is not None else None
    }
    return info

def save_vectors(vectors, filename: str, format: str = 'numpy'):
    """
    Sauvegarde les vecteurs dans un fichier.
    
    Args:
        vectors: Vecteurs à sauvegarder
        filename (str): Nom du fichier
        format (str): Format de sauvegarde ('numpy', 'csv', 'json')
    """
    if format == 'numpy':
        np.save(filename, vectors)
    elif format == 'csv':
        pd.DataFrame(vectors).to_csv(filename, index=False)
    elif format == 'json':
        pd.DataFrame(vectors).to_json(filename)


