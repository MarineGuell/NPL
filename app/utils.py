"""
Module d'Utilitaires et Prétraitement - Pipeline de Données

Ce module fournit les composants essentiels du pipeline de données du chatbot Kaeru :

1. DataLoader : Chargement et préparation des datasets
   - Lecture CSV avec nettoyage automatique (doublons, valeurs manquantes)
   - Séparation textes/labels pour l'entraînement
   - Split automatique train/test avec stratification

2. TextPreprocessor : Prétraitement centralisé des textes
   - Nettoyage complet : ponctuation, URLs, emails, caractères spéciaux
   - Suppression des stopwords anglais et lemmatisation
   - Normalisation (espaces multiples, début/fin)
   - Pipeline appliqué à tous les textes (entraînement et inférence)

3. extract_keywords : Extraction de mots-clés TF-IDF
   - Utilisé pour la recherche Wikipedia intelligente
   - Vectorisation TF-IDF avec bigrammes
   - Sélection des termes les plus importants

4. search_wikipedia_smart : Recherche Wikipedia intelligente
   - Extraction automatique des mots-clés du texte utilisateur
   - Recherche de pages Wikipedia correspondantes
   - Gestion de l'ambiguïté avec suggestions interactives
   - Retour de résumés formatés

Pipeline de prétraitement standard :
Texte brut → Nettoyage → Normalisation → Stopwords → Lemmatisation → Texte prêt

Tous les composants sont optimisés pour la cohérence entre entraînement et inférence.
"""

import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class DataLoader:
    """
    Classe pour charger et préparer les données.
    """
    def __init__(self, filepath):
        """
        Initialise le chargeur de données.
        
        Args:
            filepath (str): Chemin vers le fichier CSV
        """
        self.data = pd.read_csv(filepath)
        self.clean_data()

    def clean_data(self):
        """
        Nettoie les données en supprimant les doublons et les valeurs manquantes.
        """
        # Suppression des doublons
        self.data = self.data.drop_duplicates()
        
        # Suppression des lignes avec valeurs manquantes
        self.data = self.data.dropna()
        
        # Réinitialisation de l'index
        self.data = self.data.reset_index(drop=True)

    def get_texts_and_labels(self):
        """
        Retourne les textes et les labels.
        
        Returns:
            tuple: (texts, labels)
        """
        return self.data['text'], self.data['category']

    def split_data(self, texts, labels, test_size=0.2, random_state=42):
        """
        Divise les données en ensembles d'entraînement et de test.
        
        Args:
            texts: Les textes
            labels: Les labels
            test_size (float): Proportion des données de test
            random_state (int): Seed pour la reproductibilité
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(texts, labels, test_size=test_size, random_state=random_state)

class TextPreprocessor:
    """
    Classe pour le prétraitement des textes.
    """
    def __init__(self):
        """
        Initialise le prétraiteur de texte.
        Télécharge les ressources NLTK nécessaires.
        """
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text):
        """
        Nettoie un texte de manière approfondie.
        
        Args:
            text (str): Le texte à nettoyer
            
        Returns:
            str: Le texte nettoyé
        """
        # Basic cleaning
        text = text.strip()
        text = text.lower()
        text = ''.join(char for char in text if not char.isdigit())

        # Nettoyage des emails
        text = re.sub(r'From:.*?Subject:', '', text, flags=re.DOTALL)
        text = re.sub(r'\S+@\S+', '', text)

        # Suppression des mots avec 3+ lettres consécutives identiques
        text = re.sub(r'\b\w*(\w)\1{2,}\w*\b', '', text)

        # Suppression des URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Suppression de la ponctuation
        for punctuation in string.punctuation:
            text = text.replace(punctuation, '')

        # Tokenization et suppression des mots vides
        tokenized_text = word_tokenize(text)
        tokenized_text_cleaned = [
            w for w in tokenized_text if not w in self.stop_words
        ]

        # Lemmatization
        lemmatized = [
            self.lemmatizer.lemmatize(word, pos="v")
            for word in tokenized_text_cleaned
        ]

        # Reconstruction du texte
        cleaned_text = ' '.join(word for word in lemmatized)

        return cleaned_text

    def normalize(self, text):
        """
        Normalise un texte.
        
        Args:
            text (str): Le texte à normaliser
            
        Returns:
            str: Le texte normalisé
        """
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        # Suppression des espaces en début et fin
        text = text.strip()
        
        return text

    def transform(self, texts):
        """
        Transforme une liste de textes.
        
        Args:
            texts: Les textes à transformer
            
        Returns:
            Series: Les textes transformés
        """
        # Nettoyage
        cleaned_texts = texts.apply(self.clean)
        
        # Normalisation
        normalized_texts = cleaned_texts.apply(self.normalize)
        
        return normalized_texts

def encode_labels(labels):
    """
    Encode les labels en utilisant LabelEncoder.
    
    Args:
        labels: Les labels à encoder
        
    Returns:
        tuple: (encoded_labels, encoder)
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels, encoder

def extract_keywords(text, max_keywords=5):
    """
    Extrait les mots-clés les plus importants d'un texte en utilisant TF-IDF.
    
    Args:
        text (str): Le texte à analyser
        max_keywords (int): Nombre maximum de mots-clés à extraire
        
    Returns:
        list: Liste des mots-clés triés par importance
    """
    # === EXTRACTION DE MOTS-CLÉS PAR TF-IDF ===
    #
    # ÉTAPE 1: Nettoyage et préparation du texte
    # - Conversion en minuscules pour normalisation
    # - Suppression de la ponctuation pour se concentrer sur les mots
    # - Remplacement par des espaces pour éviter les mots collés
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    # ÉTAPE 2: Vectorisation TF-IDF avec bigrammes
    # - Création d'un vectorizer TF-IDF spécialisé
    # - stop_words='english': Suppression des mots vides (the, a, is, etc.)
    # - ngram_range=(1, 2): Capture mots individuels ET expressions de 2 mots
    # - max_features=100: Limite le vocabulaire aux 100 termes les plus fréquents
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # Mots individuels et bigrammes
        max_features=100
    )
    
    # ÉTAPE 3: Calcul des scores TF-IDF
    # - Transformation du texte en matrice TF-IDF
    # - Extraction des noms de features (mots/expressions)
    # - Récupération des scores d'importance pour chaque terme
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    
    # ÉTAPE 4: Tri et sélection des mots-clés
    # - Association de chaque terme avec son score TF-IDF
    # - Tri décroissant par score (les plus importants en premier)
    # - Filtrage des termes avec score > 0 (élimination du bruit)
    # - Sélection des max_keywords termes les plus importants
    keyword_scores = list(zip(feature_names, scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Retour des mots-clés les plus importants
    return [keyword for keyword, score in keyword_scores[:max_keywords] if score > 0]

def search_wikipedia_smart(text):
    """
    Recherche Wikipedia intelligente basée sur l'extraction de mots-clés.
    
    Args:
        text (str): Le texte de l'utilisateur
        
    Returns:
        dict: Résultat avec statut, suggestions et données
    """
    import wikipedia
    
    try:
        # === RECHERCHE WIKIPEDIA INTELLIGENTE ===
        #
        # ÉTAPE 1: Extraction des mots-clés importants
        # - Utilisation de la fonction extract_keywords pour identifier
        #   les termes les plus significatifs dans le texte utilisateur
        # - Ces mots-clés serviront de base pour la recherche Wikipedia
        keywords = extract_keywords(text)
        
        # Vérification de la présence de mots-clés
        if not keywords:
            return {
                'status': 'error',
                'message': "I couldn't find any important keywords in your text, kero! 🐸"
            }
        
        # ÉTAPE 2: Recherche de pages Wikipedia pour chaque mot-clé
        # - Pour chaque mot-clé extrait, recherche de pages Wikipedia correspondantes
        # - Utilisation de wikipedia.search() pour trouver des pages similaires
        # - Limitation à 3 résultats par mot-clé pour éviter la surcharge
        suggestions = {}
        for keyword in keywords:
            try:
                # Recherche de pages similaires sur Wikipedia
                search_results = wikipedia.search(keyword, results=3)
                if search_results:
                    suggestions[keyword] = search_results
            except Exception:
                continue
        
        # Vérification de la présence de résultats
        if not suggestions:
            return {
                'status': 'error',
                'message': f"I couldn't find any Wikipedia pages for the keywords: {', '.join(keywords)}, kero! 🐸"
            }
        
        # ÉTAPE 3: Gestion des cas de recherche
        # - CAS A: Un seul mot-clé avec une seule page → Succès direct
        # - CAS B: Plusieurs options disponibles → Ambiguïté nécessitant confirmation
        if len(suggestions) == 1:
            keyword = list(suggestions.keys())[0]
            pages = suggestions[keyword]
            if len(pages) == 1:
                # CAS A: Une seule page trouvée, l'utiliser directement
                try:
                    # Récupération du résumé de la page Wikipedia
                    summary = wikipedia.summary(pages[0], sentences=3)
                    return {
                        'status': 'success',
                        'summary': summary,
                        'page': pages[0]
                    }
                except Exception as e:
                    return {
                        'status': 'error',
                        'message': f"Error accessing Wikipedia page: {str(e)}, kero! 🐸"
                    }
        
        # CAS B: Plusieurs options disponibles, retourner les suggestions
        # - L'interface utilisateur devra présenter ces options avec des boutons
        # - L'utilisateur pourra choisir la page qui l'intéresse le plus
        return {
            'status': 'ambiguous',
            'suggestions': suggestions,
            'keywords': keywords
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f"An error occurred: {str(e)}, kero! 🐸"
        }

def search_wikipedia(query):
    """
    Recherches Wikipedia for a given query.
    """
    import wikipedia
    try:
        # Set language to English
        wikipedia.set_lang("en")
        # Get the page
        page = wikipedia.page(query, auto_suggest=False)
        # Return a summary (e.g., first 3 sentences)
        return wikipedia.summary(query, sentences=3)
    except wikipedia.exceptions.PageError:
        return f"Sorry, I couldn't find a Wikipedia page for '{query}'."
    except wikipedia.exceptions.DisambiguationError as e:
        return f"'{query}' is ambiguous. Please be more specific. Options: {e.options[:5]}"
    except Exception as e:
        return f"An error occurred: {str(e)}"
