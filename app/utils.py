"""
Module d'utilitaires pour le prétraitement et le chargement des données.
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

def normalize_text(text):
    """
    Normalise un texte unique.
    
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
