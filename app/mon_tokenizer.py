"""
Module de tokenizer personnalisé pour la cohérence entre les modèles.
"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

class SharedTokenizer:
    """
    Tokenizer partagé entre :
    - Le modèle LSTM (DLModel)
    - L'autoencodeur (AutoencoderSummarizer)
    """
    
    def __init__(self, max_words=5000, max_len=200): # TODO d'autres taille ?
        """
        Initialise le tokenizer partagé.
        
        Args:
            max_words (int): Taille maximale du vocabulaire
            max_len (int): Longueur maximale des séquences
        """
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.is_fitted = False
        
    def fit_on_texts(self, texts):
        """
        Entraîne le tokenizer
        
        Args:
            texts (list): Liste de textes à utiliser pour l'entraînement
        """
        self.tokenizer.fit_on_texts(texts)
        self.is_fitted = True
        print(f"Tokenizer entraîné sur {len(texts)} textes")
        print(f"Vocabulaire: {len(self.tokenizer.word_index)} mots")
        
    def texts_to_sequences(self, texts):
        """
        Convertit les textes en séquences numériques.
        
        Args:
            texts (list): Liste de textes à convertir
            
        Returns:
            list: Séquences numériques
        """
        if not self.is_fitted:
            raise RuntimeError("Le tokenizer n'est pas encore entraîné!")
        return self.tokenizer.texts_to_sequences(texts)

    def save_tokenizer(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"✅ Tokenizer sauvegardé dans {path}")

    def load_tokenizer(self, path):
        with open(path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.is_fitted = True
        print(f"✅ Tokenizer chargé depuis {path}")
