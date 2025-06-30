import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
from tensorflow.keras.models import load_model
import re
import pickle
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import wikipedia

# ============================================================================
# TOKENIZER
# ============================================================================

class Mon_Tokenizer:
    """
    Tokenizer partagé entre les modèles DL pour assurer la cohérence du vocabulaire.
    """
    TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "models", "shared_tokenizer.pkl")
    
    def __init__(self, max_words=5000, max_len=200):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        self.is_fitted = False
        
        # Chargement automatique si le tokenizer existe déjà
        if os.path.exists(self.TOKENIZER_PATH):
            self.load_tokenizer()
    
    def fit_on_texts(self, texts):
        """Entraîne le tokenizer sur les textes."""
        if not self.is_fitted:
            self.tokenizer.fit_on_texts(texts)
            self.is_fitted = True
            self.save_tokenizer()
    
    def texts_to_sequences(self, texts):
        """Convertit les textes en séquences."""
        return self.tokenizer.texts_to_sequences(texts)
    
    def save_tokenizer(self):
        """Sauvegarde le tokenizer."""
        with open(self.TOKENIZER_PATH, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"✅ Tokenizer partagé sauvegardé dans {self.TOKENIZER_PATH}")
    
    def load_tokenizer(self):
        """Charge le tokenizer."""
        with open(self.TOKENIZER_PATH, 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.is_fitted = True
        print(f"✅ Tokenizer partagé chargé depuis {self.TOKENIZER_PATH}")
