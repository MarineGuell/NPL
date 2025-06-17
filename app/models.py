"""
Module contenant les modèles de classification de texte.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

class MLModel:
    """
    Modèle de Machine Learning pour la classification de texte.
    Utilise TF-IDF et Naive Bayes.
    """
    def __init__(self):
        """
        Initialise le pipeline ML.
        """
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('model', MultinomialNB())
        ])

    def train(self, X_train, y_train):
        """
        Entraîne le modèle.
        
        Args:
            X_train: Les textes d'entraînement
            y_train: Les labels d'entraînement
        """
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle.
        
        Args:
            X_test: Les textes de test
            y_test: Les labels de test
        """
        preds = self.pipeline.predict(X_test)
        print(classification_report(y_test, preds))
        ConfusionMatrixDisplay.from_predictions(y_test, preds)

    def predict(self, texts):
        """
        Fait des prédictions sur de nouveaux textes.
        
        Args:
            texts: Les textes à classifier
            
        Returns:
            array: Les prédictions
        """
        return self.pipeline.predict(texts)

    def predict_proba(self, texts):
        """
        Retourne les probabilités de prédiction.
        
        Args:
            texts: Les textes à classifier
            
        Returns:
            array: Les probabilités
        """
        return self.pipeline.predict_proba(texts)

class DLModel:
    """
    Modèle de Deep Learning pour la classification de texte.
    Utilise LSTM.
    """
    def __init__(self, max_words=5000, max_len=200):
        """
        Initialise le modèle DL.
        
        Args:
            max_words (int): Nombre maximum de mots dans le vocabulaire
            max_len (int): Longueur maximale des séquences
        """
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        self.model = None
        self.encoder = None

    def prepare(self, texts, labels):
        """
        Prépare les données pour le modèle DL.
        
        Args:
            texts: Les textes
            labels: Les labels
            
        Returns:
            tuple: (X, y) préparés
        """
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len)
        y = self.encoder.fit_transform(labels)
        y = to_categorical(y)
        return X, y

    def build_model(self, num_classes):
        """
        Construit l'architecture du modèle.
        
        Args:
            num_classes (int): Nombre de classes
        """
        self.model = Sequential([
            Embedding(self.max_words, 128, input_length=self.max_len),
            LSTM(64, dropout=0.2, return_sequences=True),
            Dropout(0.5),
            LSTM(32),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    def train(self, X, y, validation_split=0.2, epochs=15, batch_size=64):
        """
        Entraîne le modèle.
        
        Args:
            X: Les features
            y: Les labels
            validation_split (float): Proportion des données de validation
            epochs (int): Nombre d'époques
            batch_size (int): Taille des batchs
            
        Returns:
            tuple: (history, X_test, y_test)
        """
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        self.build_model(num_classes=y.shape[1])
        
        history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )]
        )
        
        return history, X_test, y_test

    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle.
        
        Args:
            X_test: Les features de test
            y_test: Les labels de test
        """
        loss, acc = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {acc:.4f}")

    def predict(self, texts):
        """
        Fait des prédictions sur de nouveaux textes.
        
        Args:
            texts: Les textes à classifier
            
        Returns:
            list: Les prédictions
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        preds = self.model.predict(padded)
        return [self.encoder.classes_[np.argmax(p)] for p in preds]

    def predict_proba(self, texts):
        """
        Retourne les probabilités de prédiction.
        
        Args:
            texts: Les textes à classifier
            
        Returns:
            array: Les probabilités
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        return self.model.predict(padded) 