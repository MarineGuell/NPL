"""
DLModel : Classification par Deep Learning  
   - Architecture LSTM bidirectionnel avec BatchNormalization
   - Sauvegarde automatique du tokenizer et encoder dans app/models/
   - Early stopping et validation split
   - Même prétraitement que ML + tokenization Keras
   """

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

from mon_tokenizer import Mon_Tokenizer

# ============================================================================
# MODÈLE DL
# ============================================================================

class DLModel:
    """
    Modèle de Deep Learning pour la classification de texte.
    Utilise LSTM avec le tokenizer partagé.
    - Génération automatique des performances dans app/performances/
    - Early stopping avancé avec ReduceLROnPlateau
    """
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "dl_model.h5")
    ENCODER_PATH = os.path.join(os.path.dirname(__file__), "models", "dl_label_encoder.pkl")
    
    def __init__(self, max_words=5000, max_len=200):
        """
        Initialise le modèle DL.
        """
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Mon_Tokenizer(max_words=5000, max_len=200)  # Utilise le tokenizer partagé
        self.model = None
        self.encoder = LabelEncoder()
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.history = None
        
        print(f"🔍 Recherche du modèle DL dans: {self.MODEL_PATH}")
        print(f"🔍 Recherche de l'encoder dans: {self.ENCODER_PATH}")
        
        # Chargement automatique si le modèle existe
        if os.path.exists(self.MODEL_PATH):
            print("✅ Modèle DL trouvé, chargement...")
            self.model = load_model(self.MODEL_PATH)
            if os.path.exists(self.ENCODER_PATH):
                print("✅ Encoder trouvé, chargement...")
                with open(self.ENCODER_PATH, 'rb') as f:
                    self.encoder = pickle.load(f)
            else:
                print("❌ Encoder non trouvé")
        else:
            print("❌ Modèle DL non trouvé")

    def prepare(self, texts, labels):
        """
        Prépare les données pour le modèle DL.
        
        entrées:
            texts: Les textes
            labels: Les labels
            
        sorties:
            tuple: (X, y) préparés
        """
        # Vérification du tokenizer
        if not self.tokenizer.is_fitted:
            print("⚠️ Tokenizer non fitted, entraînement automatique...")
            self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len)
        y = self.encoder.fit_transform(labels)
        y = to_categorical(y)
        return X, y

    def build_model(self, num_classes):
        """
        Construit l'architecture du modèle.
        
        entrées:
            num_classes (int): Nombre de classes
        """
        self.model = Sequential([
            Embedding(self.max_words, 128, input_length=self.max_len),
            Bidirectional(LSTM(64, dropout=0.2, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.5),
            Bidirectional(LSTM(32)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    def train(self, X, y, validation_split=0.2, epochs=20, batch_size=64):
        """
        Entraîne le modèle avec early stopping avancé.
        
        entrées:
            X: Les features
            y: Les labels
            validation_split (float): Proportion des données de validation
            epochs (int): Nombre d'époques maximum
            batch_size (int): Taille des batchs
            
        sorties:
            tuple: (history, X_test, y_test)
        """
        from sklearn.model_selection import train_test_split
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        self.build_model(num_classes=y.shape[1])
        
        # Callbacks avancés pour l'early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        
        print("🔄 Entraînement du modèle DL avec early stopping...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Prédictions sur le test set
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_classes = np.argmax(self.y_pred, axis=1)
        self.y_test_classes = np.argmax(self.y_test, axis=1)
        
        # Sauvegarde du modèle entraîné et du label encoder
        self.model.save(self.MODEL_PATH)
        with open(self.ENCODER_PATH, 'wb') as f:
            pickle.dump(self.encoder, f)
        print(f"Modèle DL sauvegardé dans {self.MODEL_PATH}")
        print(f"Label encoder sauvegardé dans {self.ENCODER_PATH}")
        
        # Génération automatique des performances
        self._generate_performance_metrics()
        
        return self.history, self.X_test, self.y_test

    def _generate_performance_metrics(self):
        """
        Génère automatiquement les métriques de performance et les sauvegarde.
        """
        if self.y_test is None or self.y_pred is None:
            print("⚠️ Pas de données de test pour générer les métriques")
            return
            
        # Création du dossier performances
        os.makedirs('app/performances', exist_ok=True)
        
        # 1. Calcul des métriques
        accuracy = accuracy_score(self.y_test_classes, self.y_pred_classes)
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test_classes, self.y_pred_classes, average=None
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            self.y_test_classes, self.y_pred_classes, average='weighted'
        )
        
        # 2. Sauvegarde des métriques en CSV
        classes = self.encoder.classes_
        metrics_df = pd.DataFrame({
            'Classe': classes,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        # Ajout des métriques globales
        global_metrics = pd.DataFrame({
            'Métrique': ['Accuracy', 'Precision (weighted)', 'Recall (weighted)', 'F1-Score (weighted)'],
            'Valeur': [accuracy, precision_weighted, recall_weighted, f1_weighted]
        })
        
        # Sauvegarde CSV
        metrics_df.to_csv('app/performances/dl_metrics_by_class.csv', index=False)
        global_metrics.to_csv('app/performances/dl_global_metrics.csv', index=False)
        
        # 3. Matrice de confusion
        cm = confusion_matrix(self.y_test_classes, self.y_pred_classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, 
                   yticklabels=classes)
        plt.title('Matrice de Confusion - Modèle DL', fontsize=16, fontweight='bold')
        plt.ylabel('Vraies classes', fontsize=12)
        plt.xlabel('Classes prédites', fontsize=12)
        plt.tight_layout()
        plt.savefig('app/performances/dl_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Métriques par classe (graphique)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Precision
        ax1.bar(classes, precision, color='skyblue', alpha=0.7)
        ax1.set_title('Precision par Classe', fontweight='bold')
        ax1.set_ylabel('Precision')
        ax1.tick_params(axis='x', rotation=45)
        
        # Recall
        ax2.bar(classes, recall, color='lightcoral', alpha=0.7)
        ax2.set_title('Recall par Classe', fontweight='bold')
        ax2.set_ylabel('Recall')
        ax2.tick_params(axis='x', rotation=45)
        
        # F1-Score
        ax3.bar(classes, f1, color='lightgreen', alpha=0.7)
        ax3.set_title('F1-Score par Classe', fontweight='bold')
        ax3.set_ylabel('F1-Score')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('app/performances/dl_metrics_by_class.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Courbes d'apprentissage
        if self.history is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Accuracy
            ax1.plot(self.history.history['accuracy'], 'b-', linewidth=2, label='Entraînement')
            ax1.plot(self.history.history['val_accuracy'], 'r-', linewidth=2, label='Validation')
            ax1.set_title('Accuracy pendant l\'entraînement', fontweight='bold')
            ax1.set_xlabel('Époques')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Loss
            ax2.plot(self.history.history['loss'], 'b-', linewidth=2, label='Entraînement')
            ax2.plot(self.history.history['val_loss'], 'r-', linewidth=2, label='Validation')
            ax2.set_title('Loss pendant l\'entraînement', fontweight='bold')
            ax2.set_xlabel('Époques')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('app/performances/dl_learning_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. Sauvegarde de l'historique d'entraînement
        if self.history is not None:
            history_df = pd.DataFrame(self.history.history)
            history_df.to_csv('app/performances/dl_training_history.csv', index=False)
        
        print("✅ Métriques DL générées dans app/performances/")

    def predict(self, texts):
        """
        Fait des prédictions sur de nouveaux textes.
        
        entrées:
            texts: Les textes à classifier
            
        sorties:
            list: Les prédictions
        """
        if self.model is None:
            raise RuntimeError("The DL model isn't trained yet! 🐸 Please run the training script first, kero!")
        if not self.tokenizer.is_fitted:
            raise RuntimeError("Tokenizer DL non entraîné ! Veuillez l'entraîner ou le charger avant la prédiction.")
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        preds = self.model.predict(padded)
        return [self.encoder.classes_[np.argmax(p)] for p in preds]

    def predict_proba(self, texts):
        """
        Retourne les probabilités de prédiction.
        
        entrées:
            texts: Les textes à classifier
            
        sorties:
            array: Les probabilités
        """
        if self.model is None:
            raise RuntimeError("The DL model isn't trained yet! 🐸 Please run the training script first, kero!")
        if not self.tokenizer.is_fitted:
            raise RuntimeError("Tokenizer DL non entraîné ! Veuillez l'entraîner ou le charger avant la prédiction.")
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        return self.model.predict(padded)

    def evaluate(self):
        """
        Évalue le modèle et génère les métriques de performance.
        """
        if self.y_test is None or self.y_pred is None:
            print("⚠️ Pas de données de test pour évaluer le modèle")
            return
        
        print("📊 Évaluation du modèle DL...")
        self._generate_performance_metrics()
