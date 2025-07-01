"""
# ============================================================================
# MODÈLE DeepLearning
# ============================================================================

Classification par Deep Learning  
   - Architecture LSTM bidirectionnel avec BatchNormalization et Dropout.
   - Sauvegarde automatique du tokenizer et encoder dans app/models/
   - Même prétraitement que ML + tokenization Keras
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support
from mon_tokenizer import SharedTokenizer

class DLModel:
    """
    Modèle de Deep Learning pour la classification de texte.
    Utilise LSTM avec le tokenizer partagé.
    - Génération automatique des performances dans app/performances/
    - Early stopping avancé avec ReduceLROnPlateau
    """
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "dl_model.h5")
    ENCODER_PATH = os.path.join(os.path.dirname(__file__), "models", "dl_label_encoder.pkl")
    CLASSES_PATH = os.path.join(os.path.dirname(__file__), "models", "ml_classes.npy")
    TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "models", "shared_tokenizer.pkl")
    
    def __init__(self, max_words=5000, max_len=200):
        """
        Initialise le modèle DL.
        
        Args:
            max_words (int): Taille maximale du vocabulaire
            max_len (int): Longueur maximale des séquences
        """
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = SharedTokenizer(max_words=max_words, max_len=max_len)
        self.encoder = LabelEncoder()
        self.model = None
        self.history = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_classes = None
        self.y_test_classes = None
        
        # Chargement automatique si le modèle existe déjà
        if os.path.exists(self.MODEL_PATH):
            from tensorflow.keras.models import load_model
            self.model = load_model(self.MODEL_PATH)
            print(f"✅ Modèle DL chargé depuis {self.MODEL_PATH}")
            
        if os.path.exists(self.ENCODER_PATH):
            with open(self.ENCODER_PATH, 'rb') as f:
                self.encoder = pickle.load(f)
            print(f"✅ Label encoder chargé depuis {self.ENCODER_PATH}")

        if os.path.exists(self.CLASSES_PATH):
            self.encoder.classes_ = np.load(self.CLASSES_PATH, allow_pickle=True)
            print(f"✅ Mapping des labels (ML) chargé depuis {self.CLASSES_PATH} : {self.encoder.classes_}")
        else:
            print(f"⚠️ Mapping des labels (ML) NON trouvé dans {self.CLASSES_PATH}")

        if os.path.exists(self.TOKENIZER_PATH):
            self.tokenizer.load_tokenizer(self.TOKENIZER_PATH)
        else:
            print(f"⚠️ Tokenizer partagé NON trouvé dans {self.TOKENIZER_PATH}")

    def prepare(self, texts, labels):
        """
        Prépare les données pour le modèle DL.
        
        entrées:
            texts: Les textes
            labels: Les labels
            
        sorties:
            tuple: (X, y) préparés
        """
        # Entraînement du tokenizer si nécessaire
        if not self.tokenizer.is_fitted:
            self.tokenizer.fit_on_texts(texts)
        
        # Conversion en séquences
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len)
        
        # Encodage des labels
        y = to_categorical(self.encoder.fit_transform(labels))
        
        print(f"✅ Données DL préparées: {X.shape[0]} échantillons, {X.shape[1]} tokens")
        print(f"   Classes: {len(self.encoder.classes_)}")
        
        return X, y

    def build_model(self, num_classes):
        """
        Construit l'architecture du modèle.
        
        entrées:
            num_classes (int): Nombre de classes
        """
        self.model = Sequential([
            Embedding(self.max_words, 128, input_length=self.max_len),
            Bidirectional(LSTM(64, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        print("✅ Architecture LSTM construite")

    def train(self, X, y, validation_split=0.2, epochs=20, batch_size=64):
        """
        Entraîne le modèle
        
        entrées:
            X: Les features
            y: Les labels
            validation_split (float): Proportion des données de validation
            epochs (int): Nombre d'époques maximum
            batch_size (int): Taille des batchs
            
        sorties:
            tuple: (history, X_test, y_test)
        """
        # Division train/test
        from sklearn.model_selection import train_test_split
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Construction du modèle
        num_classes = y.shape[1]
        self.build_model(num_classes)
        

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
        # Sauvegarde du mapping des labels (partagé avec ML)
        np.save(self.CLASSES_PATH, self.encoder.classes_)
        print(f"Mapping des labels (ML) sauvegardé dans {self.CLASSES_PATH}")
        print(f"Modèle DL sauvegardé dans {self.MODEL_PATH}")
        print(f"Label encoder sauvegardé dans {self.ENCODER_PATH}")
        
        # Sauvegarde du tokenizer partagé
        self.tokenizer.save_tokenizer(self.TOKENIZER_PATH)
        
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
        """
        if self.model is None:
            raise ValueError("Le modèle n'est pas entraîné. Appelez train() d'abord.")
        if not self.tokenizer.is_fitted:
            raise RuntimeError("Tokenizer DL non entraîné ! Veuillez l'entraîner ou le charger avant la prédiction.")
        if not hasattr(self.encoder, 'classes_') or len(self.encoder.classes_) == 0:
            print(f"❌ Mapping des labels absent ou vide : {getattr(self.encoder, 'classes_', None)}")
            raise RuntimeError("Le mapping des labels (encoder.classes_) est absent ou vide ! Vérifiez le fichier ml_classes.npy ou réentraînez le modèle.")
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        preds = self.model.predict(padded)
        # On prend l'indice de la classe la plus probable pour chaque prédiction
        pred_indices = [np.argmax(p) for p in preds]
        pred_labels = [self.encoder.classes_[i] for i in pred_indices]
        return pred_labels

    def predict_proba(self, texts):
        """
        Retourne les probabilités de prédiction.
        
        entrées:
            texts: Les textes à classifier
            
        sorties:
            array: Les probabilités
        """
        if self.model is None:
            raise ValueError("Le modèle n'est pas entraîné. Appelez train() d'abord.")
        if not self.tokenizer.is_fitted:
            raise RuntimeError("Tokenizer DL non entraîné ! Veuillez l'entraîner ou le charger avant la prédiction.")
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        return self.model.predict(padded)

    def evaluate(self):
        """
        Évalue le modèle et génère les métriques de performance.
        """
        if self.X_test is None or self.y_pred is None:
            print("❌ Pas de données de test disponibles pour l'évaluation.")
            return
        
        print("📊 Évaluation du modèle DL...")
        self._generate_performance_metrics()