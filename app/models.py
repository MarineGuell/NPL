"""
Module contenant les modèles de classification de texte.
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
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from tensorflow.keras.models import load_model
import re

class MLModel:
    """
    Modèle de Machine Learning optimisé pour la classification de texte.
    Utilise TF-IDF, Naive Bayes, et GridSearchCV pour l'optimisation.
    """
    MODEL_PATH = "models/ml_model.joblib"
    def __init__(self):
        """
        Initialise le modèle. Les composants seront définis lors de l'entraînement.
        """
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.report = None
        self.confusion_matrix_path = None
        self.learning_curve_path = None
        # Créer le dossier pour les plots s'il n'existe pas
        os.makedirs("app/plots", exist_ok=True)
        # Chargement automatique si le modèle existe
        if os.path.exists(self.MODEL_PATH):
            self.model = joblib.load(self.MODEL_PATH)

    def train(self, texts, labels):
        """
        Sépare les données, optimise les hyperparamètres et entraîne le meilleur modèle.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('model', MultinomialNB())
        ])
        
        # Grille d'hyperparamètres à tester
        param_grid = {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_df': [0.95, 1.0],
            'tfidf__min_df': [1, 2],
            'model__alpha': [0.5, 1.0]
        }
        
        print("🔍 Optimisation des hyperparamètres avec GridSearchCV...")
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.y_train)
        
        self.model = grid_search.best_estimator_
        print(f"Meilleurs hyperparamètres trouvés : {grid_search.best_params_}")
        # Sauvegarde du modèle entraîné
        joblib.dump(self.model, self.MODEL_PATH)
        print(f"Modèle ML sauvegardé dans {self.MODEL_PATH}")

    def evaluate(self):
        """
        Évalue le modèle sur le jeu de test et génère les visualisations.
        """
        if self.model is None:
            print("Le modèle doit d'abord être entraîné.")
            return

        print("📊 Évaluation du modèle...")
        y_pred = self.model.predict(self.X_test)
        
        # 1. Rapport de classification
        self.report = classification_report(self.y_test, y_pred)
        print("Rapport de Classification :\n", self.report)

        # 2. Matrice de confusion
        self.confusion_matrix_path = "app/plots/ml_confusion_matrix.png"
        fig, ax = plt.subplots(figsize=(10, 10))
        ConfusionMatrixDisplay.from_estimator(self.model, self.X_test, self.y_test, ax=ax, xticks_rotation='vertical')
        plt.title("Matrice de Confusion (Modèle ML)")
        plt.savefig(self.confusion_matrix_path)
        plt.close(fig)
        print(f"Matrice de confusion sauvegardée dans {self.confusion_matrix_path}")

        # 3. Courbe d'apprentissage
        self.learning_curve_path = "app/plots/ml_learning_curve.png"
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.X_train, self.y_train, cv=3, n_jobs=-1, 
            train_sizes=np.linspace(.1, 1.0, 5)
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score d'entraînement")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score de validation croisée")
        ax.set_title("Courbe d'Apprentissage (Modèle ML)")
        ax.set_xlabel("Taille de l'échantillon d'entraînement")
        ax.set_ylabel("Score")
        ax.legend(loc="best")
        ax.grid(True)
        plt.savefig(self.learning_curve_path)
        plt.close(fig)
        print(f"Courbe d'apprentissage sauvegardée dans {self.learning_curve_path}")

    def predict(self, texts):
        if self.model is None:
            raise RuntimeError("The ML model isn't trained yet! 🐸 Please run the training script first, kero!")
        return self.model.predict(texts)

    def predict_proba(self, texts):
        if self.model is None:
            raise RuntimeError("The ML model isn't trained yet! 🐸 Please run the training script first, kero!")
        return self.model.predict_proba(texts)

class DLModel:
    """
    Modèle de Deep Learning pour la classification de texte.
    Utilise LSTM.
    """
    MODEL_PATH = "models/dl_model.h5"
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
        self.encoder = LabelEncoder()
        # Chargement automatique si le modèle existe
        if os.path.exists(self.MODEL_PATH):
            self.model = load_model(self.MODEL_PATH)

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
        
        # Sauvegarde du modèle entraîné
        self.model.save(self.MODEL_PATH)
        print(f"Modèle DL sauvegardé dans {self.MODEL_PATH}")
        
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
        if self.model is None:
            raise RuntimeError("The DL model isn't trained yet! 🐸 Please run the training script first, kero!")
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
        if self.model is None:
            raise RuntimeError("The DL model isn't trained yet! 🐸 Please run the training script first, kero!")
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        return self.model.predict(padded)

class AutoencoderSummarizer:
    """
    Modèle de résumé extractif basé sur un autoencodeur.
    
    Processus détaillé :
    1. Découpage du texte en phrases (tokenisation).
    2. Nettoyage et vectorisation de chaque phrase (tokenizer Keras).
    3. Construction d'un autoencodeur séquentiel (Embedding + LSTM + Dense + RepeatVector + LSTM + TimeDistributed).
    4. Entraînement de l'autoencodeur à reconstruire les séquences de tokens des phrases.
    5. Pour le résumé :
       - On passe chaque phrase dans l'autoencodeur.
       - On calcule l'erreur de reconstruction (différence entre la phrase originale et la phrase reconstruite).
       - On sélectionne les phrases avec l'erreur la plus faible (les plus "représentatives").
    """
    MODEL_PATH = "models/autoencoder_summarizer.h5"
    
    def __init__(self, max_words=5000, embedding_dim=128, max_sentence_length=50):
        """
        Initialise le modèle d'autoencodeur pour le résumé.
        - max_words : taille du vocabulaire pour la tokenisation.
        - embedding_dim : dimension des embeddings de mots.
        - max_sentence_length : longueur maximale des phrases (padding/troncature).
        """
        self.max_words = max_words
        self.embedding_dim = embedding_dim
        self.max_sentence_length = max_sentence_length
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        self.model = None
        self.encoder = None
        self.decoder = None
        
        # Chargement automatique si le modèle existe déjà
        if os.path.exists(self.MODEL_PATH):
            self.model = load_model(self.MODEL_PATH)
            self.encoder = Sequential(self.model.layers[:3])
            self.decoder = Sequential(self.model.layers[3:])

    def preprocess_sentences(self, text):
        """
        Découpe le texte en phrases, nettoie et vectorise chaque phrase.
        - Utilise NLTK pour la tokenisation en phrases.
        - Nettoie la ponctuation, met en minuscules, retire les phrases trop courtes.
        - Vectorise chaque phrase avec le tokenizer Keras (mots -> indices).
        Retourne :
            - sentence_vectors : matrice (nb_phrases, max_sentence_length)
            - cleaned_sentences : phrases nettoyées (pour debug)
            - original_sentences : phrases originales (pour le résumé final)
        """
        from nltk.tokenize import sent_tokenize
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return [], [], sentences
        cleaned_sentences = []
        for sentence in sentences:
            cleaned = sentence.strip().lower()
            cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned)
            if len(cleaned.split()) > 2:
                cleaned_sentences.append(cleaned)
        if len(cleaned_sentences) < 2:
            return [], [], sentences
        self.tokenizer.fit_on_texts(cleaned_sentences)
        sequences = self.tokenizer.texts_to_sequences(cleaned_sentences)
        sentence_vectors = pad_sequences(sequences, maxlen=self.max_sentence_length)
        return sentence_vectors, cleaned_sentences, sentences

    def build_autoencoder(self):
        """
        Construit l'architecture séquentielle de l'autoencodeur :
        - Embedding : transforme les indices de mots en vecteurs denses.
        - LSTM (encodeur) : encode la séquence en un vecteur latent.
        - Dense : compression supplémentaire.
        - RepeatVector : répète le vecteur latent pour chaque pas de temps.
        - LSTM (decodeur) : reconstruit la séquence.
        - TimeDistributed(Dense) : prédit un mot à chaque position.
        """
        from tensorflow.keras.layers import RepeatVector, TimeDistributed
        self.model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_sentence_length),
            LSTM(64, return_sequences=False),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            RepeatVector(self.max_sentence_length),
            LSTM(self.embedding_dim, return_sequences=True),
            TimeDistributed(Dense(self.max_words, activation='softmax'))
        ])
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        self.encoder = Sequential(self.model.layers[:3])
        self.decoder = Sequential(self.model.layers[3:])

    def train(self, texts, epochs=10, batch_size=32):
        """
        Entraîne l'autoencodeur sur toutes les phrases du dataset.
        - Découpe chaque texte en phrases, vectorise.
        - Concatène toutes les phrases pour former le jeu d'entraînement.
        - Entraîne l'autoencodeur à reconstruire chaque phrase.
        - Sauvegarde le modèle entraîné.
        """
        print("🔄 Préparation des données pour l'autoencodeur...")
        all_sentences = []
        for text in texts:
            sentence_vectors, _, _ = self.preprocess_sentences(text)
            if len(sentence_vectors) > 0:
                all_sentences.extend(sentence_vectors)
        if len(all_sentences) < 10:
            print("⚠️ Pas assez de phrases pour entraîner l'autoencodeur")
            return
        X_train = np.array(all_sentences)
        print(f"Nombre total de phrases pour l'entraînement : {X_train.shape[0]}")
        print("🔄 Construction de l'autoencodeur...")
        self.build_autoencoder()
        print("🔄 Entraînement de l'autoencodeur...")
        self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )]
        )
        self.model.save(self.MODEL_PATH)
        print(f"✅ Autoencodeur sauvegardé dans {self.MODEL_PATH}")

    def summarize(self, text, num_sentences=3):
        """
        Résume un texte en sélectionnant les phrases les mieux reconstruites par l'autoencodeur.
        - Découpe et vectorise les phrases du texte.
        - Passe chaque phrase dans l'autoencodeur et calcule l'erreur de reconstruction.
        - Sélectionne les phrases avec l'erreur la plus faible (les plus "centrales").
        - Retourne le résumé (concaténation des phrases sélectionnées dans l'ordre d'origine).
        """
        if self.model is None:
            raise RuntimeError("The autoencoder isn't trained yet! 🐸 Please run the training script first, kero!")
        sentence_vectors, cleaned_sentences, original_sentences = self.preprocess_sentences(text)
        if len(sentence_vectors) == 0:
            return "*splashes water* 🐸 This text is too short to summarize, kero!"
        reconstruction_errors = []
        for i, sentence_vector in enumerate(sentence_vectors):
            reconstructed = self.model.predict(sentence_vector.reshape(1, -1), verbose=0)
            original_sequence = sentence_vector
            reconstructed_sequence = reconstructed[0].argmax(axis=-1)
            error = np.mean(np.abs(original_sequence - reconstructed_sequence))
            reconstruction_errors.append((i, error))
        reconstruction_errors.sort(key=lambda x: x[1])
        selected_indices = [idx for idx, _ in reconstruction_errors[:num_sentences]]
        selected_indices.sort()
        summary_sentences = [original_sentences[i] for i in selected_indices]
        summary = " ".join(summary_sentences)
        return summary 