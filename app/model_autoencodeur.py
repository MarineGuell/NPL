"""
# ============================================================================
# AUTOENCODEUR
# ============================================================================

AutoencoderSummarizer : Résumé extractif par autoencodeur
   - Découpage en phrases → Vectorisation → Autoencodeur → Erreur reconstruction
   - Sélection des phrases avec erreur de reconstruction la plus faible
   - Architecture : Embedding → LSTM → Dense → RepeatVector → LSTM → TimeDistributed
   - Sauvegarde automatique du tokenizer dans app/models/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from mon_tokenizer import SharedTokenizer

class AutoencoderSummarizer:
    """
    Modèle de résumé extractif basé sur un autoencodeur.
    Utilise le tokenizer partagé pour la cohérence.
    - Génération automatique des performances dans app/performances/
    - Early stopping avancé
    
    Processus détaillé :
    1. Découpage du texte en phrases (tokenisation).
    2. Vectorisation de chaque phrase avec le tokenizer partagé.
    3. Construction d'un autoencodeur séquentiel (Embedding + LSTM + Dense + RepeatVector + LSTM + TimeDistributed).
    4. Entraînement de l'autoencodeur à reconstruire les séquences de tokens des phrases.
    5. Pour le résumé :
       - On passe chaque phrase dans l'autoencodeur.
       - On calcule l'erreur de reconstruction (différence entre la phrase originale et la phrase reconstruite).
       - On sélectionne les phrases avec l'erreur la plus faible (les plus "représentatives").
    """
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "autoencoder_summarizer.h5")
    
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
        self.tokenizer = SharedTokenizer(max_words=max_words, max_len=max_sentence_length)
        self.model = None
        self.history = None
        
        # Chargement automatique si le modèle existe déjà
        if os.path.exists(self.MODEL_PATH):
            from tensorflow.keras.models import load_model
            self.model = load_model(self.MODEL_PATH)
            print(f"✅ Autoencodeur chargé depuis {self.MODEL_PATH}")

    def preprocess_sentences(self, text):
        """
        Découpe le texte en phrases et vectorise chaque phrase.
        - Utilise NLTK pour la tokenisation en phrases.
        - Vectorise chaque phrase avec le tokenizer partagé.
        Retourne :
            - sentence_vectors : matrice (nb_phrases, max_sentence_length)
            - original_sentences : phrases originales (pour le résumé final)
        """

        
        # Découpage en phrases
        sentences = sent_tokenize(text)
        
        # Utilise le tokenizer partagé
        sequences = self.tokenizer.texts_to_sequences(sentences)
        sentence_vectors = pad_sequences(sequences, maxlen=self.max_sentence_length)
        return sentence_vectors, sentences

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

    def train(self, texts, epochs=15, batch_size=32):
        """
        Entraîne l'autoencodeur sur toutes les phrases du dataset avec early stopping.
        - Découpe chaque texte en phrases, vectorise.
        - Concatène toutes les phrases pour former le jeu d'entraînement.
        - Entraîne l'autoencodeur à reconstruire chaque phrase.
        - Sauvegarde le modèle entraîné.
        """
        print("🔄 Préparation des données pour l'autoencodeur...")
        
        # Entraîner le tokenizer partagé si nécessaire
        if not self.tokenizer.is_fitted:
            print("🔄 Entraînement du tokenizer partagé...")
            # Extraire toutes les phrases pour entraîner le tokenizer
            all_sentences_for_tokenizer = []
            for text in texts:
                from nltk.tokenize import sent_tokenize
                sentences = sent_tokenize(text)
                all_sentences_for_tokenizer.extend(sentences)
            
            if len(all_sentences_for_tokenizer) > 0:
                self.tokenizer.fit_on_texts(all_sentences_for_tokenizer)
                print(f"✅ Tokenizer entraîné sur {len(all_sentences_for_tokenizer)} phrases")
            else:
                print("❌ Aucune phrase trouvée pour entraîner le tokenizer")
                return
        
        all_sentences = []
        all_sentence_vectors = []
        for text in texts:
            sentence_vectors, original_sentences = self.preprocess_sentences(text)
            if len(sentence_vectors) > 0:
                all_sentences.extend(original_sentences)
                all_sentence_vectors.extend(sentence_vectors)
        
        if len(all_sentences) < 10:
            print("⚠️ Pas assez de phrases pour entraîner l'autoencodeur")
            print(f"   Phrases trouvées: {len(all_sentences)}")
            print(f"   Textes analysés: {len(texts)}")
            return
        
        X_train = np.array(all_sentence_vectors)
        print(f"Nombre total de phrases pour l'entraînement : {X_train.shape[0]}")
        print(f"Forme des données d'entraînement : {X_train.shape}")
        print("🔄 Construction de l'autoencodeur...")
        self.build_autoencoder()
        print("🔄 Entraînement de l'autoencodeur avec early stopping...")
        
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
        
        self.history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        self.model.save(self.MODEL_PATH)
        print(f"✅ Autoencodeur sauvegardé dans {self.MODEL_PATH}")
        
        # Génération automatique des performances
        self._generate_performance_metrics()

    def _generate_performance_metrics(self):
        """
        Génère automatiquement les métriques de performance et les sauvegarde.
        """
        if self.history is None:
            print("⚠️ Pas d'historique d'entraînement pour générer les métriques")
            return
            
        # Création du dossier performances
        os.makedirs('app/performances', exist_ok=True)
        
        # 1. Métriques finales
        final_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        final_accuracy = self.history.history['accuracy'][-1]
        final_val_accuracy = self.history.history['val_accuracy'][-1]
        
        # 2. Sauvegarde des métriques en CSV
        metrics_df = pd.DataFrame({
            'Métrique': ['Loss finale (entraînement)', 'Loss finale (validation)', 
                        'Accuracy finale (entraînement)', 'Accuracy finale (validation)'],
            'Valeur': [final_loss, final_val_loss, final_accuracy, final_val_accuracy]
        })
        metrics_df.to_csv('app/performances/autoencoder_metrics.csv', index=False)
        
        # 3. Sauvegarde de l'historique d'entraînement
        history_df = pd.DataFrame(self.history.history)
        history_df.to_csv('app/performances/autoencoder_training_history.csv', index=False)
        
        # 4. Courbes d'apprentissage
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Loss
        ax1.plot(self.history.history['loss'], 'b-', linewidth=2, label='Entraînement')
        ax1.plot(self.history.history['val_loss'], 'r-', linewidth=2, label='Validation')
        ax1.set_title('Loss de l\'Autoencodeur', fontweight='bold')
        ax1.set_xlabel('Époques')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(self.history.history['accuracy'], 'b-', linewidth=2, label='Entraînement')
        ax2.plot(self.history.history['val_accuracy'], 'r-', linewidth=2, label='Validation')
        ax2.set_title('Accuracy de l\'Autoencodeur', fontweight='bold')
        ax2.set_xlabel('Époques')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('app/performances/autoencoder_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Architecture de l'autoencodeur
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.9, 'Architecture de l\'Autoencodeur', fontsize=16, fontweight='bold', ha='center')
        plt.text(0.5, 0.8, 'Embedding → LSTM → Dense → RepeatVector → LSTM → TimeDistributed', 
                fontsize=12, ha='center')
        plt.text(0.5, 0.7, f'Vocabulaire: {self.max_words} mots', fontsize=10, ha='center')
        plt.text(0.5, 0.6, f'Embedding dim: {self.embedding_dim}', fontsize=10, ha='center')
        plt.text(0.5, 0.5, f'Longueur max phrase: {self.max_sentence_length}', fontsize=10, ha='center')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('app/performances/autoencoder_architecture.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Métriques Autoencodeur générées dans app/performances/")

    def evaluate(self):
        """
        Évalue l'autoencodeur et génère les métriques de performance.
        Cette méthode est appelée automatiquement après l'entraînement.
        """
        if self.history is None:
            print("❌ L'autoencodeur n'a pas encore été entraîné!")
            return
        
        print("📊 ÉVALUATION DE L'AUTOENCODEUR")
        print("="*50)
        
        # Métriques finales
        final_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        final_accuracy = self.history.history['accuracy'][-1]
        final_val_accuracy = self.history.history['val_accuracy'][-1]
        
        print(f"📈 Loss finale (entraînement) : {final_loss:.4f}")
        print(f"📈 Loss finale (validation) : {final_val_loss:.4f}")
        print(f"📈 Accuracy finale (entraînement) : {final_accuracy:.4f}")
        print(f"📈 Accuracy finale (validation) : {final_val_accuracy:.4f}")
        
        # Génération des métriques de performance
        self._generate_performance_metrics()
        
        print("✅ Évaluation de l'autoencodeur terminée!")

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
        sentence_vectors, original_sentences = self.preprocess_sentences(text)
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