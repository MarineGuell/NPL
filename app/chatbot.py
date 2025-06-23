"""
Module Orchestrateur du Chatbot.
Ce module connecte l'interface utilisateur, le prétraitement et les modèles.
"""

from utils import TextPreprocessor
from models import MLModel, DLModel, AutoencoderSummarizer
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ChatbotOrchestrator:
    """
    Orchestre le pipeline complet :
    - Interface utilisateur (Streamlit)
    - Prétraitement (TextPreprocessor)
    - Classification (ML et DL)
    - Résumé (ML : TF-IDF/cosinus, DL : autoencodeur extractif)
    
    Détail du résumé DL (autoencodeur) :
    1. Lors de l'entraînement global, l'autoencodeur est entraîné sur toutes les phrases du dataset.
    2. Pour résumer un texte, on découpe en phrases, on vectorise, on passe chaque phrase dans l'autoencodeur.
    3. On calcule l'erreur de reconstruction pour chaque phrase.
    4. On sélectionne les phrases avec l'erreur la plus faible (les plus "centrales").
    5. Le résumé est la concaténation de ces phrases dans l'ordre d'origine.
    """
    
    def __init__(self):
        """
        Initialise l'orchestrateur avec tous les composants nécessaires.
        """
        print("🤖 Initialisation de l'orchestrateur...")
        self.preprocessor = TextPreprocessor()
        self.ml_classifier = MLModel()
        self.dl_classifier = DLModel()
        
        # Le modèle de résumé DL (Autoencodeur) est chargé ici
        print("🔄 Chargement du modèle de résumé (Autoencodeur)...")
        self.autoencoder_summarizer = AutoencoderSummarizer()
        self.is_trained = False
        print("✅ Orchestrateur prêt.")

    def train_models(self, texts, labels):
        """
        Entraîne tous les modèles :
        - ML (TF-IDF + Naive Bayes)
        - DL (LSTM bidirectionnel)
        - Autoencodeur pour le résumé extractif
        
        Étapes détaillées :
        1. Prétraitement des textes (nettoyage, normalisation)
        2. Entraînement du modèle ML + évaluation
        3. Préparation et entraînement du modèle DL
        4. Entraînement de l'autoencodeur sur toutes les phrases du dataset
        """
        print("🔄 Prétraitement des textes pour l'entraînement...")
        processed_texts = self.preprocessor.transform(texts)
        
        print("🔄 Entraînement du modèle ML...")
        self.ml_classifier.train(processed_texts, labels)
        self.ml_classifier.evaluate()
        
        print("🔄 Préparation et entraînement du modèle DL...")
        X_dl, y_dl = self.dl_classifier.prepare(processed_texts, labels)
        self.dl_classifier.train(X_dl, y_dl)
        
        # Entraînement de l'autoencodeur pour le résumé
        print("🔄 Entraînement de l'autoencodeur pour le résumé...")
        self.autoencoder_summarizer.train(texts.tolist())
        
        self.is_trained = True
        print("✅ Modèles entraînés avec succès.")

    def classify(self, text, model_type='ml'):
        """
        Classe un texte donné en utilisant le modèle spécifié.
        
        Args:
            text (str): Le texte brut à classifier.
            model_type (str): 'ml' ou 'dl'.
            
        Returns:
            str: La prédiction formatée.
        """
        if not self.is_trained:
            return "The models aren't trained yet! 🐸 Please run the training script first, kero!"

        print(f"🔄 Classification with {model_type.upper()} model...")
        # Prétraitement complet du texte d'entrée
        processed_text = self.preprocessor.normalize(self.preprocessor.clean(text))
        
        if model_type == 'ml':
            prediction = self.ml_classifier.predict([processed_text])[0]
            proba = self.ml_classifier.predict_proba([processed_text])[0].max()
        else: # 'dl'
            prediction = self.dl_classifier.predict([processed_text])[0]
            proba = self.dl_classifier.predict_proba([processed_text])[0].max()
        
        # Réponses personnalisées selon la confiance
        if proba > 0.8:
            return f"*hops excitedly* 🐸 This text is definitely about **{prediction}**! I'm {proba:.1%} confident, kero!"
        elif proba > 0.6:
            return f"*tilts head thoughtfully* 🐸 I think this text is about **{prediction}**. I'm {proba:.1%} sure, kero!"
        else:
            return f"*croaks uncertainly* 🐸 Hmm... I'm not very confident, but I'd say it's about **{prediction}** ({proba:.1%} sure). Maybe I need more training, kero!"

    def summarize(self, text, model_type='dl'):
        """
        Résume un texte donné en utilisant la méthode spécifiée.
        - 'ml' : résumé extractif TF-IDF/cosinus (phrases les plus similaires au texte global)
        - 'dl' : résumé extractif autoencodeur (phrases les mieux reconstruites)
        
        Détail du résumé autoencodeur :
        1. Découpage du texte en phrases
        2. Vectorisation de chaque phrase
        3. Passage dans l'autoencodeur
        4. Calcul de l'erreur de reconstruction
        5. Sélection des phrases avec l'erreur la plus faible
        6. Assemblage du résumé
        """
        print(f"🔄 Summarization with {model_type.upper()} method...")
        if model_type == 'ml':
            # === MÉTHODE EXTRACTIVE BASÉE SUR LA SIMILARITÉ COSINUS ===
            # 
            # ÉTAPE 1: Tokenisation du texte en phrases
            # Découpage du texte en phrases individuelles pour analyse
            sentences = sent_tokenize(text)
            if len(sentences) < 3:
                return "*splashes water* 🐸 This text is too short to summarize, kero! It's already quite concise!"
            
            # ÉTAPE 2: Vectorisation TF-IDF du texte entier
            # - Création d'un vectorizer TF-IDF avec bigrammes (1-2 mots)
            # - Suppression des mots vides anglais pour se concentrer sur le contenu
            # - Le vectorizer "apprend" le vocabulaire du texte entier
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            text_vector = vectorizer.fit_transform([text])
            
            # ÉTAPE 3: Vectorisation de chaque phrase
            # - Utilisation du même vectorizer pour assurer la cohérence
            # - Chaque phrase est transformée en vecteur avec le même vocabulaire
            sentence_vectors = vectorizer.transform(sentences)
            
            # ÉTAPE 4: Calcul de la similarité cosinus
            # - Mesure de l'angle entre chaque phrase et le texte entier
            # - Valeur proche de 1 = très similaire, proche de 0 = très différent
            # - Les phrases avec la plus haute similarité sont les plus représentatives
            similarities = cosine_similarity(sentence_vectors, text_vector).flatten()
            
            # ÉTAPE 5: Sélection des phrases les plus représentatives
            # - Tri des phrases par similarité décroissante
            # - Sélection des 3 phrases les plus similaires au texte entier
            # - Préservation de l'ordre original pour maintenir la cohérence narrative
            num_sentences = min(3, len(sentences))
            top_indices = similarities.argsort()[-num_sentences:][::-1]
            top_indices.sort()  # Garder l'ordre original des phrases
            
            # ÉTAPE 6: Assemblage du résumé
            # - Concaténation des phrases sélectionnées
            # - Maintien de la fluidité narrative
            summary = " ".join([sentences[i] for i in top_indices])
            return f"*jumps from lily pad to lily pad* 🐸 Here's what I found most representative, kero:\n\n{summary}"
        else: # 'dl'
            # === MÉTHODE EXTRACTIVE BASÉE SUR L'AUTOENCODEUR ===
            try:
                summary = self.autoencoder_summarizer.summarize(text, num_sentences=3)
                return f"*dives deep into the pond of knowledge* 🐸 Here's my autoencoder summary, kero:\n\n{summary}"
            except Exception as e:
                return f"*croaks apologetically* 🐸 Sorry, I couldn't summarize with the autoencoder: {str(e)}, kero!" 