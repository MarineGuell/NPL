"""
Module principal du chatbot.
Ce script implémente la logique du chatbot, incluant la classification de texte,
la génération de réponses et l'intégration avec différents modèles de langage.
"""

import os
import torch
import joblib
import numpy as np
from typing import Optional, Dict, Any
import time
from functools import lru_cache
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score

from utils import summarize_text, search_wikipedia, preprocess_text
from models import SupervisedClassifier, DeepLearningClassifier, RNNTextClassifier, KerasTextClassifier

class Chatbot:
    """
    Classe principale du chatbot.
    Gère l'initialisation des modèles, la classification de texte et la génération de réponses.
    """
    
    def __init__(self):
        """Initialise le chatbot avec les modèles nécessaires."""
        self.model_name = "microsoft/DialoGPT-medium"  # Modèle de langage utilisé
        self.tokenizer = None  # Tokenizer pour le modèle de langage
        self.model = None  # Modèle de langage
        self.classifier_model = None  # Modèle de classification ML
        self.vectorizer = None  # Vectoriseur pour la classification ML
        self.bert_tokenizer = None  # Tokenizer BERT
        self.bert_model = None  # Modèle BERT pour la classification
        self.optimized_model = None  # Modèle ML optimisé
        self.bert_labels = [
            "accueil", "météo", "technologie", "cuisine", "actualités",
            "éducation", "histoire", "sport"
        ]
        self.classifier_ml = SupervisedClassifier()
        self.classifier_dl = DeepLearningClassifier()
        self.classifier_rnn = RNNTextClassifier()
        self.classifier_keras = KerasTextClassifier()
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.initialize_models()

    def optimize_ml_model(self, texts, labels):
        """Optimise le modèle ML avec GridSearchCV"""
        try:
            # Prétraitement des textes
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Création du pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('clf', MultinomialNB())
            ])
            
            # Définition des paramètres à tester
            param_grid = {
                'tfidf__max_features': [3000, 5000, 7000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'clf__alpha': [0.1, 0.5, 1.0]
            }
            
            # Calcul du nombre optimal de plis
            n_samples = len(processed_texts)
            n_splits = min(5, n_samples // 2)  # Utilise le minimum entre 5 et la moitié du nombre d'échantillons
            if n_splits < 2:
                n_splits = 2  # Minimum 2 plis pour la validation croisée
            
            # Création du GridSearchCV avec le nombre de plis adapté
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=n_splits,
                n_jobs=-1,
                verbose=1
            )
            
            # Division des données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(
                processed_texts, labels,
                test_size=0.2,
                random_state=42,
                stratify=labels
            )
            
            # Entraînement du modèle
            grid_search.fit(X_train, y_train)
            
            # Sauvegarde du meilleur modèle
            self.ml_model = grid_search.best_estimator_
            
            # Évaluation du modèle
            y_pred = self.ml_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Meilleure précision : {accuracy:.2f}")
            print(f"Meilleurs paramètres : {grid_search.best_params_}")
            
            return accuracy
            
        except Exception as e:
            print(f"Erreur lors de l'optimisation du modèle ML : {str(e)}")
            return None

    def create_base_model(self):
        """
        Crée un modèle de base avec des données d'exemple.
        """
        try:
            print("🔄 Création d'un modèle de base...")
            
            # Données d'exemple pour l'entraînement
            texts = [
                "Bonjour, comment puis-je vous aider ?",
                "Quel temps fait-il aujourd'hui ?",
                "Je cherche des informations sur l'intelligence artificielle",
                "Comment faire une recette de gâteau au chocolat ?",
                "Quelles sont les dernières actualités ?",
                "Je cherche des cours de mathématiques",
                "Parlez-moi de l'histoire de France",
                "Quels sont les résultats du match de football ?"
            ]
            
            labels = [
                "accueil",
                "météo",
                "technologie",
                "cuisine",
                "actualités",
                "éducation",
                "histoire",
                "sport"
            ]
            
            # Prétraitement des textes
            processed_texts = [preprocess_text(text) for text in texts]
            
            # Création et entraînement du modèle optimisé
            self.optimize_ml_model(processed_texts, labels)
            
            print("✅ Modèle de base créé avec succès !")
            
        except Exception as e:
            print(f"❌ Erreur lors de la création du modèle de base : {str(e)}")
            raise

    def initialize_models(self):
        """
        Initialise tous les modèles nécessaires au fonctionnement du chatbot.
        Charge les modèles existants ou en crée de nouveaux si nécessaire.
        """
        try:
            print("🔄 Chargement des modèles...")
            
            # Chargement du modèle DialoGPT pour la génération de réponses
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Chargement des modèles de classification ML
            model_path = "app/model.joblib"
            vectorizer_path = "app/vectorizer.joblib"
            
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                self.classifier_model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
            else:
                print("⚠️ Modèles de classification non trouvés. Création d'un modèle de base...")
                self.create_base_model()
            
            # Chargement du modèle BERT pour la classification DL
            self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=len(self.bert_labels),
                ignore_mismatched_sizes=True
            )
            
            print("✅ Modèles chargés avec succès !")
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation des modèles: {str(e)}")
            raise

    def classify_with_bert(self, text: str) -> dict:
        """
        Classifie un texte avec BERT et retourne la catégorie prédite et la confiance.
        """
        try:
            inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            logits = outputs.logits.detach().numpy()[0]
            probs = np.exp(logits) / np.sum(np.exp(logits))
            pred_idx = np.argmax(probs)
            return {
                "category": self.bert_labels[pred_idx] if pred_idx < len(self.bert_labels) else "autre",
                "confidence": float(probs[pred_idx]),
                "embeddings": outputs.hidden_states[-1][0][0].detach().numpy().tolist() if hasattr(outputs, 'hidden_states') else None
            }
        except Exception as e:
            print(f"❌ Erreur lors de la classification BERT: {str(e)}")
            return {"category": "erreur", "confidence": 0.0, "embeddings": None}

    def search_wikipedia(self, query: str) -> str:
        """
        Recherche des informations sur Wikipedia.
        
        Args:
            query (str): La requête de recherche
            
        Returns:
            str: Les informations trouvées
        """
        try:
            return search_wikipedia(query)
        except Exception as e:
            return f"Erreur lors de la recherche: {str(e)}"

    def summarize_text(self, text: str, use_dl: bool = True) -> str:
        """
        Résume un texte.
        
        Args:
            text (str): Le texte à résumer
            use_dl (bool): Si True, utilise le modèle de deep learning
            
        Returns:
            str: Le résumé généré
        """
        try:
            return summarize_text(text, use_dl)
        except Exception as e:
            return f"Erreur lors du résumé: {str(e)}"

    @lru_cache(maxsize=100)
    def generate_response(self, text: str, use_dl: bool = False, model_type: str = "bert") -> str:
        """
        Génère une réponse en fonction du texte d'entrée.
        
        Args:
            text (str): Le texte à analyser
            use_dl (bool): Si True, utilise le modèle de deep learning
            model_type (str): Type de modèle DL ("bert", "rnn" ou "keras")
            
        Returns:
            str: La réponse générée
        """
        try:
            if use_dl:
                if model_type == "bert":
                    sentiment = self.sentiment_analyzer(text)[0]
                    return f"Sentiment: {sentiment['label']} (Confiance: {sentiment['score']:.2f})"
                elif model_type == "rnn":
                    predictions, probs = self.classifier_rnn.predict([text])
                    sentiment = "POSITIVE" if predictions[0] == 1 else "NEGATIVE"
                    confidence = probs[0][predictions[0]]
                    return f"Sentiment: {sentiment} (Confiance: {confidence:.2f})"
                else:  # keras
                    predictions, probs = self.classifier_keras.predict([text])
                    sentiment = "POSITIVE" if predictions[0] == 1 else "NEGATIVE"
                    confidence = probs[0][predictions[0]]
                    return f"Sentiment: {sentiment} (Confiance: {confidence:.2f})"
            else:
                sentiment = self.classifier_ml.predict([text])[0]
                return f"Sentiment: {sentiment}"
        except Exception as e:
            return f"Erreur lors de l'analyse: {str(e)}"

    def cleanup(self):
        """
        Nettoie les ressources utilisées par le chatbot.
        Libère la mémoire GPU si disponible.
        """
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if self.classifier_model:
            del self.classifier_model
        if self.vectorizer:
            del self.vectorizer
        if self.bert_tokenizer:
            del self.bert_tokenizer
        if self.bert_model:
            del self.bert_model
        if self.optimized_model:
            del self.optimized_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    # Test simple du chatbot en mode console
    chatbot = Chatbot()
    print("Chatbot initialisé ! Tapez 'quit' pour quitter.")
    
    while True:
        user_input = input("Vous : ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Chatbot : Au revoir !")
            chatbot.cleanup()
            break
            
        response = chatbot.generate_response(user_input)
        print("Chatbot :", response)
