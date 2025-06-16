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
            
            # Liste de mots vides en français
            french_stop_words = [
                'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais', 'donc', 'car', 'ni',
                'ce', 'cet', 'cette', 'ces', 'mon', 'ton', 'son', 'notre', 'votre', 'leur',
                'qui', 'que', 'quoi', 'dont', 'où', 'comment', 'pourquoi', 'quand',
                'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
                'être', 'avoir', 'faire', 'dire', 'aller', 'voir', 'venir',
                'de', 'du', 'des', 'à', 'au', 'aux', 'en', 'dans', 'sur', 'sous',
                'par', 'pour', 'avec', 'sans', 'vers', 'chez', 'entre', 'parmi'
            ]
            
            # Création du vectoriseur avec les mots vides en français
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words=french_stop_words
            )
            
            # Transformation des textes
            X = self.vectorizer.fit_transform(processed_texts)
            
            # Création et entraînement du classificateur
            self.classifier_model = MultinomialNB(alpha=0.1)
            self.classifier_model.fit(X, labels)
            
            # Sauvegarde des modèles
            model_dir = "app"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            joblib.dump(self.classifier_model, os.path.join(model_dir, "model.joblib"))
            joblib.dump(self.vectorizer, os.path.join(model_dir, "vectorizer.joblib"))
            
            print("✅ Modèle de base créé et sauvegardé avec succès !")
            
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
                print("🔄 Chargement des modèles ML existants...")
                self.classifier_model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                
                # Vérification que les modèles sont correctement chargés
                if self.classifier_model is None or self.vectorizer is None:
                    raise ValueError("Les modèles ML n'ont pas été correctement chargés")
                
                # Test du modèle avec un texte simple
                test_text = "Test d'initialisation du modèle ML"
                test_processed = preprocess_text(test_text)
                test_vector = self.vectorizer.transform([test_processed])
                test_pred = self.classifier_model.predict(test_vector)
                
                if test_pred is None:
                    raise ValueError("Le modèle ML ne produit pas de prédictions")
                
                print("✅ Modèles ML chargés avec succès !")
            else:
                print("⚠️ Modèles de classification non trouvés. Création d'un modèle de base...")
                self.create_base_model()
            
            # Chargement du modèle BERT pour la classification DL
            print("🔄 Chargement du modèle BERT...")
            self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=len(self.bert_labels),
                output_hidden_states=True
            )
            
            # Vérification que le modèle BERT est bien chargé
            if self.bert_model is None or self.bert_tokenizer is None:
                raise ValueError("Le modèle BERT n'a pas été correctement initialisé")
            
            # Test du modèle BERT avec un texte simple
            test_text = "Test d'initialisation du modèle BERT"
            test_inputs = self.bert_tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
            test_outputs = self.bert_model(**test_inputs)
            
            if test_outputs is None:
                raise ValueError("Le modèle BERT ne produit pas de sorties")
            
            print("✅ Tous les modèles ont été chargés avec succès !")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation des modèles: {str(e)}")
            raise

    def classify_with_bert(self, text: str) -> dict:
        """
        Classifie un texte avec BERT et retourne la catégorie prédite et la confiance.
        """
        try:
            if self.bert_model is None or self.bert_tokenizer is None:
                raise ValueError("Le modèle BERT n'est pas initialisé")
            
            # Prétraitement du texte
            inputs = self.bert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            
            # Désactivation du gradient pour l'inférence
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            
            # Vérification des sorties
            if outputs is None or not hasattr(outputs, 'logits'):
                raise ValueError("Le modèle BERT n'a pas produit de sorties valides")
            
            # Calcul des probabilités
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            probs = probs.detach().numpy()[0]
            
            # Obtention de la prédiction
            pred_idx = np.argmax(probs)
            confidence = float(probs[pred_idx])
            
            # Obtention des embeddings si disponibles
            embeddings = None
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                embeddings = outputs.hidden_states[-1][0][0].detach().numpy().tolist()
            
            return {
                "category": self.bert_labels[pred_idx] if pred_idx < len(self.bert_labels) else "autre",
                "confidence": confidence,
                "embeddings": embeddings
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

    def classify_text(self, text: str, use_dl: bool = False, model_type: str = "bert") -> str:
        """
        Classifie un texte en utilisant soit le modèle ML soit le modèle DL.
        
        Args:
            text (str): Le texte à classifier
            use_dl (bool): Si True, utilise le modèle de deep learning
            model_type (str): Type de modèle DL ("bert", "rnn" ou "keras")
            
        Returns:
            str: La catégorie prédite et la confiance
        """
        try:
            if use_dl:
                if model_type == "bert":
                    result = self.classify_with_bert(text)
                    return f"Catégorie prédite : {result['category']} (confiance : {result['confidence']:.2f})"
                elif model_type == "rnn":
                    result = self.classifier_rnn.predict(text)
                    return f"Catégorie prédite : {result['category']} (confiance : {result['confidence']:.2f})"
                elif model_type == "keras":
                    result = self.classifier_keras.predict(text)
                    return f"Catégorie prédite : {result['category']} (confiance : {result['confidence']:.2f})"
                else:
                    return "Type de modèle DL non supporté"
            else:
                # Utilisation du modèle ML
                processed_text = preprocess_text(text)
                if self.classifier_model and self.vectorizer:
                    X = self.vectorizer.transform([processed_text])
                    prediction = self.classifier_model.predict(X)[0]
                    probabilities = self.classifier_model.predict_proba(X)[0]
                    confidence = max(probabilities)
                    return f"Catégorie prédite : {prediction} (confiance : {confidence:.2f})"
                else:
                    return "Modèle ML non initialisé"
        except Exception as e:
            return f"Erreur lors de la classification : {str(e)}"

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
