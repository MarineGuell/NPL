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
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertForSequenceClassification

from utils import summarize_text, search_wikipedia, preprocess_text

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
        self.initialize_models()

    def optimize_ml_model(self, X_train, y_train):
        """
        Optimise les hyperparamètres du modèle ML (Naive Bayes) avec GridSearchCV.
        Args:
            X_train: Données d'entraînement
            y_train: Labels d'entraînement
        """
        try:
            print("🔄 Optimisation des hyperparamètres du modèle ML...")
            
            # Création du pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', MultinomialNB())
            ])
            
            # Paramètres à optimiser
            param_grid = {
                'tfidf__max_features': [5000, 10000, 15000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'tfidf__min_df': [1, 2, 3],
                'clf__alpha': [0.1, 0.5, 1.0]
            }
            
            # GridSearchCV
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            # Entraînement
            grid_search.fit(X_train, y_train)
            
            # Sauvegarde du meilleur modèle
            self.optimized_model = grid_search.best_estimator_
            self.vectorizer = self.optimized_model.named_steps['tfidf']
            self.classifier_model = self.optimized_model.named_steps['clf']
            
            print(f"✅ Optimisation terminée ! Meilleurs paramètres : {grid_search.best_params_}")
            print(f"Score de validation : {grid_search.best_score_:.3f}")
            
            # Sauvegarde du modèle optimisé
            joblib.dump(self.optimized_model, "app/optimized_model.joblib")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'optimisation du modèle : {str(e)}")
            raise

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
        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = self.bert_model(**inputs)
        logits = outputs.logits.detach().numpy()[0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        pred_idx = np.argmax(probs)
        return {
            "category": self.bert_labels[pred_idx] if pred_idx < len(self.bert_labels) else "autre",
            "confidence": float(probs[pred_idx]),
            "embeddings": outputs.hidden_states[-1][0][0].detach().numpy().tolist() if hasattr(outputs, 'hidden_states') else None
        }

    def search_wikipedia(self, query: str) -> str:
        """
        Effectue une recherche Wikipedia et retourne un résumé.
        Args:
            query (str): La requête de recherche
        Returns:
            str: Le résumé de l'article Wikipedia
        """
        try:
            return search_wikipedia(query)
        except Exception as e:
            print(f"❌ Erreur lors de la recherche Wikipedia: {str(e)}")
            return "Désolé, je n'ai pas pu trouver d'informations sur ce sujet dans Wikipedia."

    def summarize_text(self, text: str, use_dl: bool = True) -> str:
        """
        Résume un texte en utilisant soit BART (DL) soit TF-IDF (ML).
        Args:
            text (str): Le texte à résumer
            use_dl (bool): Si True, utilise BART (DL), sinon utilise TF-IDF (ML)
        Returns:
            str: Le résumé du texte
        """
        try:
            if use_dl:
                return summarize_text(text)  # Utilise BART (DL)
            else:
                # Méthode ML avec TF-IDF
                from nltk.tokenize import sent_tokenize
                import numpy as np
                from sklearn.feature_extraction.text import TfidfVectorizer
                
                # Découpage du texte en phrases
                sentences = sent_tokenize(text)
                
                # Création du vectoriseur TF-IDF
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(sentences)
                
                # Calcul des scores pour chaque phrase
                sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
                
                # Sélection des phrases les plus importantes
                num_sentences = min(3, len(sentences))  # Limite à 3 phrases
                top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
                top_indices.sort()  # Garde l'ordre original
                
                # Construction du résumé
                summary = [sentences[i] for i in top_indices]
                return " ".join(summary)
                
        except Exception as e:
            print(f"❌ Erreur lors du résumé du texte: {str(e)}")
            return "Désolé, je n'ai pas pu résumer ce texte."

    def generate_response(self, user_input: str, use_dl: bool = False) -> Dict[str, Any]:
        """
        Génère une réponse basée sur l'entrée de l'utilisateur.
        Combine la classification et la génération de texte.
        Args:
            user_input (str): Le message de l'utilisateur
            use_dl (bool): Si True, utilise BERT pour la classification
        Returns:
            Dict[str, Any]: La réponse générée avec sa catégorie et sa confiance
        """
        try:
            response = {
                "text": "",
                "category": None,
                "confidence": None,
                "embeddings": None
            }
            
            # Classification du texte
            if use_dl and self.bert_model and self.bert_tokenizer:
                bert_result = self.classify_with_bert(user_input)
                response["category"] = bert_result["category"]
                response["confidence"] = bert_result["confidence"]
                response["embeddings"] = bert_result["embeddings"]
            elif self.optimized_model:  # Utilisation du modèle optimisé si disponible
                clean_text = preprocess_text(user_input)
                prediction = self.optimized_model.predict([clean_text])[0]
                confidence = self.optimized_model.predict_proba([clean_text]).max()
                response["category"] = prediction
                response["confidence"] = float(confidence)
            elif self.classifier_model and self.vectorizer:  # Fallback sur le modèle non optimisé
                clean_text = preprocess_text(user_input)
                vect = self.vectorizer.transform([clean_text])
                prediction = self.classifier_model.predict(vect)[0]
                confidence = self.classifier_model.predict_proba(vect).max()
                response["category"] = prediction
                response["confidence"] = float(confidence)
            
            # Génération de réponse avec DialoGPT
            input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
            output = self.model.generate(
                input_ids,
                max_length=1000,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.7,
                temperature=0.8
            )
            response["text"] = self.tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération de la réponse: {str(e)}")
            return {
                "text": "Désolé, une erreur s'est produite. Veuillez réessayer.",
                "category": None,
                "confidence": None,
                "embeddings": None
            }

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
