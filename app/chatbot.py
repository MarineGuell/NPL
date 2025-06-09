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
        Effectue une recherche Wikipedia et retourne un résumé.
        Args:
            query (str): La requête de recherche
        Returns:
            str: Le résumé de l'article Wikipedia
        """
        try:
            result = search_wikipedia(query)
            # Ajout du support LaTeX pour les formules mathématiques
            result = result.replace("$", "$$")
            return result
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
                # Utilise BART (DL) avec gestion des erreurs
                try:
                    return summarize_text(text)
                except Exception as e:
                    print(f"❌ Erreur BART, passage en mode ML: {str(e)}")
                    use_dl = False
            
            # Méthode ML avec TF-IDF
            from nltk.tokenize import sent_tokenize
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Découpage du texte en phrases
            sentences = sent_tokenize(text)
            
            if len(sentences) <= 3:
                return text
            
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

    @lru_cache(maxsize=100)
    def generate_response(self, user_input: str, use_dl: bool = False) -> Dict[str, Any]:
        """
        Génère une réponse basée sur l'entrée de l'utilisateur.
        Combine la classification et la génération de texte.
        Args:
            user_input (str): Le message de l'utilisateur
            use_dl (bool): Si True, utilise BERT pour la classification
        Returns:
            dict: La réponse générée avec la catégorie et la confiance
        """
        try:
            # Classification du texte
            if use_dl:
                classification = self.classify_with_bert(user_input)
            else:
                # Classification ML
                processed_text = preprocess_text(user_input)
                if self.vectorizer and self.classifier_model:
                    features = self.vectorizer.transform([processed_text])
                    prediction = self.classifier_model.predict(features)[0]
                    confidence = self.classifier_model.predict_proba(features).max()
                    classification = {
                        "category": prediction,
                        "confidence": float(confidence),
                        "embeddings": None
                    }
                else:
                    raise Exception("Modèle ML non initialisé")

            # Génération de la réponse avec DialoGPT
            input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
            response_ids = self.model.generate(
                input_ids,
                max_length=200,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.7,
                early_stopping=True
            )
            response = self.tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

            return {
                "text": response,
                "category": classification["category"],
                "confidence": classification["confidence"],
                "embeddings": classification["embeddings"]
            }
        except Exception as e:
            print(f"❌ Erreur lors de la génération de la réponse: {str(e)}")
            return {
                "text": "Désolé, je n'ai pas pu traiter votre demande.",
                "category": "erreur",
                "confidence": 0.0,
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
