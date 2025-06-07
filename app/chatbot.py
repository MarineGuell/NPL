"""
Module principal du chatbot.
Ce script implémente la logique du chatbot, incluant la classification de texte,
la génération de réponses et l'intégration avec différents modèles de langage.
"""

import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from typing import Optional, Dict, Any
import numpy as np

from utils import summarize_text, search_wikipedia

# Téléchargement des ressources NLTK nécessaires pour le traitement du texte
nltk.download('punkt')  # Pour la tokenization
nltk.download('stopwords')  # Pour la suppression des mots vides
nltk.download('wordnet')  # Pour la lemmatization

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
        self.classifier_model = None  # Modèle de classification
        self.vectorizer = None  # Vectoriseur pour la classification
        self.initialize_models()

    def create_base_model(self):
        """
        Crée un modèle de classification de base avec des données d'exemple.
        Utilisé si aucun modèle n'est trouvé lors de l'initialisation.
        """
        print("🔄 Création d'un modèle de classification de base...")
        
        # Données d'exemple pour l'entraînement initial
        texts = [
            "Bonjour, comment puis-je vous aider ?",
            "Quelle est la météo aujourd'hui ?",
            "Pouvez-vous me donner des informations sur Python ?",
            "Je cherche des recettes de cuisine",
            "Comment fonctionne l'intelligence artificielle ?",
            "Quelles sont les dernières nouvelles ?",
            "Je voudrais apprendre à programmer",
            "Pouvez-vous m'aider avec mes devoirs ?",
            "Je cherche des informations sur l'histoire",
            "Comment faire du sport à la maison ?"
        ]
        
        # Catégories correspondantes aux textes d'exemple
        categories = [
            "accueil",
            "météo",
            "technologie",
            "cuisine",
            "technologie",
            "actualités",
            "éducation",
            "éducation",
            "histoire",
            "sport"
        ]
        
        # Création et entraînement du vectoriseur TF-IDF
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(texts)
        
        # Création et entraînement du modèle de classification Naive Bayes
        self.classifier_model = MultinomialNB()
        self.classifier_model.fit(X, categories)
        
        # Sauvegarde des modèles pour une utilisation future
        os.makedirs("app", exist_ok=True)
        joblib.dump(self.classifier_model, "app/model.joblib")
        joblib.dump(self.vectorizer, "app/vectorizer.joblib")
        
        print("✅ Modèle de base créé et sauvegardé !")

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
            
            # Chargement des modèles de classification
            model_path = "app/model.joblib"
            vectorizer_path = "app/vectorizer.joblib"
            
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                self.classifier_model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
            else:
                print("⚠️ Modèles de classification non trouvés. Création d'un modèle de base...")
                self.create_base_model()
            
            print("✅ Modèles chargés avec succès !")
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation des modèles: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> str:
        """
        Prétraite le texte pour la classification.
        Effectue la tokenization, la suppression des mots vides et la lemmatization.
        
        Args:
            text (str): Le texte à prétraiter
            
        Returns:
            str: Le texte prétraité
        """
        text = text.lower()  # Conversion en minuscules
        text = text.translate(str.maketrans('', '', string.punctuation))  # Suppression de la ponctuation
        tokens = word_tokenize(text)  # Tokenization
        tokens = [word for word in tokens if word not in stopwords.words('english')]  # Suppression des mots vides
        tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens]  # Lemmatization
        return " ".join(tokens)

    def generate_response(self, user_input: str) -> Dict[str, Any]:
        """
        Génère une réponse basée sur l'entrée de l'utilisateur.
        Combine la classification et la génération de texte.
        
        Args:
            user_input (str): Le message de l'utilisateur
            
        Returns:
            Dict[str, Any]: La réponse générée avec sa catégorie et sa confiance
        """
        try:
            response = {
                "text": "",
                "category": None,
                "confidence": None
            }
            
            # Classification du texte si le modèle est disponible
            if self.classifier_model and self.vectorizer:
                clean_text = self.preprocess_text(user_input)
                vect = self.vectorizer.transform([clean_text])
                prediction = self.classifier_model.predict(vect)[0]
                confidence = self.classifier_model.predict_proba(vect).max()
                response["category"] = prediction
                response["confidence"] = float(confidence)
            
            # Génération de réponse avec DialoGPT
            input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, 
                                            return_tensors='pt')
            
            # Configuration de la génération de texte
            output = self.model.generate(
                input_ids,
                max_length=1000,  # Longueur maximale de la réponse
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,  # Évite la répétition de phrases
                do_sample=True,  # Active la génération stochastique
                top_k=100,  # Limite le nombre de tokens considérés
                top_p=0.7,  # Filtre les tokens par probabilité cumulative
                temperature=0.8  # Contrôle la créativité de la génération
            )
            
            # Décodage de la réponse générée
            response["text"] = self.tokenizer.decode(output[:, input_ids.shape[-1]:][0], 
                                                   skip_special_tokens=True)
            
            return response
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération de la réponse: {str(e)}")
            return {
                "text": "Désolé, une erreur s'est produite. Veuillez réessayer.",
                "category": None,
                "confidence": None
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
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    # Test simple du chatbot en mode console
    chatbot = Chatbot()
    print("🤖 Chatbot initialisé ! Tapez 'quit' pour quitter.")
    
    while True:
        user_input = input("Vous : ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Chatbot : Au revoir !")
            chatbot.cleanup()
            break
            
        response = chatbot.generate_response(user_input)
        print("Chatbot :", response)
