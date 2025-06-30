"""
Module principal du chatbot pour la classification et le résumé de texte.
"""

import os
import numpy as np
from utils import DataLoader, TextPreprocessor, encode_labels, normalize_text
from model_autoencodeur import AutoencoderSummarizer
from model_lstm import DLModel
from model_tfidf import MLModel

from transformers import pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

class TextProcessor:
    """
    Classe principale pour le traitement de texte.
    Gère l'initialisation, l'entraînement et l'utilisation des différents modèles.
    """
    
    def __init__(self, data_path="app\data\enriched_dataset_paragraphs_2.csv"):
        """
        Initialise le processeur de texte.
        
        Args:
            data_path (str): Chemin vers le fichier de données
        """
        self.data_path = data_path
        self.ml_classifier = MLModel()
        self.dl_classifier = DLModel()
        self.preprocessor = TextPreprocessor()
        self.loader = DataLoader(data_path)
        self.initialized = False
        
        # Initialisation du modèle de résumé
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Téléchargement des ressources NLTK nécessaires
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        
    def extract_keywords_lda(self, text, n_topics=3, n_words=5):
        """
        Extrait les mots-clés d'un texte en utilisant LDA.
        
        Args:
            text (str): Le texte à analyser
            n_topics (int): Nombre de topics à extraire
            n_words (int): Nombre de mots-clés par topic
            
        Returns:
            list: Liste des mots-clés extraits
        """
        # Prétraitement du texte
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # Création du vecteur de comptage
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        X = vectorizer.fit_transform([text])
        
        # Application de LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)
        
        # Extraction des mots-clés
        feature_names = vectorizer.get_feature_names_out()
        keywords = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-n_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            keywords.extend(top_words)
            
        return list(set(keywords))  # Suppression des doublons
        
    def summarize_with_keywords(self, text):
        """
        Crée un résumé basé sur les mots-clés extraits.
        
        Args:
            text (str): Le texte à résumer
            
        Returns:
            str: Le résumé avec les mots-clés
        """
        keywords = self.extract_keywords_lda(text)
        summary = f"Ce texte parle de {', '.join(keywords[:-1])} et {keywords[-1]} kero 🐸"
        return summary

    def initialize(self):
        """
        Initialise les modèles en chargeant et prétraitant les données.
        Ne réentraîne pas si les modèles sont déjà chargés depuis le disque.
        """
        if self.initialized:
            return

        # Si les modèles sont déjà chargés, on ne réentraîne pas
        if self.ml_classifier.model is not None and self.dl_classifier.model is not None:
            self.initialized = True
            return

        print("🔄 Chargement des données...")
        texts, labels = self.loader.get_texts_and_labels()
        
        print("🔄 Prétraitement des textes...")
        clean_texts = self.preprocessor.transform(texts)
        
        print("🔄 Division des données...")
        X_train, X_test, y_train, y_test = self.loader.split_data(clean_texts, labels)
        
        print("🔄 Entraînement du modèle ML...")
        self.ml_classifier.train(X_train, y_train)
        
        print("🔄 Préparation des données pour le modèle DL...")
        X_dl, y_dl = self.dl_classifier.prepare(clean_texts, labels)
        
        print("🔄 Entraînement du modèle DL...")
        history, X_test_dl, y_test_dl = self.dl_classifier.train(X_dl, y_dl)
        
        print("✅ Initialisation terminée !")
        self.initialized = True

    def process_text(self, text, task="classification", model_type="ml"):
        """
        Traite un texte selon la tâche et le type de modèle spécifiés.
        """
        print("Initialisation")
        try:
            if not self.initialized:
                self.initialize()
            # Prétraitement du texte
            clean_text = self.preprocessor.clean(text)
            normalized_text = normalize_text(clean_text)
            result = {
                "original_text": text,
                "cleaned_text": clean_text,
                "normalized_text": normalized_text
            }
            if task == "classification":
                if model_type == "ml":
                    print('Initialisation - Classification - Machine Learning')
                    prediction = self.ml_classifier.predict([normalized_text])[0]
                    probabilities = self.ml_classifier.predict_proba([normalized_text])[0]
                    result.update({
                        "task": "classification",
                        "model": "ml",
                        "prediction": prediction,
                        "confidence": float(max(probabilities)),
                        "probabilities": probabilities.tolist()
                    })
                else:
                    print('Initialisation - Classification - Deep Learning')
                    prediction = self.dl_classifier.predict([normalized_text])[0]
                    probabilities = self.dl_classifier.predict_proba([normalized_text])[0]
                    result.update({
                        "task": "classification",
                        "model": "dl",
                        "prediction": prediction,
                        "confidence": float(max(probabilities)),
                        "probabilities": probabilities.tolist()
                    })
            elif task == "summarization":
                if model_type == "ml":                    
                    print('Initialisation - Résumé - Machine Learning')
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from nltk.tokenize import sent_tokenize
                    sentences = sent_tokenize(text)
                    vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform(sentences)
                    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
                    num_sentences = min(3, len(sentences))
                    top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
                    top_indices.sort()
                    summary = " ".join([sentences[i] for i in top_indices])
                    important_words = []
                    if self.ml_classifier.vectorizer is not None:
                        clean_text = self.preprocessor.clean(text)
                        X = self.ml_classifier.vectorizer.transform([clean_text])
                        feature_names = self.ml_classifier.vectorizer.get_feature_names_out()
                        scores = X.toarray()[0]
                        top_indices_words = scores.argsort()[-5:][::-1]
                        important_words = [feature_names[i] for i in top_indices_words if scores[i] > 0]
                    result.update({
                        "task": "summarization",
                        "model": "ml",
                        "summary": summary,
                        "important_words": important_words
                    })
                else:
                    print('Initialisation - Classification - Deep Learning')
                    summary = self.summarizer(normalized_text, 
                                            max_length=130, 
                                            min_length=30, 
                                            do_sample=False)[0]['summary_text']
                    result.update({
                        "task": "summarization",
                        "model": "dl",
                        "summary": summary
                    })
            elif task == "keywords":
                summary = self.summarize_with_keywords(text)
                result.update({
                    "task": "keywords",
                    "model": "lda",
                    "summary": summary
                })
            return result
        except Exception as e:
            return {"error": f"Erreur lors du traitement : {e}"}

    def evaluate_models(self):
        """
        Évalue les modèles sur les données de test.
        """
        if not self.initialized:
            self.initialize()
        
        print("\nÉvaluation du modèle ML :")
        self.ml_classifier.evaluate(X_test, y_test)
        
        print("\nÉvaluation du modèle DL :")
        self.dl_classifier.evaluate(X_test_dl, y_test_dl)

    def cleanup(self):
        """
        Nettoie les ressources utilisées par les modèles.
        """
        if hasattr(self.dl_classifier, 'model'):
            del self.dl_classifier.model
        if hasattr(self.dl_classifier, 'tokenizer'):
            del self.dl_classifier.tokenizer
        self.initialized = False

    def classify(self, text, model_type='ml'):
        """
        Classification d'un texte avec le modèle ML ou DL.
        Retourne une chaîne formatée pour l'interface.
        """
        result = self.process_text(text, task='classification', model_type=model_type)
        label = result.get('prediction', 'N/A')
        confidence = result.get('confidence', 0)
        return f"Prédiction : {label}\nConfiance : {confidence:.2f}"

    def summarize(self, text, model_type='ml'):
        """
        Résumé d'un texte avec le modèle ML ou DL.
        Retourne une chaîne formatée pour l'interface.
        """
        result = self.process_text(text, task='summarization', model_type=model_type)
        summary = result.get('summary', '')
        if model_type == 'ml' and result.get('important_words'):
            mots = result['important_words']
            mots_str = ', '.join(mots)
            return f"Résumé : {summary}\n\nMots-clés importants : {mots_str}"
        return summary

# if __name__ == "__main__":
#     # Test simple du processeur
#     processor = TextProcessor()
#     processor.initialize()
    
#     # Exemples de textes à traiter
#     sample_texts = [
#         "Cricket Australia is set to begin the team's pre-season...",
#         "Additionally, the microsite on Amazon.in highlights...",
#         "Having undergone a surgery for shoulder dislocation..."
#     ]
    
#     print("\nClassification avec le modèle ML :")
#     for text in sample_texts:
#         result = processor.process_text(text, task="classification", model_type="ml")
#         print(f"\nTexte : {text[:50]}...")
#         print(f"Prédiction : {result['prediction']}")
#         print(f"Confiance : {result['confidence']:.2f}")
    
#     print("\nClassification avec le modèle DL :")
#     for text in sample_texts:
#         result = processor.process_text(text, task="classification", model_type="dl")
#         print(f"\nTexte : {text[:50]}...")
#         print(f"Prédiction : {result['prediction']}")
#         print(f"Confiance : {result['confidence']:.2f}")
    
#     print("\nRésumé avec le modèle ML :")
#     for text in sample_texts:
#         result = processor.process_text(text, task="summarization", model_type="ml")
#         print(f"\nTexte : {text[:50]}...")
#         print(f"Résumé : {result['summary']}")
    
#     print("\nRésumé avec le modèle DL :")
#     for text in sample_texts:
#         result = processor.process_text(text, task="summarization", model_type="dl")
#         print(f"\nTexte : {text[:50]}...")
#         print(f"Résumé : {result['summary']}")
    
#     print("\nRésumé basé sur les mots-clés :")
#     for text in sample_texts:
#         result = processor.process_text(text, task="keywords", model_type="lda")
#         print(f"\nTexte : {text[:50]}...")
#         print(f"Résumé : {result['summary']}")
    
#     processor.cleanup() 