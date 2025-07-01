"""
Module principal du chatbot Kaeru - Orchestrateur des modèles ML/DL.

Ce module centralise l'utilisation des différents modèles (ML et DL) pour la classification
et le résumé de textes. Il gère le chargement des modèles, le prétraitement des données
et l'orchestration des prédictions.
"""

import numpy as np
from transformers import pipeline
from utils import DataLoader, TextPreprocessor, normalize_text
from model_tfidf import MLModel
from model_lstm import DLModel
from model_autoencodeur import AutoencoderSummarizer

class TextProcessor:
    """
    Orchestrateur principal pour le traitement de textes avec les modèles ML/DL.
    
    Cette classe centralise l'utilisation des modèles de classification et de résumé,
    gère le prétraitement des données et fournit une interface unifiée pour les prédictions.
    """
    
    def __init__(self, data_path="app\data\enriched_dataset_paragraphs_2.csv"):
        """
        Initialise l'orchestrateur avec les modèles et les données.
        
        Args:
            data_path (str): Chemin vers le fichier de données CSV
        """
        self.loader = DataLoader(data_path)
        self.preprocessor = TextPreprocessor()
        self.ml_classifier = MLModel()
        self.dl_classifier = DLModel()
        self.autoencoder = AutoencoderSummarizer()
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.initialized = False

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
            print("Terminé")
            return result
        except Exception as e:
            return {"error": f"Erreur lors du traitement : {e}"}

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
