"""
Script de Prédiction Standalone - Chatbot Kaeru

Script indépendant pour faire des prédictions avec les modèles entraînés du chatbot Kaeru.
Permet d'utiliser les modèles sans passer par l'interface Streamlit.

Fonctionnalités :
- Chargement automatique des modèles depuis models/
- Prédiction avec modèle ML (TF-IDF + Naive Bayes) ou DL (LSTM)
- Prétraitement automatique des textes (même pipeline que l'orchestrateur)
- Affichage des prédictions et niveaux de confiance
- Exemples de textes prédéfinis pour test

Modèles Chargés :
- ml_model.joblib : Pipeline ML complet
- dl_model.h5 : Modèle LSTM bidirectionnel
- tokenizer.pkl : Tokenizer du modèle DL
- encoder.pkl : LabelEncoder du modèle DL

Pipeline de Prédiction :
1. Chargement des modèles et objets associés depuis models/
2. Prétraitement du texte (nettoyage, normalisation, stopwords, lemmatisation)
3. Transformation numérique :
   - ML : Vectorisation TF-IDF
   - DL : Tokenization + padding
4. Prédiction avec le modèle sélectionné
5. Affichage du résultat et du niveau de confiance

Usage :
    python app/predict.py

Configuration :
    Modifier les exemples de textes dans main() pour tester différents cas
    Changer use_dl=True/False pour alterner entre ML et DL
    Tous les modèles doivent être entraînés via train_models.py avant utilisation
"""

import joblib
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import TextPreprocessor

class TextPredictor:
    """
    Classe pour faire des prédictions avec les modèles entraînés.
    """
    
    def __init__(self, models_dir="models"):
        """
        Initialise le prédicteur.
        
        Args:
            models_dir (str): Dossier contenant les modèles
        """
        self.models_dir = models_dir
        self.preprocessor = TextPreprocessor()
        
        # Chargement du modèle ML
        self.ml_model = joblib.load(f"{models_dir}/ml_model.joblib")
        
        # Chargement du modèle DL
        self.dl_model = load_model(f"{models_dir}/dl_model.h5")
        
        # Chargement du tokenizer
        with open(f"{models_dir}/tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)
        
        # Chargement de l'encodeur
        with open(f"{models_dir}/encoder.pkl", "rb") as f:
            self.encoder = pickle.load(f)

    def predict(self, text, use_dl=False):
        """
        Fait une prédiction sur un texte.
        
        Args:
            text (str): Le texte à classifier
            use_dl (bool): Si True, utilise le modèle DL
            
        Returns:
            tuple: (prédiction, probabilités)
        """
        # Prétraitement du texte
        clean_text = self.preprocessor.clean(text)
        
        if use_dl:
            # Prédiction avec le modèle DL
            sequences = self.tokenizer.texts_to_sequences([clean_text])
            padded = pad_sequences(sequences, maxlen=200)
            probabilities = self.dl_model.predict(padded)[0]
            prediction = self.encoder.classes_[probabilities.argmax()]
        else:
            # Prédiction avec le modèle ML
            probabilities = self.ml_model.predict_proba([clean_text])[0]
            prediction = self.ml_model.predict([clean_text])[0]
        
        return prediction, probabilities

def main():
    """
    Fonction principale pour les prédictions.
    """
    # Création du prédicteur
    predictor = TextPredictor()
    
    # Exemples de textes à classifier
    sample_texts = [
        "Cricket Australia is set to begin the team's pre-season...",
        "Additionally, the microsite on Amazon.in highlights...",
        "Having undergone a surgery for shoulder dislocation..."
    ]
    
    print("\nPrédictions avec le modèle ML :")
    for text in sample_texts:
        pred, prob = predictor.predict(text, use_dl=False)
        print(f"\nTexte : {text[:50]}...")
        print(f"Prédiction : {pred}")
        print(f"Confiance : {max(prob):.2f}")
    
    print("\nPrédictions avec le modèle DL :")
    for text in sample_texts:
        pred, prob = predictor.predict(text, use_dl=True)
        print(f"\nTexte : {text[:50]}...")
        print(f"Prédiction : {pred}")
        print(f"Confiance : {max(prob):.2f}")

if __name__ == "__main__":
    main() 