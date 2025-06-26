"""
Script d'Entraînement Global - Chatbot Kaeru

Script principal pour entraîner tous les modèles du chatbot Kaeru en une seule commande.
Pipeline d'entraînement complet et automatisé.

Pipeline d'Entraînement :
1. Chargement du dataset CSV (balanced_dataset_50k.csv par défaut)
2. Nettoyage automatique (suppression doublons, valeurs manquantes)
3. Séparation textes/labels pour l'entraînement
4. Initialisation de l'orchestrateur (prétraitement + modèles)
5. Entraînement séquentiel :
   - Modèle ML : Pipeline TF-IDF + Naive Bayes optimisé par GridSearchCV
   - Modèle DL : LSTM bidirectionnel avec sauvegarde tokenizer/encoder
   - Autoencodeur : Entraînement sur toutes les phrases du dataset
6. Sauvegarde automatique de tous les modèles dans models/
7. Évaluation et visualisations pour le modèle ML

Modèles Entraînés :
- ml_model.joblib : Pipeline ML complet + vectorizer
- dl_model.h5 : Modèle LSTM + tokenizer.pkl + encoder.pkl
- autoencoder_summarizer.h5 : Autoencodeur + autoencoder_tokenizer.pkl

Visualisations Générées (app/plots/) :
- ml_confusion_matrix.png : Matrice de confusion
- ml_learning_curve.png : Courbe d'apprentissage

Usage :
    python app/train_models.py

Configuration :
    Modifier DATASET_PATH pour changer le dataset d'entraînement
    Tous les modèles sont prêts pour l'inférence après l'entraînement
"""

import os
from utils import DataLoader
from chatbot import ChatbotOrchestrator

# === À MODIFIER : chemin du dataset d'entraînement ===
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data', 'balanced_dataset_50k.csv')

if __name__ == "__main__":
    print(f"Chargement du dataset : {DATASET_PATH}")
    loader = DataLoader(DATASET_PATH)
    texts, labels = loader.get_texts_and_labels()

    # Initialisation de l'orchestrateur (prétraitement, modèles, autoencodeur)
    orchestrator = ChatbotOrchestrator()
    print("\n=== Entraînement des modèles... ===")
    # Entraînement global (ML, DL, autoencodeur résumé)
    orchestrator.train_models(texts, labels)
    print("\n✅ Entraînement terminé. Les modèles sont sauvegardés et prêts à être utilisés dans l'interface Streamlit.") 