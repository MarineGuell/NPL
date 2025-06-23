"""
Script d'entraînement des modèles ML, DL et autoencodeur pour le chatbot NLP.

Pipeline détaillé :
1. Chargement du dataset (CSV)
2. Nettoyage des données (suppression doublons/NA)
3. Séparation textes/labels
4. Initialisation de l'orchestrateur (prétraitement, modèles, autoencodeur)
5. Entraînement :
   - ML : pipeline TF-IDF + Naive Bayes optimisé
   - DL : LSTM bidirectionnel
   - Autoencodeur : entraîné sur toutes les phrases du dataset pour le résumé extractif
6. Sauvegarde des modèles
7. Prêt pour l'inférence dans l'interface

Usage :
    python train_models.py
Modifie la variable DATASET_PATH pour choisir le dataset.
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