"""
Script d'Entraînement Global - Chatbot Kaeru

Script principal pour entraîner tous les modèles du chatbot Kaeru en une seule commande.
Pipeline d'entraînement complet et automatisé.

Pipeline d'Entraînement :
1. Chargement du dataset CSV (enriched_dataset_paragraphs.csv par défaut)
2. Nettoyage automatique (suppression doublons, valeurs manquantes)
3. Séparation textes/labels pour l'entraînement
4. Initialisation de l'orchestrateur (prétraitement + modèles)
5. Entraînement séquentiel :
   - Modèle ML : Pipeline TF-IDF + Naive Bayes optimisé par GridSearchCV
   - Modèle DL : LSTM bidirectionnel avec tokenizer partagé
   - Autoencodeur : Entraînement sur toutes les phrases du dataset avec tokenizer partagé
6. Sauvegarde automatique de tous les modèles dans models/
7. Évaluation et visualisations pour le modèle ML

Modèles Entraînés :
- ml_model.joblib : Pipeline ML complet + vectorizer
- dl_model.h5 : Modèle LSTM + tokenizer partagé
- dl_label_encoder.pkl : Label encoder pour le modèle DL
- autoencoder_summarizer.h5 : Autoencodeur + tokenizer partagé
- shared_tokenizer.pkl : Tokenizer partagé entre les modèles DL

Visualisations Générées (app/plots/) :
- ml_confusion_matrix.png : Matrice de confusion
- ml_learning_curve.png : Courbe d'apprentissage

Usage :
    python app/train_models.py

Alternative modulaire :
    python app/train_models_modular.py --all          # Tous les modèles
    python app/train_models_modular.py --ml           # Modèle ML seulement
    python app/train_models_modular.py --dl           # Modèle DL seulement
    python app/train_models_modular.py --autoencoder  # Autoencodeur seulement

Configuration :
    Modifier DATASET_PATH pour changer le dataset d'entraînement
    Tous les modèles sont prêts pour l'inférence après l'entraînement
"""

import os
from utils import DataLoader
from chatbot import ChatbotOrchestrator

# === CHEMIN DU DATASET D'ENTRAÎNEMENT ===
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data', 'enriched_dataset_paragraphs.csv')

if __name__ == "__main__":
    print("🚀 Début de l'entraînement des modèles du chatbot Kaeru...")
    print(f"📊 Dataset utilisé : {DATASET_PATH}")
    print("🔄 Tokenizer partagé entre les modèles DL pour la cohérence")
    print("💡 Pour un entraînement modulaire, utilisez : python app/train_models_modular.py --help")
    
    # Vérification de l'existence du dataset
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Erreur : Le fichier {DATASET_PATH} n'existe pas!")
        print("💡 Veuillez d'abord exécuter le script de collecte de données :")
        print("   python app/data/create_data.py")
        exit(1)
    
    # Chargement des données
    print(f"\n📚 Chargement du dataset : {DATASET_PATH}")
    loader = DataLoader(DATASET_PATH)
    texts, labels = loader.get_texts_and_labels()
    
    print(f"✅ {len(texts)} textes chargés pour {len(set(labels))} catégories")
    print(f"📋 Catégories : {sorted(set(labels))}")

    # Initialisation de l'orchestrateur (prétraitement, modèles, autoencodeur)
    orchestrator = ChatbotOrchestrator()
    print("\n=== Entraînement des modèles... ===")
    
    # Entraînement global (ML, DL, autoencodeur résumé)
    orchestrator.train_models(texts, labels)
    
    print("\n" + "="*60)
    print("✅ Entraînement terminé avec succès!")
    print("📁 Modèles sauvegardés dans le dossier models/ :")
    print("   - ml_model.joblib (classification ML)")
    print("   - vectorizer.joblib (vectorizer TF-IDF)")
    print("   - dl_model.h5 (classification DL)")
    print("   - dl_label_encoder.pkl (label encoder DL)")
    print("   - autoencoder_summarizer.h5 (résumé DL)")
    print("   - shared_tokenizer.pkl (tokenizer partagé)")
    print("\n🎯 Les modèles sont prêts à être utilisés dans l'interface Streamlit!")
    print("🐸 Kero! Votre chatbot est maintenant entraîné!") 