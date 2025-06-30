"""
Script Principal d'Entraînement - Chatbot Kaeru

Script principal pour entraîner tous les modèles du chatbot Kaeru.
Permet d'entraîner tous les modèles ou un modèle spécifique.
Sauvegarde automatique des modèles dans models/.
Évaluation automatique des performances.

Usage :
    # Entraîner tous les modèles (recommandé)
    python app/train_models.py --all
    
    # Entraîner un modèle spécifique
    python app/train_models.py --model ml
    python app/train_models.py --model dl
    python app/train_models.py --model autoencoder
    
    # Entraîner et évaluer automatiquement
    python app/train_models.py --all --evaluate
    
    # Afficher l'aide
    python app/train_models.py --help

Pipeline d'Entraînement :
1. Chargement du dataset CSV (enriched_dataset_paragraphs.csv par défaut)
2. Nettoyage automatique (suppression doublons, valeurs manquantes)
3. Prétraitement des textes (nettoyage, normalisation, POS-tagging)
4. Entraînement séquentiel :
   - Modèle ML : Pipeline TF-IDF + Naive Bayes optimisé par GridSearchCV
   - Modèle DL : LSTM bidirectionnel avec tokenizer partagé
   - Autoencodeur : Entraînement sur toutes les phrases du dataset
5. Sauvegarde automatique de tous les modèles dans models/
6. Évaluation et visualisations automatiques

Modèles Entraînés :
- ml_model.joblib : Pipeline ML complet + vectorizer
- dl_model.h5 : Modèle LSTM + tokenizer partagé
- dl_label_encoder.pkl : Label encoder pour le modèle DL
- autoencoder_summarizer.h5 : Autoencodeur + tokenizer partagé
- shared_tokenizer.pkl : Tokenizer partagé entre les modèles DL
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# Ajout du chemin pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import DataLoader, TextPreprocessor
from models import MLModel, DLModel, AutoencoderSummarizer

def train_ml_model(texts, labels):
    """
    Entraîne le modèle ML.
    
    Args:
        texts: Les textes d'entraînement
        labels: Les labels d'entraînement
        
    Returns:
        MLModel: Le modèle entraîné
    """
    print("\n" + "="*60)
    print("🤖 ENTRAÎNEMENT DU MODÈLE ML")
    print("="*60)
    
    ml_model = MLModel()
    ml_model.train(texts, labels)
    
    return ml_model

def train_dl_model(texts, labels):
    """
    Entraîne le modèle DL.
    
    Args:
        texts: Les textes d'entraînement
        labels: Les labels d'entraînement
        
    Returns:
        DLModel: Le modèle entraîné
    """
    print("\n" + "="*60)
    print("🧠 ENTRAÎNEMENT DU MODÈLE DL")
    print("="*60)
    
    dl_model = DLModel()
    X, y = dl_model.prepare(texts, labels)
    history, X_test, y_test = dl_model.train(X, y)
    
    return dl_model

def train_autoencoder(texts, labels):
    """
    Entraîne l'autoencodeur.
    
    Args:
        texts: Les textes d'entraînement
        labels: Les labels d'entraînement (non utilisés pour l'autoencodeur)
        
    Returns:
        AutoencoderSummarizer: L'autoencodeur entraîné
    """
    print("\n" + "="*60)
    print("🔄 ENTRAÎNEMENT DE L'AUTOENCODEUR")
    print("="*60)
    
    # Afficher le chemin absolu du dataset
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.dataset)) if 'args' in globals() else None
    if dataset_path:
        print(f"📁 Chemin absolu du dataset utilisé : {dataset_path}")
    print(f"📝 Nombre de textes reçus : {len(texts)}")
    
    # Prétraitement spécial pour l'autoencodeur (préserve les phrases)
    print("🔧 Prétraitement spécial pour l'autoencodeur...")
    preprocessor = TextPreprocessor()
    autoencoder_texts = preprocessor.transform_for_autoencoder(texts)
    print(f"📝 {len(autoencoder_texts)} textes prétraités pour l'autoencodeur")
    
    # Compter le nombre total de phrases
    from nltk.tokenize import sent_tokenize
    total_phrases = 0
    for text in autoencoder_texts:
        total_phrases += len(sent_tokenize(text))
    print(f"📊 Nombre total de phrases trouvées dans le corpus : {total_phrases}")
    
    # Afficher quelques exemples pour vérification
    print("\n📋 Exemples de textes prétraités pour l'autoencodeur:")
    for i, text in enumerate(autoencoder_texts[:3]):
        print(f"   {i+1}. {text[:100]}...")
    
    autoencoder = AutoencoderSummarizer()
    autoencoder.train(autoencoder_texts)
    
    # Évaluation automatique de l'autoencodeur (comme pour ML et DL)
    print("\n📊 ÉVALUATION AUTOMATIQUE DE L'AUTOENCODEUR")
    print("="*60)
    autoencoder.evaluate()
    
    return autoencoder

def evaluate_models(ml_model=None, dl_model=None, autoencoder=None):
    """
    Évalue les modèles entraînés.
    
    Args:
        ml_model: Le modèle ML (optionnel)
        dl_model: Le modèle DL (optionnel)
        autoencoder: L'autoencodeur (optionnel) - déjà évalué automatiquement
    """
    print("\n" + "="*60)
    print("📊 ÉVALUATION DES MODÈLES")
    print("="*60)
    
    if ml_model:
        ml_model.evaluate()
    
    if dl_model:
        dl_model.evaluate()
    
    # L'autoencodeur est déjà évalué automatiquement après l'entraînement
    if autoencoder:
        print("✅ Autoencodeur déjà évalué automatiquement après l'entraînement")

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Script d'entraînement modulaire pour le chatbot Kaeru",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
  python app/train_models.py --all                    # Entraîner tous les modèles
  python app/train_models.py --model ml               # Entraîner seulement le modèle ML
  python app/train_models.py --model dl               # Entraîner seulement le modèle DL
  python app/train_models.py --model autoencoder      # Entraîner seulement l'autoencodeur
  python app/train_models.py --all --evaluate         # Entraîner et évaluer tous les modèles
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Entraîner tous les modèles (ML, DL, Autoencodeur)'
    )
    
    parser.add_argument(
        '--model',
        choices=['ml', 'dl', 'autoencoder'],
        help='Entraîner un modèle spécifique'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Évaluer automatiquement les modèles après entraînement'
    )
    
    parser.add_argument(
        '--dataset',
        default='data/enriched_dataset_paragraphs_2.csv',
        help='Chemin vers le dataset (défaut: data/enriched_dataset_paragraphs_2.csv)'
    )
    
    args = parser.parse_args()
    
    # Vérification des arguments
    if not args.all and not args.model:
        parser.error("Vous devez spécifier --all ou --model")
    
    if args.all and args.model:
        parser.error("Vous ne pouvez pas utiliser --all et --model en même temps")
    
    # Chemin du dataset
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.dataset))
    if not os.path.exists(dataset_path):
        print(f"❌ Erreur : Le fichier {dataset_path} n'existe pas!")
        print("💡 Assurez-vous d'avoir créé le dataset avec create_data.py")
        return
    
    print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT MODULAIRE")
    print("="*60)
    print(f"📁 Dataset : {dataset_path}")
    print(f"⏰ Début : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Chargement des données
    print("\n📚 Chargement des données...")
    loader = DataLoader(dataset_path)
    texts, labels = loader.get_texts_and_labels()
    
    # Prétraitement
    print("🔧 Prétraitement des textes...")
    preprocessor = TextPreprocessor()
    processed_texts = preprocessor.transform(texts)
    
    print(f"✅ {len(processed_texts)} textes prétraités pour {len(set(labels))} catégories")
    
    # Variables pour stocker les modèles entraînés
    ml_model = None
    dl_model = None
    autoencoder = None
    
    # Entraînement selon les arguments
    if args.all:
        print("\n🎯 ENTRAÎNEMENT DE TOUS LES MODÈLES")
        print("="*60)
        
        # Entraînement ML
        ml_model = train_ml_model(processed_texts, labels)
        
        # Entraînement DL
        dl_model = train_dl_model(processed_texts, labels)
        
        # Entraînement Autoencodeur
        autoencoder = train_autoencoder(texts, labels)
        
    elif args.model == 'ml':
        print("\n🎯 ENTRAÎNEMENT DU MODÈLE ML")
        print("="*60)
        ml_model = train_ml_model(processed_texts, labels)
        
    elif args.model == 'dl':
        print("\n🎯 ENTRAÎNEMENT DU MODÈLE DL")
        print("="*60)
        dl_model = train_dl_model(processed_texts, labels)
        
    elif args.model == 'autoencoder':
        print("\n🎯 ENTRAÎNEMENT DE L'AUTOENCODEUR")
        print("="*60)
        autoencoder = train_autoencoder(texts, labels)
    
    # Évaluation si demandée
    if args.evaluate:
        evaluate_models(ml_model, dl_model, autoencoder)
    
    print("\n" + "="*60)
    print("🎉 ENTRAÎNEMENT TERMINÉ !")
    print("="*60)
    print(f"⏰ Fin : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Résumé des modèles entraînés
    print("\n📋 RÉSUMÉ DES MODÈLES :")
    if ml_model:
        print("✅ Modèle ML (TF-IDF + Naive Bayes) - models/ml_model.joblib")
        print("   📊 Performances générées dans app/performances/")
    if dl_model:
        print("✅ Modèle DL (LSTM) - models/dl_model.h5")
        print("✅ Label Encoder DL - models/dl_label_encoder.pkl")
        print("   📊 Performances générées dans app/performances/")
    if autoencoder:
        print("✅ Autoencodeur - models/autoencoder_summarizer.h5")
        print("   📊 Performances générées dans app/performances/")
    
    print("\n💡 Pour évaluer les performances :")
    print("   python app/evaluate_all_models.py")
    print("\n💡 Pour utiliser le chatbot :")
    print("   streamlit run app/interface.py")

if __name__ == "__main__":
    main() 