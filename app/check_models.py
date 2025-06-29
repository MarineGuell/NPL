"""
Script de vérification des modèles - Chatbot Kaeru

Vérifie l'état des modèles entraînés et affiche les fichiers manquants.
Utile pour diagnostiquer les problèmes d'entraînement.

Usage :
    python app/check_models.py
"""

import os
import pickle
import joblib
from tensorflow.keras.models import load_model

def check_model_files():
    """Vérifie l'existence et l'état des fichiers de modèles."""
    print("🔍 VÉRIFICATION DES MODÈLES DU CHATBOT KAERU")
    print("="*60)
    
    # Liste des fichiers requis
    required_files = [
        ("ml_model.joblib", "Modèle ML complet"),
        ("vectorizer.joblib", "Vectorizer TF-IDF"),
        ("dl_model.h5", "Modèle LSTM"),
        ("dl_label_encoder.pkl", "Label encoder DL"),
        ("autoencoder_summarizer.h5", "Autoencodeur"),
        ("shared_tokenizer.pkl", "Tokenizer partagé")
    ]
    
    models_dir = "models"
    missing_files = []
    existing_files = []
    
    print(f"📁 Vérification du dossier : {models_dir}")
    
    for filename, description in required_files:
        filepath = os.path.join(models_dir, filename)
        
        if os.path.exists(filepath):
            # Vérification de la taille du fichier
            size = os.path.getsize(filepath)
            size_mb = size / (1024 * 1024)
            
            print(f"✅ {filename} : {description} ({size_mb:.2f} MB)")
            existing_files.append(filename)
            
            # Test de chargement pour les fichiers critiques
            try:
                if filename.endswith('.joblib'):
                    model = joblib.load(filepath)
                    print(f"   🔧 Chargement réussi")
                elif filename.endswith('.h5'):
                    model = load_model(filepath)
                    print(f"   🔧 Chargement réussi")
                elif filename.endswith('.pkl'):
                    with open(filepath, 'rb') as f:
                        obj = pickle.load(f)
                    print(f"   🔧 Chargement réussi")
            except Exception as e:
                print(f"   ❌ Erreur de chargement : {e}")
                
        else:
            print(f"❌ {filename} : {description} (MANQUANT)")
            missing_files.append(filename)
    
    print(f"\n📊 RÉSUMÉ :")
    print(f"   ✅ Fichiers présents : {len(existing_files)}/6")
    print(f"   ❌ Fichiers manquants : {len(missing_files)}/6")
    
    if missing_files:
        print(f"\n🔧 FICHIERS MANQUANTS :")
        for filename in missing_files:
            print(f"   - {filename}")
        
        print(f"\n💡 POUR RÉSOUDRE :")
        print(f"   1. Entraîner tous les modèles :")
        print(f"      python app/train_models.py")
        print(f"   2. Ou entraîner modulairement :")
        print(f"      python app/train_models_modular.py --all")
        print(f"   3. Ou entraîner un modèle spécifique :")
        print(f"      python app/train_models_modular.py --ml")
        print(f"      python app/train_models_modular.py --dl")
        print(f"      python app/train_models_modular.py --autoencoder")
    else:
        print(f"\n🎉 TOUS LES MODÈLES SONT PRÊTS !")
        print(f"   Votre chatbot peut maintenant être utilisé !")
    
    return len(missing_files) == 0

def check_dataset():
    """Vérifie l'existence du dataset d'entraînement."""
    print(f"\n📚 VÉRIFICATION DU DATASET")
    print("="*60)
    
    dataset_path = os.path.join("app", "data", "enriched_dataset_paragraphs.csv")
    
    if os.path.exists(dataset_path):
        size = os.path.getsize(dataset_path)
        size_mb = size / (1024 * 1024)
        print(f"✅ Dataset trouvé : {dataset_path} ({size_mb:.2f} MB)")
        
        # Compter les lignes
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            print(f"   📊 {len(df)} entrées pour {len(df['category'].unique())} catégories")
            print(f"   📋 Catégories : {sorted(df['category'].unique())}")
        except Exception as e:
            print(f"   ❌ Erreur de lecture : {e}")
        
        return True
    else:
        print(f"❌ Dataset manquant : {dataset_path}")
        print(f"\n💡 POUR CRÉER LE DATASET :")
        print(f"   python app/data/create_data.py")
        return False

def main():
    """Fonction principale."""
    print("🚀 DIAGNOSTIC DU CHATBOT KAERU")
    print("="*60)
    
    # Vérifications
    models_ok = check_model_files()
    dataset_ok = check_dataset()
    
    print(f"\n" + "="*60)
    print("🎯 DIAGNOSTIC TERMINÉ")
    print("="*60)
    
    if models_ok and dataset_ok:
        print("✅ Votre chatbot est prêt à être utilisé !")
        print("🐸 Kero! Tout fonctionne parfaitement!")
    elif not dataset_ok:
        print("❌ Le dataset est manquant. Créez-le d'abord.")
    elif not models_ok:
        print("❌ Certains modèles sont manquants. Entraînez-les d'abord.")
    else:
        print("❌ Le dataset et les modèles sont manquants.")
        print("   Créez le dataset puis entraînez les modèles.")

if __name__ == "__main__":
    main() 