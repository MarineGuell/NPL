"""
Script de nettoyage des modèles - Supprime les anciens fichiers et vérifie les nouveaux chemins.

Ce script :
1. Supprime les anciens fichiers de modèles du dossier app/
2. Vérifie que les nouveaux fichiers existent dans models/
3. Teste le chargement des modèles depuis les nouveaux chemins
"""

import os
import shutil
from pathlib import Path

def cleanup_old_models():
    """
    Supprime les anciens fichiers de modèles du dossier app/
    """
    print("🧹 Nettoyage des anciens fichiers de modèles...")
    
    # Fichiers à supprimer du dossier app/
    files_to_remove = [
        "app/model_ml.joblib",
        "app/model_dl.h5", 
        "app/model.joblib",
        "app/vectorizer.joblib",
        "app/autoencoder_summarizer.h5"
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"✅ Supprimé : {file_path}")
            except Exception as e:
                print(f"❌ Erreur lors de la suppression de {file_path}: {e}")
        else:
            print(f"ℹ️ Fichier non trouvé : {file_path}")

def verify_new_models():
    """
    Vérifie que les nouveaux fichiers existent dans models/
    """
    print("\n🔍 Vérification des nouveaux fichiers de modèles...")
    
    # Fichiers attendus dans models/
    expected_files = [
        "models/ml_model.joblib",
        "models/dl_model.h5",
        "models/vectorizer.joblib",
        "models/tokenizer.pkl",
        "models/encoder.pkl"
    ]
    
    all_exist = True
    for file_path in expected_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ Fichier manquant : {file_path}")
            all_exist = False
    
    return all_exist

def test_model_loading():
    """
    Teste le chargement des modèles depuis les nouveaux chemins
    """
    print("\n🧪 Test de chargement des modèles...")
    
    try:
        # Test du modèle ML
        import joblib
        ml_model = joblib.load("models/ml_model.joblib")
        print("✅ Modèle ML chargé avec succès")
        
        # Test du modèle DL
        from tensorflow.keras.models import load_model
        dl_model = load_model("models/dl_model.h5")
        print("✅ Modèle DL chargé avec succès")
        
        # Test du vectorizer
        vectorizer = joblib.load("models/vectorizer.joblib")
        print("✅ Vectorizer chargé avec succès")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test de chargement : {e}")
        return False

def main():
    """
    Fonction principale de nettoyage
    """
    print("🚀 Début du nettoyage des modèles...")
    
    # 1. Nettoyage des anciens fichiers
    cleanup_old_models()
    
    # 2. Vérification des nouveaux fichiers
    models_exist = verify_new_models()
    
    # 3. Test de chargement
    if models_exist:
        loading_ok = test_model_loading()
        if loading_ok:
            print("\n🎉 Nettoyage terminé avec succès !")
            print("📁 Tous les modèles sont maintenant dans le dossier models/")
            print("🔧 Les chemins dans le code ont été corrigés")
        else:
            print("\n⚠️ Problème lors du test de chargement")
    else:
        print("\n⚠️ Certains fichiers de modèles sont manquants")

if __name__ == "__main__":
    main() 