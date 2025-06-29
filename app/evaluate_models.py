"""
Script d'Évaluation Simple - Chatbot Kaeru

Évalue les modèles déjà entraînés et génère les visualisations de performance.
Ce script ne nécessite pas de réentraînement, il charge les modèles existants.

Usage :
    python app/evaluate_models.py
"""

import os
import sys
from datetime import datetime

# Ajout du chemin pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import MLModel, DLModel, AutoencoderSummarizer

def main():
    """Fonction principale."""
    print("🚀 ÉVALUATION DES MODÈLES EXISTANTS")
    print("="*60)
    print(f"⏰ Début : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Vérification des modèles disponibles
    models_available = []
    
    if os.path.exists("models/ml_model.joblib"):
        models_available.append("ML")
        print("✅ Modèle ML trouvé")
    else:
        print("❌ Modèle ML non trouvé")
    
    if os.path.exists("models/dl_model.h5"):
        models_available.append("DL")
        print("✅ Modèle DL trouvé")
    else:
        print("❌ Modèle DL non trouvé")
    
    if os.path.exists("models/autoencoder_summarizer.h5"):
        models_available.append("Autoencodeur")
        print("✅ Autoencodeur trouvé")
    else:
        print("❌ Autoencodeur non trouvé")
    
    if not models_available:
        print("\n❌ Aucun modèle trouvé!")
        print("💡 Entraînez d'abord les modèles avec :")
        print("   python app/train_models_modular.py --all")
        return
    
    print(f"\n📊 Évaluation de {len(models_available)} modèle(s) : {', '.join(models_available)}")
    
    # Évaluation des modèles disponibles
    if "ML" in models_available:
        print("\n" + "="*60)
        print("🤖 ÉVALUATION DU MODÈLE ML")
        print("="*60)
        ml_model = MLModel()
        if ml_model.model is not None:
            ml_model.evaluate()
        else:
            print("❌ Erreur lors du chargement du modèle ML")
    
    if "DL" in models_available:
        print("\n" + "="*60)
        print("🧠 ÉVALUATION DU MODÈLE DL")
        print("="*60)
        dl_model = DLModel()
        if dl_model.model is not None:
            dl_model.evaluate()
        else:
            print("❌ Erreur lors du chargement du modèle DL")
    
    if "Autoencodeur" in models_available:
        print("\n" + "="*60)
        print("🔄 ÉVALUATION DE L'AUTOENCODEUR")
        print("="*60)
        autoencoder = AutoencoderSummarizer()
        if autoencoder.model is not None:
            autoencoder.evaluate()
        else:
            print("❌ Erreur lors du chargement de l'autoencodeur")
    
    print("\n" + "="*60)
    print("🎉 ÉVALUATION TERMINÉE !")
    print("="*60)
    print(f"⏰ Fin : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📁 Visualisations générées dans app/plots/")
    print("\n💡 Pour une évaluation complète avec comparaisons :")
    print("   python app/evaluate_all_models.py")

if __name__ == "__main__":
    main() 