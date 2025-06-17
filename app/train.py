"""
Script d'entraînement des modèles de classification de texte.
"""

import os
import matplotlib.pyplot as plt
from chatbot import TextClassifier

def plot_training_history(history):
    """
    Affiche les courbes d'apprentissage.
    
    Args:
        history: Historique d'entraînement du modèle DL
    """
    plt.figure(figsize=(12, 4))
    
    # Courbe de précision
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Précision du modèle')
    plt.xlabel('Époque')
    plt.ylabel('Précision')
    plt.legend()
    
    # Courbe de perte
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Perte du modèle')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    """
    Fonction principale pour l'entraînement des modèles.
    """
    print("🚀 Démarrage de l'entraînement des modèles...")
    
    # Création du classificateur
    classifier = TextClassifier("data/bbc_dataset.csv")
    
    # Initialisation et entraînement des modèles
    print("\n📊 Entraînement du modèle Machine Learning...")
    classifier.initialize()
    
    # Évaluation des modèles
    print("\n📈 Évaluation des modèles...")
    classifier.evaluate_models()
    
    # Sauvegarde des modèles
    print("\n💾 Sauvegarde des modèles...")
    os.makedirs("models", exist_ok=True)
    
    # Sauvegarde du modèle ML
    import joblib
    joblib.dump(classifier.ml_model.pipeline, "models/ml_model.joblib")
    
    # Sauvegarde du modèle DL
    classifier.dl_model.model.save("models/dl_model.h5")
    
    # Sauvegarde du tokenizer
    import pickle
    with open("models/tokenizer.pkl", "wb") as f:
        pickle.dump(classifier.dl_model.tokenizer, f)
    
    # Sauvegarde de l'encodeur
    with open("models/encoder.pkl", "wb") as f:
        pickle.dump(classifier.dl_model.encoder, f)
    
    print("\n✅ Entraînement terminé !")
    print("Les modèles ont été sauvegardés dans le dossier 'models/'")
    
    # Nettoyage
    classifier.cleanup()

if __name__ == "__main__":
    main() 