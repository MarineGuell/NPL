#!/usr/bin/env python3
"""
Test pour vérifier que la correction de l'autoencodeur fonctionne.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AutoencoderSummarizer
from utils import TextPreprocessor

def test_autoencoder_fix():
    """Test de la correction de l'autoencodeur"""
    print("🔧 Test de la correction de l'autoencodeur")
    print("=" * 50)
    
    # Créer des textes de test
    test_texts = [
        "Première phrase. Deuxième phrase. Troisième phrase. Quatrième phrase.",
        "Un autre texte avec plusieurs phrases. Voici la deuxième phrase. Et la troisième.",
        "Troisième texte de test. Avec encore des phrases. Pour tester l'autoencodeur.",
        "Quatrième texte. Encore des phrases. Pour s'assurer que ça fonctionne.",
        "Cinquième texte. Avec des phrases normales. Pour l'entraînement."
    ]
    
    print(f"📝 Textes de test: {len(test_texts)}")
    
    # Prétraitement
    preprocessor = TextPreprocessor()
    processed_texts = preprocessor.transform_for_autoencoder(test_texts)
    
    print(f"📝 Textes prétraités: {len(processed_texts)}")
    
    # Test de l'autoencodeur
    autoencoder = AutoencoderSummarizer()
    
    print("\n🔄 Test de l'entraînement de l'autoencodeur...")
    try:
        autoencoder.train(processed_texts, epochs=2)  # Époques réduites pour le test
        print("✅ Entraînement réussi !")
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {e}")
        return
    
    # Test de résumé
    print("\n📋 Test de résumé...")
    test_text = "Première phrase du test. Deuxième phrase du test. Troisième phrase du test. Quatrième phrase du test."
    
    try:
        summary = autoencoder.summarize(test_text, num_sentences=2)
        print(f"✅ Résumé généré: {summary}")
    except Exception as e:
        print(f"❌ Erreur lors du résumé: {e}")

def test_with_real_data():
    """Test avec un échantillon du vrai dataset"""
    print("\n📊 Test avec un échantillon du vrai dataset")
    print("=" * 50)
    
    import pandas as pd
    
    # Charger un échantillon du dataset
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "enriched_dataset_paragraphs.csv")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset non trouvé: {dataset_path}")
        return
    
    df = pd.read_csv(dataset_path)
    sample_texts = df['text'].head(10).tolist()  # 10 premiers textes
    
    print(f"📝 Échantillon: {len(sample_texts)} textes")
    
    # Prétraitement
    preprocessor = TextPreprocessor()
    processed_texts = preprocessor.transform_for_autoencoder(sample_texts)
    
    print(f"📝 Textes prétraités: {len(processed_texts)}")
    
    # Test de l'autoencodeur
    autoencoder = AutoencoderSummarizer()
    
    print("\n🔄 Test de l'entraînement avec vrai dataset...")
    try:
        autoencoder.train(processed_texts, epochs=2)  # Époques réduites pour le test
        print("✅ Entraînement réussi avec le vrai dataset !")
        
        # Test de résumé sur un vrai texte
        test_text = sample_texts[0]
        summary = autoencoder.summarize(test_text, num_sentences=2)
        print(f"✅ Résumé d'un vrai texte: {summary[:200]}...")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {e}")

if __name__ == "__main__":
    test_autoencoder_fix()
    test_with_real_data() 