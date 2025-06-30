import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils import DataLoader, TextPreprocessor
from models import AutoencoderSummarizer

def test_complete_autoencoder_pipeline():
    """Test complet du pipeline de l'autoencodeur"""
    print("🔍 TEST COMPLET DU PIPELINE AUTOENCODEUR")
    print("=" * 60)
    
    # 1. Test du chargement des données
    print("\n📚 1. TEST DU CHARGEMENT DES DONNÉES")
    print("-" * 40)
    
    dataset_path = r'D:\MES_Documents\Fac\NLP\app\data\enriched_dataset_paragraphs_2.csv'
    
    try:
        loader = DataLoader(dataset_path)
        texts, labels = loader.get_texts_and_labels()
        print(f"✅ Dataset chargé avec succès")
        print(f"   Textes: {len(texts)}")
        print(f"   Catégories: {len(set(labels))}")
        print(f"   Catégories: {list(set(labels))}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return
    
    # 2. Test du prétraitement pour l'autoencodeur
    print("\n🔧 2. TEST DU PRÉTRAITEMENT POUR L'AUTOENCODEUR")
    print("-" * 40)
    
    try:
        preprocessor = TextPreprocessor()
        autoencoder_texts = preprocessor.transform_for_autoencoder(texts)
        print(f"✅ Prétraitement réussi")
        print(f"   Textes prétraités: {len(autoencoder_texts)}")
        
        # Afficher quelques exemples
        print("\n📋 Exemples de textes prétraités:")
        for i, text in enumerate(autoencoder_texts[:3]):
            print(f"   {i+1}. {text[:100]}...")
            
    except Exception as e:
        print(f"❌ Erreur lors du prétraitement: {e}")
        return
    
    # 3. Test de la tokenisation des phrases
    print("\n🔤 3. TEST DE LA TOKENISATION DES PHRASES")
    print("-" * 40)
    
    from nltk.tokenize import sent_tokenize
    import nltk
    
    # Télécharger punkt si nécessaire
    try:
        nltk.data.find('tokenizers/punkt')
        print("✅ punkt disponible")
    except LookupError:
        print("📥 Téléchargement de punkt...")
        nltk.download('punkt', quiet=True)
        print("✅ punkt téléchargé")
    
    # Tester la tokenisation sur quelques textes
    total_sentences = 0
    texts_with_sentences = 0
    
    for i, text in enumerate(autoencoder_texts[:10]):
        try:
            sentences = sent_tokenize(text)
            if len(sentences) > 0:
                texts_with_sentences += 1
                total_sentences += len(sentences)
                if i < 3:
                    print(f"   Texte {i+1}: {len(sentences)} phrases")
        except Exception as e:
            print(f"   ❌ Erreur sur le texte {i+1}: {e}")
    
    print(f"✅ Tokenisation testée")
    print(f"   Textes avec phrases: {texts_with_sentences}/10")
    print(f"   Total phrases trouvées: {total_sentences}")
    
    # 4. Test de l'autoencodeur
    print("\n🤖 4. TEST DE L'AUTOENCODEUR")
    print("-" * 40)
    
    try:
        autoencoder = AutoencoderSummarizer()
        print("✅ Autoencodeur créé")
        
        # Test du prétraitement de l'autoencodeur sur un exemple
        test_text = autoencoder_texts[0]
        print(f"\n📝 Test sur le premier texte:")
        print(f"   Longueur: {len(test_text)} caractères")
        print(f"   Début: {test_text[:100]}...")
        
        sentence_vectors, original_sentences = autoencoder.preprocess_sentences(test_text)
        print(f"   Phrases vectorisées: {len(sentence_vectors)}")
        print(f"   Phrases originales: {len(original_sentences)}")
        
        if len(sentence_vectors) > 0:
            print(f"   Forme des vecteurs: {sentence_vectors.shape}")
            print("✅ Prétraitement de l'autoencodeur fonctionne")
        else:
            print("⚠️ Aucune phrase vectorisée (peut être normal si < 2 phrases)")
            
    except Exception as e:
        print(f"❌ Erreur lors du test de l'autoencodeur: {e}")
        return
    
    # 5. Test d'entraînement (optionnel, rapide)
    print("\n🎯 5. TEST D'ENTRAÎNEMENT RAPIDE")
    print("-" * 40)
    
    try:
        # Utiliser seulement les 100 premiers textes pour un test rapide
        test_texts = autoencoder_texts[:100]
        print(f"📝 Test d'entraînement sur {len(test_texts)} textes...")
        
        # Compter les phrases disponibles
        all_sentences = []
        for text in test_texts:
            try:
                sentences = sent_tokenize(text)
                if len(sentences) >= 2:  # Au moins 2 phrases par texte
                    all_sentences.extend(sentences)
            except:
                continue
        
        print(f"   Phrases disponibles pour l'entraînement: {len(all_sentences)}")
        
        if len(all_sentences) >= 10:
            print("✅ Suffisamment de phrases pour l'entraînement")
            print("   (Test d'entraînement complet disponible)")
        else:
            print("⚠️ Pas assez de phrases pour l'entraînement")
            print(f"   Nécessaire: 10, Disponible: {len(all_sentences)}")
            
    except Exception as e:
        print(f"❌ Erreur lors du test d'entraînement: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 TEST DU PIPELINE TERMINÉ")
    print("=" * 60)
    print("💡 Pour entraîner l'autoencodeur complet:")
    print("   python app/train_models.py --model autoencoder")

if __name__ == "__main__":
    test_complete_autoencoder_pipeline() 