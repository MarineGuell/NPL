import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils import DataLoader, TextPreprocessor
from models import AutoencoderSummarizer
from nltk.tokenize import sent_tokenize

def test_autoencoder_train_debug():
    """Test de debug pour identifier le problème dans la fonction train"""
    print("🔍 DEBUG DE LA FONCTION TRAIN DE L'AUTOENCODEUR")
    print("=" * 60)
    
    # 1. Charger les données
    dataset_path = r'D:\MES_Documents\Fac\NLP\app\data\enriched_dataset_paragraphs_2.csv'
    loader = DataLoader(dataset_path)
    texts, labels = loader.get_texts_and_labels()
    
    # 2. Prétraitement
    preprocessor = TextPreprocessor()
    autoencoder_texts = preprocessor.transform_for_autoencoder(texts)
    
    print(f"📝 {len(autoencoder_texts)} textes prétraités")
    
    # 3. Créer l'autoencodeur
    autoencoder = AutoencoderSummarizer()
    
    # 4. Debug de la fonction train
    print("\n🔧 DEBUG DE LA FONCTION TRAIN")
    print("-" * 40)
    
    # Test du tokenizer
    print("📤 Test du tokenizer...")
    if not autoencoder.tokenizer.is_fitted:
        print("🔄 Entraînement du tokenizer partagé...")
        all_sentences_for_tokenizer = []
        for text in autoencoder_texts[:100]:  # Test sur 100 textes
            try:
                sentences = sent_tokenize(text)
                all_sentences_for_tokenizer.extend(sentences)
            except Exception as e:
                print(f"   ❌ Erreur tokenisation: {e}")
        
        print(f"   Phrases pour tokenizer: {len(all_sentences_for_tokenizer)}")
        
        if len(all_sentences_for_tokenizer) > 0:
            autoencoder.tokenizer.fit_on_texts(all_sentences_for_tokenizer)
            print(f"✅ Tokenizer entraîné sur {len(all_sentences_for_tokenizer)} phrases")
        else:
            print("❌ Aucune phrase trouvée pour entraîner le tokenizer")
            return
    else:
        print("✅ Tokenizer déjà entraîné")
    
    # Test du prétraitement des phrases
    print("\n📝 Test du prétraitement des phrases...")
    all_sentences = []
    all_sentence_vectors = []
    
    for i, text in enumerate(autoencoder_texts[:10]):  # Test sur 10 textes
        print(f"   Texte {i+1}:")
        print(f"     Longueur: {len(text)} caractères")
        
        try:
            sentence_vectors, original_sentences = autoencoder.preprocess_sentences(text)
            print(f"     Phrases originales: {len(original_sentences)}")
            print(f"     Phrases vectorisées: {len(sentence_vectors)}")
            
            # Debug de la condition problématique
            if len(sentence_vectors) > 0:
                print(f"     ✅ Condition len(sentence_vectors) > 0: VRAI")
                all_sentences.extend(original_sentences)
                all_sentence_vectors.extend(sentence_vectors)
            else:
                print(f"     ❌ Condition len(sentence_vectors) > 0: FAUX")
                
        except Exception as e:
            print(f"     ❌ Erreur lors du prétraitement: {e}")
    
    print(f"\n📊 RÉSULTATS DU DEBUG:")
    print(f"   Total phrases collectées: {len(all_sentences)}")
    print(f"   Total vecteurs collectés: {len(all_sentence_vectors)}")
    
    if len(all_sentences) >= 10:
        print("✅ Suffisamment de phrases pour l'entraînement")
        print("   Le problème n'est pas dans le prétraitement")
    else:
        print("❌ Pas assez de phrases pour l'entraînement")
        print("   Le problème est dans le prétraitement")

def test_text_preprocessing_difference():
    """Test pour vérifier la différence entre les prétraitements"""
    print("\n🔍 TEST DE LA DIFFÉRENCE ENTRE PRÉTRAITEMENTS")
    print("=" * 60)
    
    # 1. Charger les données
    dataset_path = r'D:\MES_Documents\Fac\NLP\app\data\enriched_dataset_paragraphs_2.csv'
    loader = DataLoader(dataset_path)
    texts, labels = loader.get_texts_and_labels()
    
    # 2. Prétraitement normal
    preprocessor = TextPreprocessor()
    processed_texts = preprocessor.transform(texts)
    
    # 3. Prétraitement pour autoencodeur
    autoencoder_texts = preprocessor.transform_for_autoencoder(texts)
    
    print(f"📝 {len(texts)} textes originaux")
    print(f"📝 {len(processed_texts)} textes prétraités normaux")
    print(f"📝 {len(autoencoder_texts)} textes prétraités pour autoencodeur")
    
    # Comparer les premiers textes
    print("\n📋 COMPARAISON DES PRÉTRAITEMENTS:")
    for i in range(3):
        print(f"\n📄 Texte {i+1}:")
        print(f"   Original: {texts.iloc[i][:100]}...")
        print(f"   Normal: {processed_texts[i][:100]}...")
        print(f"   Autoencoder: {autoencoder_texts[i][:100]}...")
        
        # Test de tokenisation
        try:
            sentences_normal = sent_tokenize(processed_texts[i])
            sentences_autoencoder = sent_tokenize(autoencoder_texts[i])
            print(f"   Phrases (normal): {len(sentences_normal)}")
            print(f"   Phrases (autoencoder): {len(sentences_autoencoder)}")
        except Exception as e:
            print(f"   ❌ Erreur tokenisation: {e}")

def test_autoencoder_train_exact():
    """Test qui reproduit exactement la fonction train de l'autoencodeur"""
    print("\n🔍 TEST EXACT DE LA FONCTION TRAIN")
    print("=" * 60)
    
    # 1. Charger les données comme dans le script d'entraînement
    dataset_path = r'D:\MES_Documents\Fac\NLP\app\data\enriched_dataset_paragraphs_2.csv'
    loader = DataLoader(dataset_path)
    texts, labels = loader.get_texts_and_labels()
    
    print(f"📝 {len(texts)} textes originaux chargés")
    
    # 2. Prétraitement spécial pour l'autoencodeur (comme dans train_autoencoder)
    print("\n🔧 Prétraitement spécial pour l'autoencodeur...")
    preprocessor = TextPreprocessor()
    autoencoder_texts = preprocessor.transform_for_autoencoder(texts)
    
    print(f"📝 {len(autoencoder_texts)} textes prétraités pour l'autoencodeur")
    
    # 3. Créer l'autoencodeur
    autoencoder = AutoencoderSummarizer()
    
    # 4. Reproduire exactement la fonction train
    print("\n🔄 Reproduction de la fonction train...")
    print("🔄 Préparation des données pour l'autoencodeur...")
    
    # Entraîner le tokenizer partagé si nécessaire
    if not autoencoder.tokenizer.is_fitted:
        print("🔄 Entraînement du tokenizer partagé...")
        all_sentences_for_tokenizer = []
        for text in autoencoder_texts:
            sentences = sent_tokenize(text)
            all_sentences_for_tokenizer.extend(sentences)
        
        print(f"   Phrases pour tokenizer: {len(all_sentences_for_tokenizer)}")
        
        if len(all_sentences_for_tokenizer) > 0:
            autoencoder.tokenizer.fit_on_texts(all_sentences_for_tokenizer)
            print(f"✅ Tokenizer entraîné sur {len(all_sentences_for_tokenizer)} phrases")
        else:
            print("❌ Aucune phrase trouvée pour entraîner le tokenizer")
            return
    else:
        print("✅ Tokenizer déjà entraîné")
    
    # Collecter toutes les phrases
    all_sentences = []
    all_sentence_vectors = []
    
    print(f"\n📝 Traitement de {len(autoencoder_texts)} textes...")
    
    for i, text in enumerate(autoencoder_texts):
        if i % 1000 == 0:  # Afficher le progrès
            print(f"   Traitement du texte {i+1}/{len(autoencoder_texts)}")
        
        sentence_vectors, original_sentences = autoencoder.preprocess_sentences(text)
        
        # Debug de la condition problématique
        if len(sentence_vectors) > 0:
            all_sentences.extend(original_sentences)
            all_sentence_vectors.extend(sentence_vectors)
    
    print(f"\n📊 RÉSULTATS FINAUX:")
    print(f"   Phrases trouvées: {len(all_sentences)}")
    print(f"   Textes analysés: {len(autoencoder_texts)}")
    
    if len(all_sentences) < 10:
        print("⚠️ Pas assez de phrases pour entraîner l'autoencodeur")
        return
    else:
        print("✅ Suffisamment de phrases pour l'entraînement")
        print(f"   Forme des données: {len(all_sentence_vectors)} phrases")

if __name__ == "__main__":
    test_autoencoder_train_debug()
    test_text_preprocessing_difference()
    test_autoencoder_train_exact() 