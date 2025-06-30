#!/usr/bin/env python3
"""
Test de diagnostic pour l'autoencodeur - pourquoi pas assez de phrases ?
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk

def test_dataset_analysis():
    """Analyse du dataset pour comprendre le problème"""
    print("🔍 Diagnostic du dataset pour l'autoencodeur")
    print("=" * 60)
    
    # Charger le dataset
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "enriched_dataset_paragraphs.csv")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset non trouvé: {dataset_path}")
        return
    
    print(f"📁 Chargement du dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    print(f"📊 Taille du dataset: {len(df)} lignes")
    print(f"📋 Colonnes: {list(df.columns)}")
    
    # Analyser la colonne de texte
    text_column = None
    for col in df.columns:
        if 'text' in col.lower() or 'content' in col.lower() or 'paragraph' in col.lower():
            text_column = col
            break
    
    if text_column is None:
        print("❌ Colonne de texte non trouvée")
        print(f"   Colonnes disponibles: {list(df.columns)}")
        return
    
    print(f"📝 Colonne de texte identifiée: '{text_column}'")
    
    # Analyser les textes
    texts = df[text_column].dropna()
    print(f"📝 Textes non-null: {len(texts)}")
    
    # Téléchargement automatique de punkt si nécessaire
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("📥 Téléchargement automatique de punkt...")
        nltk.download('punkt', quiet=True)
    
    # Analyser la tokenisation en phrases
    print("\n🔍 ANALYSE DE LA TOKENISATION EN PHRASES")
    print("-" * 40)
    
    total_sentences = 0
    texts_with_sentences = 0
    sentence_lengths = []
    
    for i, text in enumerate(texts[:100]):  # Analyser les 100 premiers textes
        if pd.isna(text) or text == '':
            continue
            
        sentences = sent_tokenize(str(text))
        sentence_count = len(sentences)
        
        if sentence_count > 0:
            texts_with_sentences += 1
            total_sentences += sentence_count
            sentence_lengths.extend([len(s) for s in sentences])
        
        # Afficher quelques exemples
        if i < 5:
            print(f"\n📄 Texte {i+1}:")
            print(f"   Longueur: {len(str(text))} caractères")
            print(f"   Phrases détectées: {sentence_count}")
            if sentence_count > 0:
                print(f"   Première phrase: {sentences[0][:100]}...")
            else:
                print(f"   Contenu: {str(text)[:100]}...")
    
    print(f"\n📊 STATISTIQUES:")
    print(f"   Textes avec phrases: {texts_with_sentences}/100")
    print(f"   Total phrases détectées: {total_sentences}")
    print(f"   Moyenne phrases par texte: {total_sentences/texts_with_sentences if texts_with_sentences > 0 else 0:.2f}")
    
    if sentence_lengths:
        print(f"   Longueur moyenne des phrases: {sum(sentence_lengths)/len(sentence_lengths):.1f} caractères")
        print(f"   Longueur min/max des phrases: {min(sentence_lengths)}/{max(sentence_lengths)} caractères")
    
    # Analyser la ponctuation
    print("\n🔍 ANALYSE DE LA PONCTUATION")
    print("-" * 40)
    
    punctuation_stats = {}
    for i, text in enumerate(texts[:50]):  # Analyser les 50 premiers textes
        if pd.isna(text) or text == '':
            continue
            
        text_str = str(text)
        for punct in ['.', '!', '?', ';', ':']:
            count = text_str.count(punct)
            punctuation_stats[punct] = punctuation_stats.get(punct, 0) + count
    
    print("📊 Occurrences de ponctuation:")
    for punct, count in punctuation_stats.items():
        print(f"   '{punct}': {count} occurrences")
    
    # Test avec le prétraitement de l'autoencodeur
    print("\n🔍 TEST DU PRÉTRAITEMENT AUTOENCODEUR")
    print("-" * 40)
    
    from utils import TextPreprocessor
    preprocessor = TextPreprocessor()
    
    processed_texts = []
    for i, text in enumerate(texts[:20]):  # Tester les 20 premiers textes
        if pd.isna(text) or text == '':
            continue
            
        processed_text = preprocessor.clean_for_autoencoder(str(text))
        processed_texts.append(processed_text)
        
        if i < 3:
            print(f"\n📄 Texte {i+1} (après prétraitement):")
            print(f"   Original: {str(text)[:100]}...")
            print(f"   Traité: {processed_text[:100]}...")
            
            # Tokenisation après prétraitement
            sentences_after = sent_tokenize(processed_text)
            print(f"   Phrases après prétraitement: {len(sentences_after)}")
    
    # Analyser les textes prétraités
    total_sentences_after = 0
    for text in processed_texts:
        sentences = sent_tokenize(text)
        total_sentences_after += len(sentences)
    
    print(f"\n📊 RÉSULTATS APRÈS PRÉTRAITEMENT:")
    print(f"   Total phrases après prétraitement: {total_sentences_after}")
    print(f"   Moyenne phrases par texte: {total_sentences_after/len(processed_texts):.2f}")

def test_autoencoder_preprocessing():
    """Test du prétraitement de l'autoencodeur"""
    print("\n🔍 TEST DU PRÉTRAITEMENT DE L'AUTOENCODEUR")
    print("=" * 50)
    
    from models import AutoencoderSummarizer
    
    # Créer une instance de l'autoencodeur
    autoencoder = AutoencoderSummarizer()
    
    # Test avec un texte simple
    test_text = "Première phrase. Deuxième phrase. Troisième phrase."
    print(f"📄 Test 1: '{test_text}'")
    
    sentence_vectors, original_sentences = autoencoder.preprocess_sentences(test_text)
    print(f"   Phrases originales: {len(original_sentences)}")
    print(f"   Phrases vectorisées: {len(sentence_vectors)}")
    if len(original_sentences) > 0:
        print(f"   Première phrase: {original_sentences[0]}")
    
    # Test avec un texte d'une seule phrase
    test_text2 = "Une seule phrase."
    print(f"\n📄 Test 2: '{test_text2}'")
    
    sentence_vectors2, original_sentences2 = autoencoder.preprocess_sentences(test_text2)
    print(f"   Phrases originales: {len(original_sentences2)}")
    print(f"   Phrases vectorisées: {len(sentence_vectors2)}")
    
    # Test avec un texte vide
    test_text3 = ""
    print(f"\n📄 Test 3: '{test_text3}'")
    
    sentence_vectors3, original_sentences3 = autoencoder.preprocess_sentences(test_text3)
    print(f"   Phrases originales: {len(original_sentences3)}")
    print(f"   Phrases vectorisées: {len(sentence_vectors3)}")

def test_sentence_tokenization():
    """Test de diagnostic pour la tokenisation des phrases"""
    print("🔍 DIAGNOSTIC DE LA TOKENISATION DES PHRASES")
    print("=" * 50)
    
    # Charger le dataset
    try:
        df = pd.read_csv(r'D:\MES_Documents\Fac\NLP\app\data\enriched_dataset_paragraphs_2.csv')
        print(f"✅ Dataset chargé: {len(df)} textes")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du dataset: {e}")
        return
    
    # Télécharger punkt si nécessaire
    try:
        nltk.data.find('tokenizers/punkt')
        print("✅ punkt déjà disponible")
    except LookupError:
        print("📥 Téléchargement de punkt...")
        nltk.download('punkt', quiet=True)
        print("✅ punkt téléchargé")
    
    # Tester sur quelques textes
    total_sentences = 0
    texts_with_sentences = 0
    
    for i, text in enumerate(df['text'].head(10)):
        print(f"\n📝 Texte {i+1}:")
        print(f"   Longueur: {len(text)} caractères")
        print(f"   Début: {text[:100]}...")
        
        try:
            sentences = sent_tokenize(text)
            print(f"   Phrases trouvées: {len(sentences)}")
            
            if len(sentences) > 0:
                texts_with_sentences += 1
                total_sentences += len(sentences)
                print(f"   Première phrase: {sentences[0][:50]}...")
            else:
                print(f"   ⚠️ Aucune phrase détectée!")
                
        except Exception as e:
            print(f"   ❌ Erreur lors de la tokenisation: {e}")
    
    print(f"\n📊 RÉSULTATS DU DIAGNOSTIC:")
    print(f"   Textes avec phrases: {texts_with_sentences}/10")
    print(f"   Total phrases trouvées: {total_sentences}")
    
    # Tester sur l'ensemble du dataset
    print(f"\n🔄 Test sur l'ensemble du dataset...")
    all_sentences = []
    for text in df['text']:
        try:
            sentences = sent_tokenize(text)
            all_sentences.extend(sentences)
        except Exception as e:
            continue
    
    print(f"   Total phrases dans le dataset: {len(all_sentences)}")
    print(f"   Textes analysés: {len(df)}")

if __name__ == "__main__":
    test_dataset_analysis()
    test_autoencoder_preprocessing()
    test_sentence_tokenization() 