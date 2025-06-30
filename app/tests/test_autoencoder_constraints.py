#!/usr/bin/env python3
"""
Test pour analyser les contraintes de l'autoencodeur sur la longueur des textes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AutoencoderSummarizer
from nltk.tokenize import sent_tokenize
import nltk

def test_autoencoder_constraints():
    """Test des contraintes de l'autoencodeur"""
    print("🔍 Test des contraintes de l'autoencodeur")
    print("=" * 50)
    
    # Téléchargement automatique de punkt si nécessaire
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("📥 Téléchargement automatique de punkt...")
        nltk.download('punkt', quiet=True)
    
    autoencoder = AutoencoderSummarizer()
    
    # Test 1: Textes de différentes longueurs
    test_texts = [
        ("Texte très court", "Une seule phrase."),
        ("Texte court", "Première phrase. Deuxième phrase."),
        ("Texte moyen", "Première phrase. Deuxième phrase. Troisième phrase. Quatrième phrase."),
        ("Texte long", "Première phrase. Deuxième phrase. Troisième phrase. Quatrième phrase. Cinquième phrase. Sixième phrase. Septième phrase. Huitième phrase. Neuvième phrase. Dixième phrase."),
        ("Texte très long", "Première phrase. Deuxième phrase. Troisième phrase. Quatrième phrase. Cinquième phrase. Sixième phrase. Septième phrase. Huitième phrase. Neuvième phrase. Dixième phrase. Onzième phrase. Douzième phrase. Treizième phrase. Quatorzième phrase. Quinzième phrase.")
    ]
    
    print("\n1. ANALYSE DES CONTRAINTES DE LONGUEUR")
    print("-" * 40)
    
    for name, text in test_texts:
        print(f"\n📄 Test: {name}")
        print(f"📝 Texte: {text[:50]}...")
        
        # Analyse du texte
        sentences = sent_tokenize(text)
        print(f"📊 Nombre de phrases: {len(sentences)}")
        
        # Test de préprocessing
        sentence_vectors, original_sentences = autoencoder.preprocess_sentences(text)
        print(f"🔧 Phrases vectorisées: {len(sentence_vectors)}")
        
        # Vérification des contraintes
        if len(sentences) < 2:
            print("⚠️  Contrainte: Moins de 2 phrases - pas de résumé possible")
        elif len(sentence_vectors) == 0:
            print("⚠️  Contrainte: Aucune phrase vectorisée")
        else:
            print("✅ Texte valide pour l'autoencodeur")
    
    # Test 2: Contraintes d'entraînement
    print("\n2. CONTRAINTES D'ENTRAÎNEMENT")
    print("-" * 40)
    
    print("📋 Contraintes identifiées dans le code:")
    print("   • Minimum 10 phrases pour l'entraînement (ligne 800)")
    print("   • Minimum 2 phrases par texte pour le préprocessing (ligne 720)")
    print("   • Longueur maximale des phrases: 50 tokens (ligne 683)")
    print("   • Taille du vocabulaire: 5000 mots (ligne 683)")
    
    # Test 3: Simulation d'entraînement
    print("\n3. SIMULATION D'ENTRAÎNEMENT")
    print("-" * 40)
    
    # Créer des textes de test avec différentes longueurs
    short_texts = ["Phrase unique."] * 5
    medium_texts = ["Première phrase. Deuxième phrase."] * 5
    long_texts = ["Première phrase. Deuxième phrase. Troisième phrase. Quatrième phrase. Cinquième phrase."] * 5
    
    print("📊 Test avec textes courts (1 phrase chacun):")
    total_sentences = sum(len(sent_tokenize(text)) for text in short_texts)
    print(f"   Total phrases: {total_sentences}")
    print(f"   Résultat: {'❌ ÉCHEC' if total_sentences < 10 else '✅ SUCCÈS'}")
    
    print("📊 Test avec textes moyens (2 phrases chacun):")
    total_sentences = sum(len(sent_tokenize(text)) for text in medium_texts)
    print(f"   Total phrases: {total_sentences}")
    print(f"   Résultat: {'❌ ÉCHEC' if total_sentences < 10 else '✅ SUCCÈS'}")
    
    print("📊 Test avec textes longs (5 phrases chacun):")
    total_sentences = sum(len(sent_tokenize(text)) for text in long_texts)
    print(f"   Total phrases: {total_sentences}")
    print(f"   Résultat: {'❌ ÉCHEC' if total_sentences < 10 else '✅ SUCCÈS'}")
    
    # Test 4: Recommandations
    print("\n4. RECOMMANDATIONS")
    print("-" * 40)
    
    print("🎯 Pour un bon apprentissage de l'autoencodeur:")
    print("   • Minimum 10 phrases au total dans le dataset")
    print("   • Idéalement 2-5 phrases par texte")
    print("   • Textes de longueur moyenne (pas trop courts, pas trop longs)")
    print("   • Diversité dans le contenu des phrases")
    print("   • Ponctuation correcte pour la tokenisation")
    
    print("\n⚠️  Problèmes potentiels:")
    print("   • Textes trop courts (1 phrase) → pas de résumé possible")
    print("   • Textes sans ponctuation → pas de tokenisation en phrases")
    print("   • Dataset trop petit (< 10 phrases) → échec d'entraînement")
    print("   • Phrases trop longues (> 50 tokens) → troncature")

def test_autoencoder_usage():
    """Test de l'utilisation de l'autoencodeur"""
    print("\n5. TEST D'UTILISATION")
    print("-" * 40)
    
    autoencoder = AutoencoderSummarizer()
    
    # Test avec différents types de textes
    test_cases = [
        ("Texte court valide", "Première phrase. Deuxième phrase. Troisième phrase."),
        ("Texte avec ponctuation incorrecte", "Première phrase Deuxième phrase Troisième phrase"),
        ("Texte très court", "Une seule phrase."),
        ("Texte long", "Première phrase. Deuxième phrase. Troisième phrase. Quatrième phrase. Cinquième phrase. Sixième phrase. Septième phrase. Huitième phrase.")
    ]
    
    for name, text in test_cases:
        print(f"\n📄 Test: {name}")
        print(f"📝 Texte: {text}")
        
        try:
            # Test de résumé (si le modèle est entraîné)
            if autoencoder.model is not None:
                summary = autoencoder.summarize(text, num_sentences=2)
                print(f"📋 Résumé: {summary}")
            else:
                print("⚠️  Modèle non entraîné")
                
            # Test de préprocessing
            sentence_vectors, sentences = autoencoder.preprocess_sentences(text)
            print(f"🔧 Phrases détectées: {len(sentences)}")
            print(f"🔧 Phrases vectorisées: {len(sentence_vectors)}")
            
        except Exception as e:
            print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    test_autoencoder_constraints()
    test_autoencoder_usage() 