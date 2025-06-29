"""
Script de test pour le POS-tagging avancé.
Montre les améliorations apportées par le POS-tagging dans la lemmatisation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import TextPreprocessor

def test_pos_tagging():
    """Test du POS-tagging et de la lemmatisation améliorée."""
    
    print("🧪 TEST DU POS-TAGGING AVANCÉ")
    print("="*50)
    
    # Initialisation du prétraiteur
    preprocessor = TextPreprocessor()
    
    # Textes de test
    test_texts = [
        "The running man is better than the walking woman.",
        "Scientists are studying the effects of climate change.",
        "The beautiful flowers are blooming in the garden.",
        "Machine learning algorithms process data efficiently."
    ]
    
    print("\n📝 TEXTES DE TEST :")
    for i, text in enumerate(test_texts, 1):
        print(f"{i}. {text}")
    
    print("\n🔍 ANALYSE DÉTAILLÉE :")
    print("="*50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- TEXTE {i} ---")
        print(f"Original: {text}")
        
        # Version avec POS-tagging
        cleaned_text, pos_info = preprocessor.clean_with_pos_info(text)
        print(f"Nettoyé: {cleaned_text}")
        
        print("\nPOS-tagging détaillé:")
        print("Mot → Tag NLTK → Tag WordNet → Lemmatisé")
        print("-" * 50)
        
        for word, nltk_tag, wordnet_tag in pos_info:
            print(f"{word:15} → {nltk_tag:8} → {wordnet_tag:8}")
    
    print("\n📊 STATISTIQUES POS :")
    print("="*50)
    
    # Calcul des statistiques POS
    pos_stats = preprocessor.get_pos_statistics(test_texts)
    
    print("Répartition des parties du discours :")
    for tag, stats in pos_stats.items():
        print(f"{tag:8}: {stats['count']:3} mots ({stats['percentage']:5.1f}%)")
    
    print("\n🎯 COMPARAISON LEMMATISATION :")
    print("="*50)
    
    # Exemples de comparaison
    comparison_words = [
        ("running", "VBG"),  # Verbe au participe présent
        ("better", "JJR"),   # Adjectif comparatif
        ("studying", "VBG"), # Verbe au participe présent
        ("beautiful", "JJ"), # Adjectif
        ("efficiently", "RB") # Adverbe
    ]
    
    print("Mot → POS → Lemmatisation avec POS → Lemmatisation sans POS")
    print("-" * 70)
    
    for word, pos_tag in comparison_words:
        # Avec POS-tagging
        wordnet_pos = preprocessor.get_wordnet_pos(pos_tag)
        lemmatized_with_pos = preprocessor.lemmatizer.lemmatize(word, pos=wordnet_pos)
        
        # Sans POS-tagging (suppose verbe)
        lemmatized_without_pos = preprocessor.lemmatizer.lemmatize(word, pos="v")
        
        print(f"{word:12} → {pos_tag:4} → {lemmatized_with_pos:15} → {lemmatized_without_pos}")
    
    print("\n✅ AVANTAGES DU POS-TAGGING :")
    print("="*50)
    print("• Lemmatisation plus précise (better → good, pas better)")
    print("• Distinction entre verbes et adjectifs")
    print("• Meilleure qualité du prétraitement")
    print("• Possibilité de filtrer par types de mots")
    print("• Amélioration des performances des modèles")

def test_pipeline_complet():
    """Test du pipeline complet avec POS-tagging."""
    
    print("\n🚀 TEST DU PIPELINE COMPLET")
    print("="*50)
    
    preprocessor = TextPreprocessor()
    
    # Texte complexe
    complex_text = """
    The machine learning algorithms are efficiently processing large datasets 
    while scientists are studying the effects of climate change. The beautiful 
    results show significant improvements in accuracy and performance.
    """
    
    print(f"Texte original:\n{complex_text}")
    
    # Pipeline complet
    cleaned_text = preprocessor.clean(complex_text)
    print(f"\nTexte après prétraitement complet:\n{cleaned_text}")
    
    # Version avec informations POS
    cleaned_with_pos, pos_info = preprocessor.clean_with_pos_info(complex_text)
    
    print(f"\nAnalyse POS détaillée:")
    print("Mot → POS → Lemmatisé")
    print("-" * 30)
    
    for word, nltk_tag, wordnet_tag in pos_info[:15]:  # Afficher les 15 premiers
        print(f"{word:15} → {nltk_tag:4} → {wordnet_tag}")

if __name__ == "__main__":
    test_pos_tagging()
    test_pipeline_complet()
    
    print("\n🎉 TEST TERMINÉ !")
    print("Le POS-tagging est maintenant intégré au pipeline de prétraitement.") 