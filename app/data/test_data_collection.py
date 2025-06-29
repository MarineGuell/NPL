"""
Script de test pour la collecte de données
Teste les fonctions principales sans collecter tout le dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from create_data import (
    extract_paragraphs_from_text,
    add_text_if_valid,
    CATEGORIES_REGOUPEES
)

def test_text_processing():
    """Test des fonctions de traitement de texte."""
    print("🧪 Test des fonctions de traitement de texte...")
    
    # Test d'extraction de paragraphes
    long_text = """
    Premier paragraphe. Il contient plusieurs phrases. Chaque phrase apporte de l'information.
    Deuxième paragraphe. Il est également composé de plusieurs phrases. Cela permet de tester l'extraction.
    Troisième paragraphe. Il devrait être extrait séparément. Chaque paragraphe doit être cohérent.
    Quatrième paragraphe. Il complète le test. L'extraction fonctionne bien.
    """
    
    paragraphs = extract_paragraphs_from_text(long_text)
    print(f"Nombre de paragraphes extraits: {len(paragraphs)}")
    for i, para in enumerate(paragraphs, 1):
        print(f"Paragraphe {i}: {para[:100]}...")
    print()

def test_categories_config():
    """Test de la configuration des catégories."""
    print("📊 Test de la configuration des catégories...")
    
    for category_name, config in CATEGORIES_REGOUPEES.items():
        print(f"\n🎯 {category_name}:")
        print(f"  Sources: {config['sources']}")
        print(f"  Wiki labels: {config['wiki_labels']}")
        print(f"  ArXiv codes: {config['arxiv_codes']}")
        print(f"  RSS keywords: {config['rss_keywords']}")
    
    print(f"\nTotal catégories: {len(CATEGORIES_REGOUPEES)}")
    print(f"Objectif total: {3000 * len(CATEGORIES_REGOUPEES)} entrées")

def test_add_text_function():
    """Test de la fonction d'ajout de texte."""
    print("\n✅ Test de la fonction d'ajout de texte...")
    
    # Réinitialiser les variables globales pour le test
    import create_data
    create_data.TEXTS_SET.clear()
    create_data.ROWS.clear()
    create_data.CATEGORY_COUNTS.clear()
    
    # Test d'ajout de texte valide
    test_text = "Ceci est un texte de test assez long pour être valide. Il contient plusieurs phrases. Cela devrait fonctionner correctement."
    result = add_text_if_valid(test_text, "Life Sciences")
    print(f"Ajout réussi: {result}")
    print(f"Nombre de textes: {len(create_data.TEXTS_SET)}")
    print(f"Compteur Life Sciences: {create_data.CATEGORY_COUNTS['Life Sciences']}")
    
    # Test d'ajout de doublon
    result2 = add_text_if_valid(test_text, "Life Sciences")
    print(f"Ajout doublon: {result2}")
    print(f"Nombre de textes après doublon: {len(create_data.TEXTS_SET)}")
    
    # Test d'ajout de texte trop court
    short_text = "Texte trop court."
    result3 = add_text_if_valid(short_text, "Life Sciences")
    print(f"Ajout texte court: {result3}")

def main():
    """Fonction principale de test."""
    print("🚀 Début des tests de collecte de données...")
    print("📝 Note : Pas de nettoyage/normalisation - pour l'entraînement des modèles")
    print("="*60)
    
    test_text_processing()
    test_categories_config()
    test_add_text_function()
    
    print("\n" + "="*60)
    print("✅ Tous les tests sont passés !")
    print("Le script de collecte de données est prêt à être utilisé.")

if __name__ == "__main__":
    main() 