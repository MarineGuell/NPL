"""
Script de test pour la recherche Wikipedia intelligente.

Teste l'extraction de mots-clés et la recherche Wikipedia
avec les modèles ML/DL entraînés.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wikipedia_search import WikipediaIntelligentSearch

def test_wikipedia_search():
    """
    Test complet de la recherche Wikipedia intelligente.
    """
    print("🐸 Test de la recherche Wikipedia intelligente de Kaeru")
    print("=" * 60)
    
    # Initialisation du système de recherche
    print("📚 Initialisation du système de recherche...")
    wiki_search = WikipediaIntelligentSearch()
    print("✅ Système initialisé avec succès")
    print()
    
    # Tests d'extraction de mots-clés
    test_queries = [
        "Je veux en savoir plus sur l'intelligence artificielle et le machine learning",
        "Parle-moi d'Albert Einstein et de la théorie de la relativité",
        "Qu'est-ce que le deep learning et les réseaux de neurones?",
        "Je cherche des informations sur la France et Paris",
        "Explique-moi la physique quantique et les atomes"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"🔍 Test {i}: '{query}'")
        print("-" * 40)
        
        # Test d'extraction de mots-clés
        print("📝 Extraction des mots-clés...")
        try:
            keywords = wiki_search.extract_keywords_advanced(query, top_k=5)
            print(f"✅ Mots-clés extraits: {len(keywords)}")
            for keyword, score in keywords:
                print(f"   • {keyword} (score: {score:.3f})")
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction: {e}")
        
        # Test de recherche Wikipedia
        print("🌐 Recherche Wikipedia...")
        try:
            search_result = wiki_search.intelligent_search(query, max_suggestions=3)
            
            if search_result['status'] == 'success':
                print(f"✅ Recherche réussie: {len(search_result['suggestions'])} suggestions")
                for suggestion in search_result['suggestions']:
                    print(f"   • {suggestion['title']} (via '{suggestion['keyword']}', confiance: {suggestion['confidence']})")
            else:
                print(f"❌ Erreur de recherche: {search_result['message']}")
                
        except Exception as e:
            print(f"❌ Erreur lors de la recherche: {e}")
        
        print()
    
    # Test spécifique d'extraction d'entités nommées
    print("🏷️ Test d'extraction d'entités nommées...")
    test_text = "Albert Einstein a développé la théorie de la relativité à Berlin en Allemagne."
    try:
        named_entities = wiki_search.extract_named_entities(test_text)
        print(f"✅ Entités nommées trouvées: {named_entities}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
    
    print()
    
    # Test de récupération de résumé
    print("📖 Test de récupération de résumé...")
    try:
        summary_result = wiki_search.get_page_summary("Artificial Intelligence", sentences=2)
        if summary_result['status'] == 'success':
            print("✅ Résumé récupéré avec succès")
            print(f"📄 Titre: {summary_result['title']}")
            print(f"📝 Résumé: {summary_result['summary'][:200]}...")
            
            if 'autoencoder_summary' in summary_result:
                print(f"🤖 Résumé IA: {summary_result['autoencoder_summary']}")
        else:
            print(f"❌ Erreur: {summary_result['message']}")
    except Exception as e:
        print(f"❌ Erreur lors de la récupération: {e}")
    
    print()
    print("🎉 Tests terminés!")

def test_keyword_extraction_methods():
    """
    Test détaillé des différentes méthodes d'extraction de mots-clés.
    """
    print("🔬 Test détaillé des méthodes d'extraction de mots-clés")
    print("=" * 60)
    
    wiki_search = WikipediaIntelligentSearch()
    test_text = "Le deep learning et l'intelligence artificielle révolutionnent l'informatique moderne."
    
    print(f"📝 Texte de test: '{test_text}'")
    print()
    
    # Test TF-IDF
    print("1️⃣ Extraction TF-IDF:")
    try:
        tfidf_keywords = wiki_search.extract_keywords_tfidf(test_text, top_k=3)
        for keyword, score in tfidf_keywords:
            print(f"   • {keyword} (TF-IDF: {score:.3f})")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    print()
    
    # Test avec confiance ML
    print("2️⃣ Extraction avec confiance ML:")
    try:
        ml_keywords = wiki_search.extract_keywords_ml_confidence(test_text, top_k=3)
        for keyword, score in ml_keywords:
            print(f"   • {keyword} (confiance ML: {score:.3f})")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    print()
    
    # Test entités nommées
    print("3️⃣ Extraction d'entités nommées:")
    try:
        named_entities = wiki_search.extract_named_entities(test_text)
        print(f"   • Entités trouvées: {named_entities}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    print()
    
    # Test extraction avancée
    print("4️⃣ Extraction avancée (combinaison):")
    try:
        advanced_keywords = wiki_search.extract_keywords_advanced(test_text, top_k=5)
        for keyword, score in advanced_keywords:
            print(f"   • {keyword} (score final: {score:.3f})")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

if __name__ == "__main__":
    print("🚀 Démarrage des tests de recherche Wikipedia intelligente")
    print()
    
    # Test des méthodes d'extraction
    test_keyword_extraction_methods()
    print("\n" + "="*60 + "\n")
    
    # Test complet
    test_wikipedia_search() 