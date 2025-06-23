"""
Script de test pour l'autoencodeur de résumé.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from models import AutoencoderSummarizer

def test_autoencoder():
    """
    Teste le fonctionnement de l'autoencodeur.
    """
    print("🧪 Test de l'autoencodeur de résumé...")
    
    # Texte de test
    test_text = """
    L'intelligence artificielle (IA) est un domaine de l'informatique qui vise à créer des systèmes capables de simuler l'intelligence humaine. 
    Les techniques d'IA incluent l'apprentissage automatique, le traitement du langage naturel et la vision par ordinateur. 
    L'apprentissage automatique permet aux machines d'apprendre à partir de données sans être explicitement programmées. 
    Le deep learning, une sous-catégorie de l'apprentissage automatique, utilise des réseaux de neurones artificiels. 
    Ces réseaux sont inspirés du fonctionnement du cerveau humain et peuvent traiter de grandes quantités de données. 
    L'IA trouve des applications dans de nombreux domaines comme la médecine, la finance et les transports. 
    Cependant, l'IA soulève également des questions éthiques et sociales importantes.
    """
    
    # Initialisation de l'autoencodeur
    autoencoder = AutoencoderSummarizer()
    
    # Test du prétraitement
    print("🔄 Test du prétraitement...")
    sentence_vectors, cleaned_sentences, original_sentences = autoencoder.preprocess_sentences(test_text)
    print(f"Nombre de phrases extraites : {len(original_sentences)}")
    print(f"Nombre de phrases nettoyées : {len(cleaned_sentences)}")
    print(f"Forme des vecteurs de phrases : {sentence_vectors.shape if len(sentence_vectors) > 0 else 'Aucun vecteur'}")
    
    # Test de l'entraînement (si pas de modèle existant)
    if autoencoder.model is None:
        print("🔄 Test de l'entraînement...")
        try:
            autoencoder.train([test_text], epochs=2, batch_size=8)
            print("✅ Entraînement réussi")
        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement : {e}")
            return
    
    # Test du résumé
    print("🔄 Test du résumé...")
    try:
        summary = autoencoder.summarize(test_text, num_sentences=3)
        print("✅ Résumé généré avec succès")
        print(f"Résumé : {summary}")
    except Exception as e:
        print(f"❌ Erreur lors du résumé : {e}")

if __name__ == "__main__":
    test_autoencoder() 