"""
Script de test pour le tokenizer partagé entre les modèles DL
Vérifie que les deux modèles utilisent le même vocabulaire
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import shared_tokenizer, DLModel, AutoencoderSummarizer, SharedTokenizer

def test_shared_tokenizer():
    """Test du tokenizer partagé."""
    print("🧪 Test du tokenizer partagé...")
    
    # Textes de test
    test_texts = [
        "L'intelligence artificielle révolutionne la technologie moderne.",
        "La biologie moléculaire étudie les mécanismes de la vie.",
        "L'astronomie explore les mystères de l'univers."
    ]
    
    # Test du tokenizer partagé
    print("🔄 Entraînement du tokenizer partagé...")
    shared_tokenizer.fit_on_texts(test_texts)
    
    # Vérification du vocabulaire
    word_index = shared_tokenizer.tokenizer.word_index
    print(f"📚 Taille du vocabulaire : {len(word_index)} mots")
    print(f"🔤 Premiers mots du vocabulaire : {list(word_index.items())[:10]}")
    
    # Test de conversion
    sequences = shared_tokenizer.texts_to_sequences(test_texts)
    print(f"📝 Séquences générées : {sequences}")
    
    return True

def test_dl_models_consistency():
    """Test de la cohérence entre les modèles DL."""
    print("\n🧠 Test de cohérence entre les modèles DL...")
    
    # Initialisation des modèles
    dl_model = DLModel()
    autoencoder = AutoencoderSummarizer()
    
    # Vérification que les deux utilisent le même tokenizer
    print(f"🔗 DLModel tokenizer : {type(dl_model.tokenizer)}")
    print(f"🔗 Autoencoder tokenizer : {type(autoencoder.tokenizer)}")
    print(f"✅ Même instance : {dl_model.tokenizer is autoencoder.tokenizer}")
    
    # Test avec les mêmes textes
    test_text = "L'intelligence artificielle et la biologie moléculaire."
    
    # Test DLModel
    if dl_model.tokenizer.is_fitted:
        dl_sequences = dl_model.tokenizer.texts_to_sequences([test_text])
        print(f"📊 DLModel séquence : {dl_sequences}")
    
    # Test Autoencoder
    if autoencoder.tokenizer.is_fitted:
        auto_sequences = autoencoder.tokenizer.texts_to_sequences([test_text])
        print(f"📊 Autoencoder séquence : {auto_sequences}")
        print(f"✅ Séquences identiques : {dl_sequences == auto_sequences}")
    
    return True

def test_save_load_tokenizer():
    """Test de sauvegarde et chargement du tokenizer."""
    print("\n💾 Test de sauvegarde/chargement du tokenizer...")
    
    # Sauvegarde
    shared_tokenizer.save_tokenizer()
    
    # Vérification que le fichier existe
    if os.path.exists(shared_tokenizer.TOKENIZER_PATH):
        print(f"✅ Tokenizer sauvegardé : {shared_tokenizer.TOKENIZER_PATH}")
        
        # Test de chargement
        new_tokenizer = SharedTokenizer()
        new_tokenizer.load_tokenizer()
        
        # Vérification de la cohérence
        test_text = "Test de cohérence du tokenizer."
        original_seq = shared_tokenizer.texts_to_sequences([test_text])
        loaded_seq = new_tokenizer.texts_to_sequences([test_text])
        
        print(f"✅ Cohérence sauvegarde/chargement : {original_seq == loaded_seq}")
    else:
        print("❌ Erreur : Le tokenizer n'a pas été sauvegardé")
        return False
    
    return True

def main():
    """Fonction principale de test."""
    print("🚀 Test du tokenizer partagé entre les modèles DL...")
    print("="*60)
    
    try:
        # Tests
        test_shared_tokenizer()
        test_dl_models_consistency()
        test_save_load_tokenizer()
        
        print("\n" + "="*60)
        print("✅ Tous les tests sont passés !")
        print("🎯 Le tokenizer partagé fonctionne correctement.")
        print("🔗 Les modèles DL utilisent maintenant le même vocabulaire.")
        
    except Exception as e:
        print(f"\n❌ Erreur lors des tests : {e}")
        return False
    
    return True

if __name__ == "__main__":
    main() 