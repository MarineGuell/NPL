import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DataLoader, TextPreprocessor
from models import AutoencoderSummarizer

def train_autoencoder_simple():
    """Entraînement simplifié de l'autoencodeur"""
    print("🚀 ENTRAÎNEMENT SIMPLIFIÉ DE L'AUTOENCODEUR")
    print("=" * 60)
    
    # 1. Charger les données
    dataset_path = r'D:\MES_Documents\Fac\NLP\app\data\enriched_dataset_paragraphs_2.csv'
    loader = DataLoader(dataset_path)
    texts, labels = loader.get_texts_and_labels()
    
    print(f"📝 {len(texts)} textes chargés")
    
    # 2. Prétraitement pour l'autoencodeur
    preprocessor = TextPreprocessor()
    autoencoder_texts = preprocessor.transform_for_autoencoder(texts)
    
    print(f"📝 {len(autoencoder_texts)} textes prétraités")
    
    # 3. Créer et entraîner l'autoencodeur
    autoencoder = AutoencoderSummarizer()
    
    print("🔄 Début de l'entraînement...")
    autoencoder.train(autoencoder_texts, epochs=5)  # Époques réduites pour le test
    
    print("✅ Entraînement terminé !")

if __name__ == "__main__":
    train_autoencoder_simple() 