import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DataLoader, TextPreprocessor
from model_autoencodeur import AutoencoderSummarizer

def main():
    print("\n=== ENTRAÎNEMENT DE L'AUTOENCODEUR ===")
    dataset_path = r'app\data\enriched_dataset_paragraphs_2.csv'
    print(f"📁 Dataset utilisé : {dataset_path}")
    loader = DataLoader(dataset_path)
    texts, labels = loader.get_texts_and_labels()
    print(f"📝 Textes : {len(texts)}")

    preprocessor = TextPreprocessor()
    autoencoder_texts = preprocessor.transform_for_autoencoder(texts)
    print(f"✅ Textes prétraités pour l'autoencodeur : {len(autoencoder_texts)}")

    autoencoder = AutoencoderSummarizer()
    autoencoder.train(autoencoder_texts)
    print("✅ Autoencodeur entraîné et sauvegardé dans models/autoencoder_summarizer.h5")
    autoencoder.evaluate()
    print("📊 Métriques générées dans app/performances/")

if __name__ == "__main__":
    main() 