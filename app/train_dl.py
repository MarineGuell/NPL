import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DataLoader, TextPreprocessor
from model_lstm import DLModel

def main():
    print("\n=== ENTRAÎNEMENT DU MODÈLE DL ===")
    dataset_path = r'app\data\enriched_dataset_paragraphs_2.csv'
    print(f"📁 Dataset utilisé : {dataset_path}")
    loader = DataLoader(dataset_path)
    texts, labels = loader.get_texts_and_labels()
    print(f"📝 Textes : {len(texts)} | Catégories : {len(set(labels))}")

    preprocessor = TextPreprocessor()
    processed_texts = preprocessor.transform(texts)
    print(f"✅ Textes prétraités : {len(processed_texts)}")

    dl_model = DLModel()
    X, y = dl_model.prepare(processed_texts, labels)
    dl_model.train(X, y)
    print("✅ Modèle DL entraîné et sauvegardé dans models/dl_model.h5")
    dl_model.evaluate()
    print("📊 Métriques générées dans app/performances/")

if __name__ == "__main__":
    main() 

