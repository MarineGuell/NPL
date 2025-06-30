import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DataLoader, TextPreprocessor
from utils import DataLoader, TextPreprocessor
from model_tfidf import MLModel

def main():
    print("\n=== ENTRAÎNEMENT DU MODÈLE ML ===")
    dataset_path = 'app\data\enriched_dataset_paragraphs_2.csv'
    print(f"📁 Dataset utilisé : {dataset_path}")
    loader = DataLoader(dataset_path)
    texts, labels = loader.get_texts_and_labels()
    print(f"📝 Textes : {len(texts)} | Catégories : {len(set(labels))}")

    preprocessor = TextPreprocessor()
    processed_texts = preprocessor.transform(texts)
    print(f"✅ Textes prétraités : {len(processed_texts)}")

    ml_model = MLModel()
    ml_model.train(processed_texts, labels)
    print("✅ Modèle ML entraîné et sauvegardé dans models/ml_model.joblib")
    ml_model.evaluate()
    print("📊 Métriques générées dans app/performances/")

    return ml_model

if __name__ == "__main__":
    main() 