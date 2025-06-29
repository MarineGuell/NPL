"""
Script d'Évaluation Globale - Chatbot Kaeru

Évalue tous les modèles entraînés et génère des rapports de performance complets.
Génère des visualisations comparatives entre les modèles.

Usage :
    python app/evaluate_all_models.py

Visualisations générées :
- Comparaison des performances globales
- Matrices de confusion pour chaque modèle
- Courbes d'apprentissage
- Métriques par classe
- Rapport d'évaluation complet
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import warnings
warnings.filterwarnings('ignore')

# Ajout du chemin pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import DataLoader, TextPreprocessor
from models import MLModel, DLModel, AutoencoderSummarizer

class ModelEvaluator:
    """
    Évaluateur global pour tous les modèles du chatbot Kaeru.
    """
    
    def __init__(self, dataset_path):
        """
        Initialise l'évaluateur.
        
        Args:
            dataset_path (str): Chemin vers le dataset
        """
        self.dataset_path = dataset_path
        self.texts = None
        self.labels = None
        self.processed_texts = None
        self.results = {}
        
        # Chargement des données
        self._load_data()
        
        # Initialisation des modèles
        self.ml_model = MLModel()
        self.dl_model = DLModel()
        self.autoencoder = AutoencoderSummarizer()
    
    def _load_data(self):
        """Charge et prétraite les données."""
        print("📚 Chargement des données pour l'évaluation...")
        
        if not os.path.exists(self.dataset_path):
            print(f"❌ Erreur : Le fichier {self.dataset_path} n'existe pas!")
            return False
        
        # Chargement
        loader = DataLoader(self.dataset_path)
        self.texts, self.labels = loader.get_texts_and_labels()
        
        # Prétraitement
        preprocessor = TextPreprocessor()
        self.processed_texts = preprocessor.transform(self.texts)
        
        print(f"✅ {len(self.texts)} textes chargés pour {len(set(self.labels))} catégories")
        return True
    
    def evaluate_ml_model(self):
        """Évalue le modèle ML."""
        print("\n" + "="*60)
        print("🤖 ÉVALUATION DU MODÈLE ML")
        print("="*60)
        
        if self.ml_model.model is None:
            print("❌ Modèle ML non entraîné!")
            return False
        
        # Évaluation automatique (les données de test sont déjà dans le modèle)
        self.ml_model.evaluate()
        
        # Stockage des résultats
        if hasattr(self.ml_model, 'y_test') and hasattr(self.ml_model, 'y_pred'):
            accuracy = accuracy_score(self.ml_model.y_test, self.ml_model.y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.ml_model.y_test, self.ml_model.y_pred, average='weighted'
            )
            
            self.results['ML'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'model_type': 'TF-IDF + Naive Bayes'
            }
        
        return True
    
    def evaluate_dl_model(self):
        """Évalue le modèle DL."""
        print("\n" + "="*60)
        print("🧠 ÉVALUATION DU MODÈLE DL")
        print("="*60)
        
        if self.dl_model.model is None:
            print("❌ Modèle DL non entraîné!")
            return False
        
        # Évaluation automatique (les données de test sont déjà dans le modèle)
        self.dl_model.evaluate()
        
        # Stockage des résultats
        if hasattr(self.dl_model, 'y_test_classes') and hasattr(self.dl_model, 'y_pred_classes'):
            accuracy = accuracy_score(self.dl_model.y_test_classes, self.dl_model.y_pred_classes)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.dl_model.y_test_classes, self.dl_model.y_pred_classes, average='weighted'
            )
            
            self.results['DL'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'model_type': 'LSTM Bidirectionnel'
            }
        
        return True
    
    def evaluate_autoencoder(self):
        """Évalue l'autoencodeur."""
        print("\n" + "="*60)
        print("🔄 ÉVALUATION DE L'AUTOENCODEUR")
        print("="*60)
        
        if self.autoencoder.model is None:
            print("❌ Autoencodeur non entraîné!")
            return False
        
        # Évaluation automatique
        self.autoencoder.evaluate()
        
        # Stockage des résultats (métriques d'entraînement)
        if self.autoencoder.history is not None:
            final_loss = self.autoencoder.history.history['loss'][-1]
            final_val_loss = self.autoencoder.history.history['val_loss'][-1]
            final_accuracy = self.autoencoder.history.history['accuracy'][-1]
            final_val_accuracy = self.autoencoder.history.history['val_accuracy'][-1]
            
            self.results['Autoencoder'] = {
                'final_loss': final_loss,
                'final_val_loss': final_val_loss,
                'final_accuracy': final_accuracy,
                'final_val_accuracy': final_val_accuracy,
                'model_type': 'Autoencodeur Extractif'
            }
        
        return True
    
    def generate_comparison_plots(self):
        """Génère des visualisations comparatives entre les modèles."""
        if not self.results:
            print("⚠️ Aucun résultat d'évaluation disponible!")
            return
        
        print("\n" + "="*60)
        print("📊 GÉNÉRATION DES VISUALISATIONS COMPARATIVES")
        print("="*60)
        
        # Création du dossier plots
        os.makedirs('app/plots', exist_ok=True)
        
        # 1. Comparaison des métriques de classification
        if 'ML' in self.results and 'DL' in self.results:
            self._plot_classification_comparison()
        
        # 2. Comparaison des courbes d'apprentissage
        self._plot_learning_curves_comparison()
        
        # 3. Résumé des performances
        self._plot_performance_summary()
        
        # 4. Rapport d'évaluation complet
        self._generate_evaluation_report()
    
    def _plot_classification_comparison(self):
        """Génère une comparaison des métriques de classification."""
        models = ['ML', 'DL']
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            colors = ['skyblue', 'lightcoral']
            
            bars = axes[i].bar(models, values, color=colors, alpha=0.7)
            axes[i].set_title(f'{metric.capitalize()} par Modèle', fontweight='bold')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_ylim(0, 1)
            
            # Ajout des valeurs sur les barres
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('app/plots/model_comparison_classification.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Comparaison classification générée")
    
    def _plot_learning_curves_comparison(self):
        """Génère une comparaison des courbes d'apprentissage."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Modèle DL
        if hasattr(self.dl_model, 'history') and self.dl_model.history is not None:
            # Accuracy DL
            axes[0,0].plot(self.dl_model.history.history['accuracy'], 'b-', linewidth=2, label='Entraînement')
            axes[0,0].plot(self.dl_model.history.history['val_accuracy'], 'r-', linewidth=2, label='Validation')
            axes[0,0].set_title('Accuracy - Modèle DL', fontweight='bold')
            axes[0,0].set_ylabel('Accuracy')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # Loss DL
            axes[0,1].plot(self.dl_model.history.history['loss'], 'b-', linewidth=2, label='Entraînement')
            axes[0,1].plot(self.dl_model.history.history['val_loss'], 'r-', linewidth=2, label='Validation')
            axes[0,1].set_title('Loss - Modèle DL', fontweight='bold')
            axes[0,1].set_ylabel('Loss')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # Autoencodeur
        if hasattr(self.autoencoder, 'history') and self.autoencoder.history is not None:
            # Accuracy Autoencodeur
            axes[1,0].plot(self.autoencoder.history.history['accuracy'], 'g-', linewidth=2, label='Entraînement')
            axes[1,0].plot(self.autoencoder.history.history['val_accuracy'], 'orange', linewidth=2, label='Validation')
            axes[1,0].set_title('Accuracy - Autoencodeur', fontweight='bold')
            axes[1,0].set_xlabel('Époques')
            axes[1,0].set_ylabel('Accuracy')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            
            # Loss Autoencodeur
            axes[1,1].plot(self.autoencoder.history.history['loss'], 'g-', linewidth=2, label='Entraînement')
            axes[1,1].plot(self.autoencoder.history.history['val_loss'], 'orange', linewidth=2, label='Validation')
            axes[1,1].set_title('Loss - Autoencodeur', fontweight='bold')
            axes[1,1].set_xlabel('Époques')
            axes[1,1].set_ylabel('Loss')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('app/plots/learning_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Comparaison courbes d'apprentissage générée")
    
    def _plot_performance_summary(self):
        """Génère un résumé des performances."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Données pour le résumé
        models = []
        accuracies = []
        colors = []
        
        if 'ML' in self.results:
            models.append('ML\n(TF-IDF + NB)')
            accuracies.append(self.results['ML']['accuracy'])
            colors.append('skyblue')
        
        if 'DL' in self.results:
            models.append('DL\n(LSTM)')
            accuracies.append(self.results['DL']['accuracy'])
            colors.append('lightcoral')
        
        if models:
            bars = ax.bar(models, accuracies, color=colors, alpha=0.7)
            ax.set_title('Résumé des Performances - Accuracy', fontsize=16, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_ylim(0, 1)
            
            # Ajout des valeurs sur les barres
            for bar, value in zip(bars, accuracies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Ligne de référence à 0.8
            ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Seuil de performance (0.8)')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('app/plots/performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Résumé des performances généré")
    
    def _generate_evaluation_report(self):
        """Génère un rapport d'évaluation complet."""
        report_path = 'app/plots/evaluation_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RAPPORT D'ÉVALUATION - CHATBOT KAERU\n")
            f.write("="*80 + "\n")
            f.write(f"Date d'évaluation : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset : {len(self.texts)} textes, {len(set(self.labels))} catégories\n")
            f.write("="*80 + "\n\n")
            
            # Résultats par modèle
            for model_name, results in self.results.items():
                f.write(f"MODÈLE : {model_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Type : {results.get('model_type', 'N/A')}\n")
                
                if 'accuracy' in results:
                    f.write(f"Accuracy : {results['accuracy']:.4f}\n")
                    f.write(f"Precision : {results['precision']:.4f}\n")
                    f.write(f"Recall : {results['recall']:.4f}\n")
                    f.write(f"F1-Score : {results['f1']:.4f}\n")
                
                if 'final_loss' in results:
                    f.write(f"Loss finale (entraînement) : {results['final_loss']:.4f}\n")
                    f.write(f"Loss finale (validation) : {results['final_val_loss']:.4f}\n")
                    f.write(f"Accuracy finale (entraînement) : {results['final_accuracy']:.4f}\n")
                    f.write(f"Accuracy finale (validation) : {results['final_val_accuracy']:.4f}\n")
                
                f.write("\n")
            
            # Recommandations
            f.write("RECOMMANDATIONS\n")
            f.write("-" * 40 + "\n")
            
            if 'ML' in self.results and 'DL' in self.results:
                ml_acc = self.results['ML']['accuracy']
                dl_acc = self.results['DL']['accuracy']
                
                if ml_acc > dl_acc:
                    f.write(f"✅ Le modèle ML ({ml_acc:.3f}) surperforme le modèle DL ({dl_acc:.3f})\n")
                else:
                    f.write(f"✅ Le modèle DL ({dl_acc:.3f}) surperforme le modèle ML ({ml_acc:.3f})\n")
            
            best_acc = max([results.get('accuracy', 0) for results in self.results.values()])
            if best_acc > 0.9:
                f.write("🎉 Excellente performance globale (>90%)\n")
            elif best_acc > 0.8:
                f.write("✅ Bonne performance globale (>80%)\n")
            elif best_acc > 0.7:
                f.write("⚠️ Performance acceptable (>70%)\n")
            else:
                f.write("❌ Performance à améliorer (<70%)\n")
        
        print(f"✅ Rapport d'évaluation généré : {report_path}")
    
    def evaluate_all(self):
        """Évalue tous les modèles."""
        print("🚀 ÉVALUATION GLOBALE DES MODÈLES")
        print("="*60)
        
        # Évaluation de chaque modèle
        self.evaluate_ml_model()
        self.evaluate_dl_model()
        self.evaluate_autoencoder()
        
        # Génération des visualisations comparatives
        self.generate_comparison_plots()
        
        print("\n" + "="*60)
        print("🎉 ÉVALUATION TERMINÉE !")
        print("="*60)
        print("📁 Visualisations générées dans app/plots/ :")
        print("   - model_comparison_classification.png")
        print("   - learning_curves_comparison.png")
        print("   - performance_summary.png")
        print("   - evaluation_report.txt")
        print("   - [Et toutes les visualisations individuelles]")

def main():
    """Fonction principale."""
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'enriched_dataset_paragraphs.csv')
    
    evaluator = ModelEvaluator(dataset_path)
    evaluator.evaluate_all()

if __name__ == "__main__":
    main() 