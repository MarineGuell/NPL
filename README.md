# 🐸 Kaeru Chatbot - Assistant NLP Intelligent

Un chatbot avancé de traitement du langage naturel (NLP) développé avec une personnalité de grenouille japonaise qui ponctue ses phrases par "kero". Le projet propose 5 fonctionnalités principales basées sur des modèles de Machine Learning et Deep Learning.

## 🚀 Fonctionnalités Principales

### 1. Classification de Texte
- **Classification ML** : Pipeline TF-IDF + Naive Bayes optimisé par GridSearchCV
- **Classification DL** : Réseau LSTM bidirectionnel avec BatchNormalization

### 2. Résumé de Texte
- **Résumé ML** : Méthode extractive basée sur la similarité cosinus TF-IDF
- **Résumé DL** : Autoencodeur extractif (sélection des phrases les mieux reconstruites)

### 3. Recherche Wikipedia Intelligente
- Extraction automatique des mots-clés importants
- Recherche intelligente avec gestion de l'ambiguïté
- Interface interactive pour la sélection des pages

## 🏗️ Architecture et Pipeline

### Pipeline de Données
1. **Prétraitement** : Nettoyage, normalisation, suppression des stopwords, lemmatisation
2. **Vectorisation** : TF-IDF (ML) ou Tokenization + Padding (DL)
3. **Entraînement** : Optimisation des hyperparamètres et sauvegarde automatique
4. **Inférence** : Chargement des modèles et prédiction avec formatage personnalisé

### Structure du Projet
```
NLP/
├── app/
│   ├── interface.py           # Interface Streamlit (5 fonctions)
│   ├── chatbot.py            # Orchestrateur principal
│   ├── models.py             # Modèles ML, DL, Autoencodeur
│   ├── utils.py              # Prétraitement et utilitaires
│   ├── train_models.py       # Script d'entraînement global
│   ├── train_models_modular.py # Script d'entraînement modulaire
│   ├── evaluate_all_models.py # Évaluation complète avec comparaisons
│   ├── evaluate_models.py    # Évaluation simple des modèles existants
│   ├── predict.py            # Script de prédiction standalone
│   ├── data/                 # Datasets d'entraînement
│   ├── plots/                # Visualisations (matrices, courbes, comparaisons)
│   └── static/               # CSS et ressources
├── models/                   # Modèles sauvegardés
│   ├── ml_model.joblib      # Modèle ML + vectorizer
│   ├── dl_model.h5          # Modèle DL + tokenizer + encoder
│   ├── autoencoder_summarizer.h5  # Autoencodeur résumé
│   └── shared_tokenizer.pkl  # Tokenizer partagé DL
├── requirements.txt          # Dépendances
└── README.md                # Documentation
```

## 🛠️ Installation et Configuration

### Prérequis
- Python 3.8+
- TensorFlow 2.x
- Scikit-learn
- Streamlit
- NLTK

### Installation
```bash
# Cloner le repository
git clone [URL_DU_REPO]
cd NLP

# Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les ressources NLTK
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

## 🎯 Utilisation

### Entraînement des Modèles

#### Entraînement Global (Recommandé)
```bash
# Entraîner tous les modèles sur le dataset
python app/train_models.py
```

#### Entraînement Modulaire
```bash
# Entraîner tous les modèles
python app/train_models_modular.py --all

# Entraîner un modèle spécifique
python app/train_models_modular.py --model ml
python app/train_models_modular.py --model dl
python app/train_models_modular.py --model autoencoder

# Entraîner et évaluer automatiquement
python app/train_models_modular.py --all --evaluate
```

### Évaluation des Performances

#### Évaluation Simple
```bash
# Évaluer les modèles déjà entraînés
python app/evaluate_models.py
```

#### Évaluation Complète avec Comparaisons
```bash
# Évaluation complète avec visualisations comparatives
python app/evaluate_all_models.py
```

### Interface Utilisateur
```bash
# Lancer l'interface Streamlit
streamlit run app/interface.py
```

### Prédictions Standalone
```bash
# Utiliser le script de prédiction
python app/predict.py
```

## 📊 Évaluation et Visualisations

### Métriques Évaluées
- **Accuracy** : Précision globale de classification
- **Precision** : Précision par classe
- **Recall** : Rappel par classe
- **F1-Score** : Score F1 par classe
- **Matrice de confusion** : Visualisation des erreurs
- **Courbes d'apprentissage** : Évolution de l'entraînement

### Visualisations Générées (dossier `app/plots/`)

#### Visualisations Individuelles
- `ml_confusion_matrix.png` : Matrice de confusion du modèle ML
- `ml_learning_curve.png` : Courbe d'apprentissage du modèle ML
- `ml_metrics_by_class.png` : Métriques par classe (ML)
- `dl_confusion_matrix.png` : Matrice de confusion du modèle DL
- `dl_learning_curves.png` : Courbes d'apprentissage du modèle DL
- `dl_metrics_by_class.png` : Métriques par classe (DL)
- `autoencoder_learning_curves.png` : Courbes d'apprentissage de l'autoencodeur
- `autoencoder_architecture.png` : Architecture de l'autoencodeur

#### Visualisations Comparatives
- `model_comparison_classification.png` : Comparaison des métriques de classification
- `learning_curves_comparison.png` : Comparaison des courbes d'apprentissage
- `performance_summary.png` : Résumé des performances globales
- `evaluation_report.txt` : Rapport d'évaluation complet

### Rapport d'Évaluation
Le script `evaluate_all_models.py` génère un rapport complet incluant :
- Métriques détaillées par modèle
- Comparaisons entre modèles
- Recommandations d'amélioration
- Seuils de performance

## 🔧 Détail des Fonctionnalités

### Classification ML (TF-IDF + Naive Bayes)
- **Prétraitement** : Nettoyage complet (ponctuation, URLs, emails, stopwords, lemmatisation)
- **Vectorisation** : TF-IDF avec bigrammes et optimisation des hyperparamètres
- **Modèle** : Naive Bayes optimisé par GridSearchCV
- **Évaluation** : Matrice de confusion, courbe d'apprentissage, rapport de classification

### Classification DL (LSTM Bidirectionnel)
- **Prétraitement** : Même pipeline que ML + tokenization Keras
- **Architecture** : Embedding → LSTM Bidirectionnel → BatchNormalization → Dense
- **Entraînement** : Early stopping, validation split, sauvegarde automatique
- **Inférence** : Chargement du modèle, tokenizer et encoder

### Résumé ML (Similarité Cosinus)
- **Processus** : Découpage en phrases → Vectorisation TF-IDF → Calcul similarité cosinus
- **Sélection** : 3 phrases les plus similaires au texte global
- **Préservation** : Ordre original des phrases pour la cohérence narrative

### Résumé DL (Autoencodeur Extractif)
- **Processus** : Découpage en phrases → Vectorisation → Autoencodeur → Erreur reconstruction
- **Sélection** : Phrases avec l'erreur de reconstruction la plus faible
- **Architecture** : Embedding → LSTM → Dense → RepeatVector → LSTM → TimeDistributed

### Recherche Wikipedia Intelligente
- **Extraction** : Mots-clés TF-IDF du texte utilisateur
- **Recherche** : Pages Wikipedia correspondantes
- **Gestion** : Ambiguïté avec boutons interactifs
- **Résultat** : Résumé Wikipedia formaté

## 📊 Modèles et Sauvegarde

### Fichiers Sauvegardés (dossier `models/`)
- `ml_model.joblib` : Pipeline ML complet (TF-IDF + Naive Bayes)
- `vectorizer.joblib` : Vectorizer TF-IDF du modèle ML
- `dl_model.h5` : Modèle LSTM bidirectionnel
- `dl_label_encoder.pkl` : LabelEncoder du modèle DL
- `autoencoder_summarizer.h5` : Autoencodeur pour le résumé
- `shared_tokenizer.pkl` : Tokenizer partagé pour les modèles DL

### Tokenizer Partagé
- **Avantage** : Cohérence du vocabulaire entre modèles DL
- **Sauvegarde** : Automatique dans `models/shared_tokenizer.pkl`
- **Chargement** : Automatique lors de l'initialisation des modèles

## 🎨 Interface Utilisateur

### Personnalité du Chatbot
- **Personnage** : Grenouille japonaise (Kaeru)
- **Style** : Messages ponctués par "kero" avec actions descriptives
- **Confiance** : Réponses adaptées selon le niveau de confiance du modèle

### Fonctions Disponibles
1. **Classification (Machine Learning)** : Prédiction rapide avec TF-IDF
2. **Classification (Deep Learning)** : Prédiction avancée avec LSTM
3. **Summarization (Machine Learning)** : Résumé extractif TF-IDF
4. **Summarization (Deep Learning)** : Résumé extractif autoencodeur
5. **Wikipedia Search** : Recherche intelligente avec gestion d'ambiguïté

## 🔄 Pipeline de Données Complet

### Entraînement
1. Chargement du dataset CSV
2. Nettoyage (doublons, valeurs manquantes)
3. Prétraitement global (TextPreprocessor)
4. Entraînement ML (GridSearchCV + évaluation)
5. Entraînement DL (LSTM + sauvegarde tokenizer/encoder)
6. Entraînement autoencodeur (phrases du dataset)
7. Sauvegarde de tous les modèles dans `models/`

### Inférence
1. Réception du texte utilisateur
2. Prétraitement adapté selon la fonction
3. Transformation numérique (vectorisation/tokenization)
4. Prédiction avec le modèle approprié
5. Formatage de la réponse avec personnalité

## 🐛 Dépannage

### Problèmes Courants
- **Modèles non entraînés** : Exécuter `python app/train_models.py`
- **Erreurs NLTK** : Vérifier le téléchargement des ressources
- **Fichiers manquants** : Vérifier la présence des modèles dans `models/`

### Logs et Debug
- Les scripts affichent des messages détaillés avec emojis
- Les erreurs sont capturées et affichées de manière conviviale
- Les modèles sont automatiquement rechargés s'ils existent

## 📝 Contribution

Ce projet est développé dans le cadre du cours de Natural Language Processing.
- **Enseignant** : Nicolas Miotto
- **École** : Ynov
- **Technologies** : Python, TensorFlow, Scikit-learn, Streamlit, NLTK

## 📄 Licence

Ce projet est sous licence MIT.

---

**" *hops excitedly* 🐸 Ready to help you with NLP tasks, kero!"**
