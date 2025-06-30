# 🐸 Kaeru FrogBot


## Fonctionnalités Principales

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

### Structure du Projet
```
NLP/
├── app/
│   ├── interface.py           # Interface Streamlit (5 fonctions)
│   ├── chatbot.py             # Orchestrateur principal
│   ├── model_tfidf.py         # Modèles Machine Learning
│   ├── model_autoencodeur.py  # Modèles Autoencodeur
│   ├── model_lstm.py          # Modèles Deep Learning
│   ├── train_ml.py            # Entrainement Modèles Machine Learning
│   ├── train_autoencodeur.py  # Entrainement Modèles Autoencodeur
│   ├── train_dl.py            # Entrainement Modèles Deep Learning
│   ├── mon_tokenizer.py       # Tokenizer
│   ├── wikipedia_search.py    # Wikipedia
│   ├── utils.py               # Prétraitement et utilitaires
│   ├── models/                # Modeles entrainés enregistrés
│   ├── performances/          # evaluation des Modeles entrainés enregistrés
│   ├── data/                  # Datasets d'entraînement
│   ├── plots/                 # Visualisations (matrices, courbes, comparaisons)
│   └── static/                # CSS
├── requirements.txt           # Dépendances
└── README.md                  # Documentation
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
# Activer l'environnement
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les ressources NLTK
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

## Utilisation

### Interface Utilisateur
```bash
# Lancer l'interface Streamlit
streamlit run app/interface.py
```

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

## 📊 entrainement, sauvegarde et performances

### Entraînement
1. Chargement du dataset CSV
2. Nettoyage (doublons, valeurs manquantes)
3. Prétraitement global (TextPreprocessor)
4. Entraînement ML (GridSearchCV + évaluation)
5. Entraînement DL (LSTM + sauvegarde tokenizer/encoder)
6. Entraînement autoencodeur (phrases du dataset)
7. Sauvegarde de tous les modèles dans `models/`
Dans le dossier performances/

### Fichiers Sauvegardés (dossier `models/`)
- `ml_model.joblib` : Pipeline ML complet (TF-IDF + Naive Bayes)
- `vectorizer.joblib` : Vectorizer TF-IDF du modèle ML
- `dl_model.h5` : Modèle LSTM bidirectionnel
- `dl_label_encoder.pkl` : LabelEncoder du modèle DL
- `autoencoder_summarizer.h5` : Autoencodeur pour le résumé
- `shared_tokenizer.pkl` : Tokenizer partagé pour les modèles DL

### Métriques Évaluées
- **Accuracy** : Précision globale de classification
- **Precision** : Précision par classe
- **Recall** : Rappel par classe
- **F1-Score** : Score F1 par classe
- **Matrice de confusion** : Visualisation des erreurs
- **Courbes d'apprentissage** : Évolution de l'entraînement

### Visualisations Individuelles
- `ml_confusion_matrix.png` : Matrice de confusion du modèle ML
- `ml_learning_curve.png` : Courbe d'apprentissage du modèle ML
- `ml_metrics_by_class.png` : Métriques par classe (ML)
- `dl_confusion_matrix.png` : Matrice de confusion du modèle DL
- `dl_learning_curves.png` : Courbes d'apprentissage du modèle DL
- `dl_metrics_by_class.png` : Métriques par classe (DL)
- `autoencoder_learning_curves.png` : Courbes d'apprentissage de l'autoencodeur
- `autoencoder_architecture.png` : Architecture de l'autoencodeur

### Visualisations Comparatives
- `model_comparison_classification.png` : Comparaison des métriques de classification
- `learning_curves_comparison.png` : Comparaison des courbes d'apprentissage
- `performance_summary.png` : Résumé des performances globales
- `evaluation_report.txt` : Rapport d'évaluation complet

------------------------------------------------------------
Ce projet a été développé dans le cadre du cours de Natural Language Processing enseigné par Nicolas Miotto, à l'école Ynov Toulouse
------------------------------------------------------------
