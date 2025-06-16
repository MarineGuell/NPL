# Chatbot Avancé

Ce projet est un chatbot intelligent développé dans le cadre du cours de Natural Language Processing enseigné par Miotto à l'école Ynov. Le chatbot utilise des techniques avancées de traitement du langage naturel pour comprendre et répondre aux questions des utilisateurs.

## Fonctionnalités

### Classification de Texte
- Classification avec Machine Learning (Naive Bayes)
- Classification avec Deep Learning (BERT)
- Classification avec RNN (PyTorch)
- Classification avec Keras (TensorFlow)

### Traitement de Texte
- Nettoyage de texte
- Gestion des expressions régulières
- Encodage (One-Hot, TF-IDF, Word2Vec, BERT)
- Transformation (Lemmatisation, Stemming, Stop Words)

### Autres Fonctionnalités
- Recherche sur Wikipedia avec support LaTeX
- Résumé automatique de texte (BART ou TF-IDF)
- Interface utilisateur intuitive avec Streamlit
- Prétraitement du texte avancé
- API REST avec FastAPI
- Documentation interactive avec Swagger UI

## Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd NPL-1
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
venv\Scripts\activate
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Dépendances Principales

- **Deep Learning** : PyTorch, TensorFlow, Transformers
- **NLP** : NLTK, Gensim, Sentence-Transformers
- **ML** : scikit-learn, pandas, numpy
- **Web** : FastAPI, Streamlit, BeautifulSoup4
- **Autres** : emoji, contractions, wikipedia

## Utilisation

### Interface Streamlit
1. Lancer l'interface :
```bash
streamlit run app/interface.py
```

### API FastAPI
1. Lancer l'API :
```bash
cd app
python main.py
```

2. Accéder aux interfaces :
- Interface Streamlit : http://localhost:8501
- Documentation API : http://localhost:8000/docs
- Interface API : http://localhost:8000

## Structure du Projet

```
NPL-1/
├── app/
│   ├── main.py          # Point d'entrée de l'API FastAPI
│   ├── interface.py     # Interface utilisateur Streamlit
│   ├── chatbot.py       # Logique du chatbot
│   ├── utils.py         # Fonctions utilitaires
│   ├── models.py        # Modèles de classification
│   └── models/          # Dossier pour les modèles sauvegardés
├── logs/                # Logs de l'application
├── requirements.txt     # Dépendances du projet
└── README.md           # Documentation
```

## Fonctionnalités Détaillées

### 1. Classification de Texte
- **Machine Learning** : Naive Bayes pour la classification rapide
- **Deep Learning** :
  - BERT pour la classification avancée
  - RNN pour la classification séquentielle
  - Keras pour la classification personnalisable

### 2. Traitement de Texte
- **Nettoyage** : Suppression des caractères spéciaux, URLs, etc.
- **Regex** : Recherche et remplacement de patterns
- **Encodage** : Conversion du texte en vecteurs
- **Transformation** : Normalisation du texte

### 3. Recherche Wikipedia
- Recherche intelligente dans Wikipedia
- Extraction et présentation des informations pertinentes
- Support des formules mathématiques en LaTeX

### 4. Résumé de Texte
- **Deep Learning** : BART pour des résumés de haute qualité
- **Machine Learning** : TF-IDF pour des résumés rapides

### 5. API REST
- Documentation interactive avec Swagger UI
- Endpoints RESTful
- Validation des données
- Gestion des erreurs

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Licence

Ce projet est sous licence MIT.

## Auteurs

- Développé dans le cadre du cours de Natural Language Processing
- Enseignant : Miotto
- École : Ynov

🚀 Démarrage du serveur Chatbot API...
📝 Documentation disponible sur : http://localhost:8000/docs
