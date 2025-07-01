# 🐸 Kaeru FrogBot

## Fonctionnalités Principales

### 1. Classification de Texte
- **Classification ML** : Pipeline TF-IDF + Naive Bayes
- **Classification DL** : Réseau LSTM bidirectionnel avec BatchNormalization

### 2. Résumé de Texte
- **Résumé ML** : Méthode basée sur la similarité cosinus TF-IDF
- **Résumé DL** : Autoencodeur extractif (sélection des phrases les mieux reconstruites)

### 3. Recherche Wikipedia

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


### Installation
```bash
# Cloner le repository
git clone https://github.com/MarineGuell/NPL.git
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


------------------------------------------------------------

Ce projet a été développé dans le cadre du cours de Natural Language Processing enseigné par Nicolas Miotto, à l'école Ynov Toulouse

Il a était dévelloppé avec l'aide de l'IA générative Claude

------------------------------------------------------------
