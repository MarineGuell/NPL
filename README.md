# Chatbot Intelligent

Ce projet est un chatbot intelligent développé dans le cadre du cours de Natural Language Processing enseigné par Miotto à l'école Ynov. Le chatbot utilise des techniques avancées de traitement du langage naturel pour comprendre et répondre aux questions des utilisateurs.

## Fonctionnalités

- Classification catégorielle de texte
- Recherche sur Wikipedia
- Résumé automatique de texte
- Interface utilisateur intuitive avec Streamlit
- Monitoring des performances en temps réel
- Prétraitement du texte avancé

## Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd NPL-1
```

2. Créer un environnement virtuel :
```bash
python -m venv .venv
source .venv/bin/activate  # Sur Windows : .venv\Scripts\activate
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Lancer l'application :
```bash
cd app
python main.py
```

2. Accéder à l'interface :
- L'interface Streamlit s'ouvrira automatiquement dans votre navigateur
- Si ce n'est pas le cas, accédez à : http://localhost:8501

## Structure du Projet

```
NPL-1/
├── app/
│   ├── main.py          # Point d'entrée de l'application
│   ├── interface.py     # Interface utilisateur Streamlit
│   ├── chatbot.py       # Logique du chatbot
│   ├── monitoring.py    # Système de monitoring
│   ├── utils.py         # Fonctions utilitaires
│   ├── model.joblib     # Modèle de classification sauvegardé
│   └── vectorizer.joblib # Vectoriseur de texte sauvegardé
├── logs/                # Logs de l'application
├── requirements.txt     # Dépendances du projet
└── README.md           # Documentation
```

## Fonctionnalités Détaillées

### 1. Classification Catégorielle
- Analyse et classification automatique de textes
- Utilisation d'un modèle entraîné pour la catégorisation

### 2. Recherche Wikipedia
- Recherche intelligente dans Wikipedia
- Extraction et présentation des informations pertinentes

### 3. Résumé de Texte
- Génération automatique de résumés
- Conservation des informations essentielles

### 4. Monitoring
- Suivi des performances en temps réel
- Statistiques d'utilisation
- Métriques de performance

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
