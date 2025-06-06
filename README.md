# Chatbot Intelligent

Ce projet est un chatbot intelligent développé dans le cadre du cours de Natural Language Processing. Le chatbot utilise des techniques avancées de traitement du langage naturel pour comprendre et répondre aux questions des utilisateurs.

## Fonctionnalités

- Interface API REST avec FastAPI
- Modèle de langage basé sur DialoGPT
- Capacité de recherche sur Wikipedia
- Résumé automatique de texte
- Prétraitement du texte

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

1. Lancer le serveur :
```bash
cd app
python main.py
```
2. Lancer l'interface :
```bash
cd app
streamlit run interface.py
```

3. Accéder à l'API :
- Interface Swagger : http://localhost:8000/docs
- Endpoint principal : http://localhost:8000
- Endpoint de chat : http://localhost:8000/chat

## Structure du Projet

```
NPL-1/
├── app/
│   ├── main.py          # Point d'entrée de l'application
│   ├── chatbot.py       # Logique du chatbot
│   └── utils.py         # Fonctions utilitaires
├── data/                # Données d'entraînement et de test
├── notebook/            # Notebooks Jupyter
├── requirements.txt     # Dépendances du projet
└── README.md           # Documentation
```

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Licence

Ce projet est sous licence MIT.

🚀 Démarrage du serveur Chatbot API...
📝 Documentation disponible sur : http://localhost:8000/docs
