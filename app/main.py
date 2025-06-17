"""
Point d'entrée principal de l'application.
Ce script initialise et configure le serveur FastAPI pour le chatbot.
Il gère les routes API et l'intégration avec les autres composants du système.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from chatbot import Chatbot
import time
from monitoring import Monitoring
from utils import search_wikipedia, summarize_text

# Initialisation de l'application FastAPI avec titre et description
app = FastAPI(title="Chatbot API", description="API pour le chatbot de support")

# Configuration CORS pour permettre les requêtes cross-origin
# Nécessaire pour l'intégration avec l'interface web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines en développement
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes HTTP
    allow_headers=["*"],  # Autorise tous les headers
)

# Initialisation des composants principaux
chatbot = Chatbot()  # Instance du chatbot pour le traitement des messages
monitoring = Monitoring()  # Système de monitoring pour le suivi des performances

# Modèle Pydantic pour la validation des messages entrants
class Message(BaseModel):
    text: str  # Le texte du message à traiter

# Route racine - Page d'accueil de l'API
@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API du Chatbot"}

# # Route pour obtenir les métriques actuelles du système
# @app.get("/metrics")
# async def get_metrics():
#     """Endpoint pour obtenir les métriques actuelles du système"""
#     return monitoring.get_metrics()

# # Route pour obtenir un résumé des performances
# @app.get("/performance")
# async def get_performance():
#     """Endpoint pour obtenir un résumé des performances du système"""
#     return monitoring.get_performance_summary()

# Route principale de chat - Traite les messages et génère des réponses
@app.post("/chat")
async def chat(message: Message):
    start_time = time.time()  # Début du chronométrage
    try:
        # Génération de la réponse avec le chatbot
        response = chatbot.generate_response(message.text)
        processing_time = time.time() - start_time  # Calcul du temps de traitement
        
        # Enregistrement de la requête dans le système de monitoring
        monitoring.log_request(message.text, response, processing_time)
        
        return response
    except Exception as e:
        # Gestion des erreurs avec logging
        processing_time = time.time() - start_time
        monitoring.log_request(message.text, {"text": str(e)}, processing_time, error=e)
        raise HTTPException(status_code=500, detail=str(e))

# Route pour la classification
@app.post("/classify")
async def classify(message: Message):
    start_time = time.time()
    try:
        # Classification du message
        result = chatbot.classify_text(message.text)
        response = {"text": result, "category": "classification", "confidence": 1.0}
        processing_time = time.time() - start_time
        
        # Log de la requête
        monitoring.log_request(message.text, response, processing_time)
        
        return response 
    except Exception as e:
        # Gestion des erreurs avec logging
        processing_time = time.time() - start_time
        monitoring.log_request(message.text, {"text": str(e)}, processing_time, error=e)
        raise HTTPException(status_code=500, detail=str(e))

# Route pour la recherche Wikipedia
@app.post("/wiki")
async def wiki_search(message: Message):
    start_time = time.time()
    try:
        # Recherche Wikipedia et formatage de la réponse
        result = search_wikipedia(message.text)
        response = {"text": result, "category": "wikipedia", "confidence": 1.0}
        processing_time = time.time() - start_time
        
        # Log de la requête
        monitoring.log_request(message.text, response, processing_time)
        
        return response
    except Exception as e:
        # Gestion des erreurs avec logging
        processing_time = time.time() - start_time
        monitoring.log_request(message.text, {"text": str(e)}, processing_time, error=e)
        raise HTTPException(status_code=500, detail=str(e))

# Route pour le résumé de texte
@app.post("/summarize")
async def summarize(message: Message):
    start_time = time.time()
    try:
        # Génération du résumé et formatage de la réponse
        result = summarize_text(message.text)
        response = {"text": result, "category": "summary", "confidence": 1.0}
        processing_time = time.time() - start_time
        
        # Log de la requête
        monitoring.log_request(message.text, response, processing_time)
        
        return response
    except Exception as e:
        # Gestion des erreurs avec logging
        processing_time = time.time() - start_time
        monitoring.log_request(message.text, {"text": str(e)}, processing_time, error=e)
        raise HTTPException(status_code=500, detail=str(e))

# Point d'entrée du script
if __name__ == "__main__":
    print("C'est parti, Kero ! 🐸")
    print("📝 Documentation disponible sur : http://localhost:8000/docs")
    # Démarrage du serveur avec hot-reload activé
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 