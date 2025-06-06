from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from chatbot import Chatbot
import time
from monitoring import Monitoring
from utils import search_wikipedia, summarize_text

app = FastAPI(title="Chatbot API", description="API pour le chatbot de support")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation du chatbot et du monitoring
chatbot = Chatbot()
monitoring = Monitoring()

class Message(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API du Chatbot"}

@app.get("/metrics")
async def get_metrics():
    """Endpoint pour obtenir les métriques actuelles"""
    return monitoring.get_metrics()

@app.get("/performance")
async def get_performance():
    """Endpoint pour obtenir un résumé des performances"""
    return monitoring.get_performance_summary()

@app.post("/chat")
async def chat(message: Message):
    start_time = time.time()
    try:
        # Génération de la réponse avec le chatbot
        response = chatbot.generate_response(message.text)
        processing_time = time.time() - start_time
        
        # Log de la requête
        monitoring.log_request(message.text, response, processing_time)
        
        return response
    except Exception as e:
        processing_time = time.time() - start_time
        monitoring.log_request(message.text, {"text": str(e)}, processing_time, error=e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/wiki")
async def wiki_search(message: Message):
    start_time = time.time()
    try:
        # Recherche Wikipedia
        result = search_wikipedia(message.text)
        response = {"text": result, "category": "wikipedia", "confidence": 1.0}
        processing_time = time.time() - start_time
        
        # Log de la requête
        monitoring.log_request(message.text, response, processing_time)
        
        return response
    except Exception as e:
        processing_time = time.time() - start_time
        monitoring.log_request(message.text, {"text": str(e)}, processing_time, error=e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize(message: Message):
    start_time = time.time()
    try:
        # Résumé de texte
        result = summarize_text(message.text)
        response = {"text": result, "category": "summary", "confidence": 1.0}
        processing_time = time.time() - start_time
        
        # Log de la requête
        monitoring.log_request(message.text, response, processing_time)
        
        return response
    except Exception as e:
        processing_time = time.time() - start_time
        monitoring.log_request(message.text, {"text": str(e)}, processing_time, error=e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Démarrage du serveur Chatbot API...")
    print("📝 Documentation disponible sur : http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 