from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from chatbot import Chatbot

app = FastAPI(title="Chatbot API", description="API pour le chatbot de support")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation du chatbot
chatbot = Chatbot()

class Message(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API du Chatbot"}

@app.post("/chat")
async def chat(message: Message):
    try:
        # Génération de la réponse avec le chatbot
        response = chatbot.generate_response(message.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Démarrage du serveur Chatbot API...")
    print("📝 Documentation disponible sur : http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 