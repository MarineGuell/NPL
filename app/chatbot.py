import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from utils import summarize_text, search_wikipedia

# Charger le modèle et le vectoriseur
model = joblib.load("app/model.joblib")
vectorizer = joblib.load("app/vectorizer.joblib")

# Prétraitement
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Lancer le chatbot
# def run_chatbot():
#     print("🤖 Chatbot NLP - Catégorisation de texte")
#     while True:
#         user_input = input("Vous : ")
#         if user_input.lower() in ["quit", "exit", "bye"]:
#             print("Chatbot : À bientôt !")
#             break

#         clean = preprocess_text(user_input)
#         vect = vectorizer.transform([clean])
#         prediction = model.predict(vect)[0]

#         print(f"Chatbot : Ce texte semble parler de **{prediction}**.")

def run_chatbot():
    print("🤖 Chatbot NLP - Classifieur + Résumeur")
    print("Tapez 'resume: votre texte' pour générer un résumé.")
    print("Tapez 'quit' pour quitter.")
    
    while True:
        user_input = input("Vous : ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Chatbot : À bientôt !")
            break

        if user_input.lower().startswith("resume:"):
            texte = user_input[7:].strip()
            summary = summarize_text(texte)
            print("Chatbot (résumé) :", summary)

        elif user_input.lower().startswith("wiki:"):
            query = user_input[5:].strip()
            result = search_wikipedia(query)
            print("Chatbot (wikipedia) :", result)

        else:
            clean = preprocess_text(user_input)
            vect = vectorizer.transform([clean])
            prediction = model.predict(vect)[0]
            print(f"Chatbot (catégorie) : Ce texte semble parler de **{prediction}**.")


class Chatbot:
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-medium"
        self.tokenizer = None
        self.model = None
        self.initialize_model()

    def initialize_model(self):
        """Initialise le modèle et le tokenizer"""
        try:
            print("🔄 Chargement du modèle DialoGPT...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            print("✅ Modèle chargé avec succès !")
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation du modèle: {str(e)}")
            raise

    def generate_response(self, user_input: str) -> str:
        """Génère une réponse basée sur l'entrée de l'utilisateur"""
        try:
            # Encodage de l'entrée utilisateur
            input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, 
                                            return_tensors='pt')
            
            # Génération de la réponse
            output = self.model.generate(
                input_ids,
                max_length=1000,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.7,
                temperature=0.8
            )
            
            # Décodage de la réponse
            response = self.tokenizer.decode(output[:, input_ids.shape[-1]:][0], 
                                           skip_special_tokens=True)
            
            return response if response else "Je ne comprends pas votre demande."
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération de la réponse: {str(e)}")
            return "Désolé, une erreur s'est produite. Veuillez réessayer."


if __name__ == "__main__":
    # Test simple du chatbot
    chatbot = Chatbot()
    print("🤖 Chatbot initialisé ! Tapez 'quit' pour quitter.")
    
    while True:
        user_input = input("Vous : ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Chatbot : Au revoir !")
            break
            
        response = chatbot.generate_response(user_input)
        print("Chatbot :", response)
