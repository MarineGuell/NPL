import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

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


if __name__ == "__main__":
    run_chatbot()
