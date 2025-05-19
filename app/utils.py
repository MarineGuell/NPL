### nettoyage de texte
def cleaning(sentence):

    # Basic cleaning
    sentence = sentence.strip()
    sentence = sentence.lower()
    sentence = ''.join(char for char in sentence if not char.isdigit())
###
    # retirer la premiere adresse mail , puis toutes les autres
    sentence = re.sub(r'From:.*?Subject:', '', sentence, flags=re.DOTALL)
    sentence = re.sub(r'\S+@\S+', '', sentence)

    # Remove words with 3+ consecutive repeating letters
    sentence = re.sub(r'\b\w*(\w)\1{2,}\w*\b', '', sentence)

    # Remove URLs
    sentence = re.sub(r'http\S+|www\S+|https\S+', '', sentence, flags=re.MULTILINE)
###
    # Advanced cleaning
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '')

    tokenized_sentence = word_tokenize(sentence)
    tokenized_sentence_cleaned = [
        w for w in tokenized_sentence if not w in set(stopwords.words('english'))
    ]

    lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos = "v")
        for word in tokenized_sentence_cleaned
    ]

    cleaned_sentence = ' '.join(word for word in lemmatized)

    return cleaned_sentence


## Fonction de résumé simple (baseline)

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def summarize_text(text, num_sentences=2):
    # 1. Découper le texte en phrases
    sentences = sent_tokenize(text)
    
    # 2. Appliquer TF-IDF sur les phrases
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    # 3. Calculer un score pour chaque phrase
    scores = np.sum(X.toarray(), axis=1)

    # 4. Sélectionner les phrases les plus importantes
    top_indices = scores.argsort()[-num_sentences:][::-1]

    # 5. Trier les phrases selon leur ordre d'origine
    top_indices.sort()
    summary = [sentences[i] for i in top_indices]

    return " ".join(summary)

## Fonction de recherche sur Wikipédia
import wikipedia
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

# Téléchargement des ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def summarize_text(text: str, max_length: int = 150) -> str:
    """
    Résume un texte en utilisant un modèle de résumé automatique
    """
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Erreur lors du résumé du texte: {str(e)}")
        return text

def search_wikipedia(query: str, sentences: int = 2) -> str:
    """
    Recherche des informations sur Wikipedia
    """
    try:
        # Recherche de la page Wikipedia
        search_results = wikipedia.search(query, results=1)
        if not search_results:
            return "Désolé, je n'ai pas trouvé d'informations sur ce sujet."
        
        # Récupération du contenu
        page = wikipedia.page(search_results[0])
        content = page.content
        
        # Tokenization en phrases
        sentences_list = sent_tokenize(content)
        
        # Retourne les premières phrases
        return " ".join(sentences_list[:sentences])
    except Exception as e:
        print(f"Erreur lors de la recherche Wikipedia: {str(e)}")
        return "Désolé, une erreur s'est produite lors de la recherche."

def preprocess_text(text: str) -> str:
    """
    Prétraite le texte en le nettoyant
    """
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des caractères spéciaux
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    
    return text.strip()
