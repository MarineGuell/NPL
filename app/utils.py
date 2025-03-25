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

    # 5. Trier les phrases selon leur ordre d’origine
    top_indices.sort()
    summary = [sentences[i] for i in top_indices]

    return " ".join(summary)

## Fonction de recherche sur Wikipédia
import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia('en')

def search_wikipedia(query, sentences=2):
    page = wiki_wiki.page(query)
    if not page.exists():
        return f"Désolé, je n'ai rien trouvé sur « {query} » sur Wikipédia."

    summary = page.summary
    return summarize_text(summary, num_sentences=sentences)
