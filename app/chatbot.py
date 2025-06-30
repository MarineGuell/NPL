"""
Module Orchestrateur du Chatbot Kaeru - Pipeline Central de NLP

Ce module orchestre le pipeline complet du chatbot :
- Interface utilisateur (Streamlit) ↔ Orchestrateur ↔ Modèles
- Prétraitement centralisé (TextPreprocessor)
- Classification (ML : TF-IDF+Naive Bayes, DL : LSTM bidirectionnel)
- Résumé (ML : similarité cosinus, DL : autoencodeur extractif)
- Recherche Wikipedia intelligente

Pipeline de données :
1. Réception texte utilisateur → Prétraitement (nettoyage, normalisation)
2. Transformation numérique (vectorisation TF-IDF ou tokenization)
3. Prédiction avec modèle approprié (ML/DL selon fonction)
4. Formatage réponse avec personnalité grenouille japonaise

Fonctions disponibles via l'interface :
- Classification ML : Pipeline optimisé GridSearchCV
- Classification DL : LSTM bidirectionnel avec BatchNormalization  
- Résumé ML : Similarité cosinus TF-IDF (3 phrases les plus représentatives)
- Résumé DL : Autoencodeur extractif (phrases les mieux reconstruites)
- Recherche Wikipedia : Extraction mots-clés + recherche intelligente

Tous les modèles sont automatiquement chargés depuis models/ et sauvegardés lors de l'entraînement.
"""

from utils import TextPreprocessor
from models import MLModel, DLModel, AutoencoderSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Téléchargement automatique de punkt si nécessaire
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📥 Téléchargement automatique de punkt pour le chatbot...")
    try:
        nltk.download('punkt', quiet=True)
        print("✅ punkt téléchargé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement de punkt: {e}")

class ChatbotOrchestrator:
    """
    Orchestre le pipeline complet :
    - Interface utilisateur (Streamlit)
    - Prétraitement (TextPreprocessor)
    - Classification (ML et DL)
    - Résumé (ML : TF-IDF/cosinus, DL : autoencodeur extractif)
    
    Détail du résumé DL (autoencodeur) :
    1. Lors de l'entraînement global, l'autoencodeur est entraîné sur toutes les phrases du dataset.
    2. Pour résumer un texte, on découpe en phrases, on vectorise, on passe chaque phrase dans l'autoencodeur.
    3. On calcule l'erreur de reconstruction pour chaque phrase.
    4. On sélectionne les phrases avec l'erreur la plus faible (les plus "centrales").
    5. Le résumé est la concaténation de ces phrases dans l'ordre d'origine.
    """
    
    def __init__(self):
        """
        Initialise l'orchestrateur avec tous les composants nécessaires.
        """
        print("🤖 Initialisation de l'orchestrateur...")
        self.preprocessor = TextPreprocessor()
        self.ml_classifier = MLModel()
        self.dl_classifier = DLModel()
        
        # Le modèle de résumé DL (Autoencodeur) est chargé ici
        print("🔄 Chargement du modèle de résumé (Autoencodeur)...")
        self.autoencoder_summarizer = AutoencoderSummarizer()
        
        # Vérification automatique si les modèles sont entraînés
        self.is_trained = self._check_models_trained()
        print("✅ Orchestrateur prêt.")

    def _check_models_trained(self):
        """
        Vérifie si les modèles sont déjà entraînés et disponibles.
        """
        ml_trained = (self.ml_classifier.model is not None and 
                     self.ml_classifier.vectorizer is not None)
        dl_trained = self.dl_classifier.model is not None
        autoencoder_trained = self.autoencoder_summarizer.model is not None
        
        if ml_trained and dl_trained and autoencoder_trained:
            print("✅ Tous les modèles sont chargés et prêts à l'utilisation")
            return True
        elif ml_trained and dl_trained:
            print("✅ Modèles de classification chargés (autoencodeur non disponible)")
            return True
        else:
            print("⚠️ Modèles non entraînés - veuillez exécuter le script d'entraînement")
            return False

    def train_models(self, texts, labels):
        """
        Entraîne tous les modèles :
        - ML (TF-IDF + Naive Bayes)
        - DL (LSTM bidirectionnel)
        - Autoencodeur pour le résumé extractif
        
        Étapes détaillées :
        1. Prétraitement des textes (nettoyage, normalisation)
        2. Entraînement du modèle ML + évaluation
        3. Préparation et entraînement du modèle DL
        4. Entraînement de l'autoencodeur sur toutes les phrases du dataset
        """
        print("🔄 Prétraitement des textes pour l'entraînement...")
        processed_texts = self.preprocessor.transform(texts)
        
        print("🔄 Entraînement du modèle ML...")
        self.ml_classifier.train(processed_texts, labels)
        self.ml_classifier.evaluate()
        
        print("🔄 Préparation et entraînement du modèle DL...")
        X_dl, y_dl = self.dl_classifier.prepare(processed_texts, labels)
        self.dl_classifier.train(X_dl, y_dl)
        
        # Entraînement de l'autoencodeur pour le résumé
        print("🔄 Entraînement de l'autoencodeur pour le résumé...")
        self.autoencoder_summarizer.train(texts.tolist())
        
        self.is_trained = True
        print("✅ Modèles entraînés avec succès.")

    def classify(self, text, model_type='ml'):
        """
        Classe un texte donné en utilisant le modèle spécifié.
        
        Args:
            text (str): Le texte brut à classifier.
            model_type (str): 'ml' ou 'dl'.
            
        Returns:
            str: La prédiction formatée.
        """
        if not self.is_trained:
            return "The models aren't trained yet! 🐸 Please run the training script first, kero!"

        print(f"🔄 Classification with {model_type.upper()} model...")
        # Prétraitement complet du texte d'entrée
        processed_text = self.preprocessor.normalize(self.preprocessor.clean(text))
        
        if model_type == 'ml':
            prediction = self.ml_classifier.predict([processed_text])[0]
            proba = self.ml_classifier.predict_proba([processed_text])[0].max()
        else: # 'dl'
            prediction = self.dl_classifier.predict([processed_text])[0]
            proba = self.dl_classifier.predict_proba([processed_text])[0].max()
        
        print(f"🐸 Debug - Prédiction: {prediction}, Probabilité max: {proba:.3f}")
        
        # Réponses personnalisées selon la confiance
        if proba > 0.8:
            response = f"*hops excitedly* 🐸 This text is definitely about **{prediction}**! I'm {proba:.1%} confident, kero!"
        elif proba > 0.6:
            response = f"*tilts head thoughtfully* 🐸 I think this text is about **{prediction}**. I'm {proba:.1%} sure, kero!"
        else:
            response = f"*croaks uncertainly* 🐸 Hmm... I'm not very confident, but I'd say it's about **{prediction}** ({proba:.1%} sure). Maybe I need more training, kero!"
        
        print(f"🐸 Debug - Réponse générée: {response}")
        return response

    def summarize(self, text, model_type='dl'):
        """
        Résume un texte donné en utilisant la méthode spécifiée.
        - 'ml' : résumé extractif TF-IDF/cosinus (phrases les plus similaires au texte global)
        - 'dl' : résumé extractif autoencodeur (phrases les mieux reconstruites)
        
        Détail du résumé autoencodeur :
        1. Découpage du texte en phrases
        2. Vectorisation de chaque phrase
        3. Passage dans l'autoencodeur
        4. Calcul de l'erreur de reconstruction
        5. Sélection des phrases avec l'erreur la plus faible
        6. Assemblage du résumé
        """
        print(f"🔄 Summarization with {model_type.upper()} method...")
        if model_type == 'ml':
            # === MÉTHODE EXTRACTIVE BASÉE SUR LA SIMILARITÉ COSINUS ===
            # 
            # ÉTAPE 1: Tokenisation du texte en phrases
            # Découpage du texte en phrases individuelles pour analyse
            sentences = sent_tokenize(text)
            if len(sentences) < 3:
                return "This text is too short to summarize! It's already quite concise."
            
            # ÉTAPE 2: Vectorisation TF-IDF du texte entier
            # - Création d'un vectorizer TF-IDF avec bigrammes (1-2 mots)
            # - Suppression des mots vides anglais pour se concentrer sur le contenu
            # - Le vectorizer "apprend" le vocabulaire du texte entier
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            text_vector = vectorizer.fit_transform([text])
            
            # ÉTAPE 3: Vectorisation de chaque phrase
            # - Utilisation du même vectorizer pour assurer la cohérence
            # - Chaque phrase est transformée en vecteur avec le même vocabulaire
            sentence_vectors = vectorizer.transform(sentences)
            
            # ÉTAPE 4: Calcul de la similarité cosinus
            # - Mesure de l'angle entre chaque phrase et le texte entier
            # - Valeur proche de 1 = très similaire, proche de 0 = très différent
            # - Les phrases avec la plus haute similarité sont les plus représentatives
            similarities = cosine_similarity(sentence_vectors, text_vector).flatten()
            
            # ÉTAPE 5: Sélection des phrases les plus représentatives
            # - Tri des phrases par similarité décroissante
            # - Sélection des 3 phrases les plus similaires au texte entier
            # - Préservation de l'ordre original pour maintenir la cohérence narrative
            num_sentences = min(3, len(sentences))
            top_indices = similarities.argsort()[-num_sentences:][::-1]
            top_indices.sort()  # Garder l'ordre original des phrases
            
            # ÉTAPE 6: Assemblage du résumé
            # - Concaténation des phrases sélectionnées
            # - Maintien de la fluidité narrative
            summary = " ".join([sentences[i] for i in top_indices])
            return f"In short, your text says : {summary} Kero 🐸"
        else: # 'dl'
            # === MÉTHODE EXTRACTIVE BASÉE SUR L'AUTOENCODEUR ===
            try:
                summary = self.autoencoder_summarizer.summarize(text, num_sentences=3)
                return f"In short, your text says : {summary} Kero 🐸"
            except Exception as e:
                return f"*croaks apologetically* 🐸 Sorry, I couldn't summarize with the autoencoder: {str(e)}, kero!" 