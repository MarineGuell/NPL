"""
Module d'utilitaires pour le prétraitement et le chargement des données.
"""

import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import wikipedia

class DataLoader:
    """
    Classe pour charger et préparer les données.
    """
    def __init__(self, filepath):
        """
        Initialise le chargeur de données.
        
        Args:
            filepath (str): Chemin vers le fichier CSV
        """
        self.data = pd.read_csv(filepath)
        self.clean_data()

    def clean_data(self):
        """
        Nettoie les données en supprimant les doublons et les valeurs manquantes.
        """
        # Suppression des doublons
        self.data = self.data.drop_duplicates()
        
        # Suppression des lignes avec valeurs manquantes
        self.data = self.data.dropna()
        
        # Réinitialisation de l'index
        self.data = self.data.reset_index(drop=True)

    def get_texts_and_labels(self):
        """
        Retourne les textes et les labels.
        
        Returns:
            tuple: (texts, labels)
        """
        return self.data['text'], self.data['category']

    def split_data(self, texts, labels, test_size=0.2, random_state=42):
        """
        Divise les données en ensembles d'entraînement et de test.
        
        Args:
            texts: Les textes
            labels: Les labels
            test_size (float): Proportion des données de test
            random_state (int): Seed pour la reproductibilité
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(texts, labels, test_size=test_size, random_state=random_state)

class TextPreprocessor:
    """
    Classe pour le prétraitement des textes avec POS-tagging avancé.
    """
    def __init__(self):
        """
        Initialise le prétraiteur de texte.
        Télécharge automatiquement les ressources NLTK nécessaires.
        """
        # Téléchargement automatique des ressources NLTK avec gestion d'erreurs
        self._download_nltk_resources()
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Mapping des tags POS NLTK vers WordNet
        self.pos_mapping = {
            'J': 'a',  # Adjective
            'V': 'v',  # Verb
            'N': 'n',  # Noun
            'R': 'r'   # Adverb
        }

    def _download_nltk_resources(self):
        """
        Télécharge automatiquement les ressources NLTK nécessaires.
        """
        resources = [
            'stopwords',
            'wordnet', 
            'punkt',
            'averaged_perceptron_tagger',
            'averaged_perceptron_tagger_eng'  # Version anglaise spécifique
        ]
        
        for resource in resources:
            try:
                # Vérifier si la ressource existe déjà
                try:
                    if resource == 'stopwords':
                        stopwords.words('english')
                    elif resource == 'wordnet':
                        from nltk.corpus import wordnet
                        wordnet.synsets('test')
                    elif resource == 'punkt':
                        from nltk.tokenize import word_tokenize
                        word_tokenize('test')
                    elif resource in ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
                        from nltk.tag import pos_tag
                        pos_tag(['test'])
                except LookupError:
                    # La ressource n'existe pas, la télécharger
                    print(f"📥 Téléchargement automatique de {resource}...")
                    nltk.download(resource, quiet=True)
                    print(f"✅ {resource} téléchargé avec succès")
                    
            except Exception as e:
                print(f"⚠️ Erreur lors du téléchargement de {resource}: {e}")
                print("Tentative de téléchargement manuel...")
                try:
                    nltk.download(resource, quiet=True)
                except:
                    print(f"❌ Impossible de télécharger {resource}")
                    # Si c'est le POS tagger qui pose problème, essayer l'autre version
                    if resource == 'averaged_perceptron_tagger_eng':
                        try:
                            print("🔄 Tentative avec averaged_perceptron_tagger...")
                            nltk.download('averaged_perceptron_tagger', quiet=True)
                        except:
                            print("❌ Impossible de télécharger le POS tagger")

    def get_wordnet_pos(self, tag):
        """
        Convertit les tags POS de NLTK vers les tags WordNet.
        
        Args:
            tag (str): Tag POS de NLTK
            
        Returns:
            str: Tag POS pour WordNet
        """
        # Premier caractère du tag (plus général)
        first_char = tag[0].upper()
        return self.pos_mapping.get(first_char, 'n')  # Par défaut: nom

    def clean(self, text):
        """
        Nettoie un texte de manière approfondie avec POS-tagging.
        
        Args:
            text (str): Le texte à nettoyer
            
        Returns:
            str: Le texte nettoyé
        """
        # Basic cleaning
        text = text.strip()
        text = text.lower()
        text = ''.join(char for char in text if not char.isdigit())

        # Nettoyage des emails
        text = re.sub(r'From:.*?Subject:', '', text, flags=re.DOTALL)
        text = re.sub(r'\S+@\S+', '', text)

        # Suppression des mots avec 3+ lettres consécutives identiques
        text = re.sub(r'\b\w*(\w)\1{2,}\w*\b', '', text)

        # Suppression des URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Suppression de la ponctuation
        for punctuation in string.punctuation:
            text = text.replace(punctuation, '')

        # 1. ✅ TOKENISATION
        tokenized_text = word_tokenize(text)
        
        # 2. ✅ SUPPRESSION DES STOPWORDS
        tokenized_text_cleaned = [
            w for w in tokenized_text if not w in self.stop_words
        ]

        # 3. ✅ POS-TAGGING AVEC GESTION D'ERREUR
        try:
            pos_tagged = pos_tag(tokenized_text_cleaned)
            
            # 4. ✅ LEMMATISATION AVANCÉE AVEC POS-TAGGING
            lemmatized = []
            for word, tag in pos_tagged:
                # Conversion du tag POS pour WordNet
                wordnet_pos = self.get_wordnet_pos(tag)
                # Lemmatisation avec le bon POS
                lemmatized_word = self.lemmatizer.lemmatize(word, pos=wordnet_pos)
                lemmatized.append(lemmatized_word)
                
        except LookupError as e:
            # Fallback si le POS tagger n'est pas disponible
            print(f"⚠️ POS tagger non disponible: {e}")
            print("🔄 Utilisation de la lemmatisation simple...")
            
            # Lemmatisation simple sans POS tagging
            lemmatized = []
            for word in tokenized_text_cleaned:
                lemmatized_word = self.lemmatizer.lemmatize(word)
                lemmatized.append(lemmatized_word)
                
        except Exception as e:
            # Autre erreur, retourner le texte nettoyé sans lemmatisation
            print(f"⚠️ Erreur lors du POS tagging: {e}")
            print("🔄 Retour du texte nettoyé sans lemmatisation...")
            lemmatized = tokenized_text_cleaned

        # Reconstruction du texte
        cleaned_text = ' '.join(word for word in lemmatized)

        return cleaned_text

    def clean_with_pos_info(self, text):
        """
        Version avancée qui retourne aussi les informations POS.
        
        Args:
            text (str): Le texte à nettoyer
            
        Returns:
            tuple: (texte_nettoyé, liste_des_pos_tags)
        """
        # Basic cleaning
        text = text.strip()
        text = text.lower()
        text = ''.join(char for char in text if not char.isdigit())

        # Nettoyage des emails et URLs
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Suppression de la ponctuation
        for punctuation in string.punctuation:
            text = text.replace(punctuation, '')

        # Tokenisation
        tokenized_text = word_tokenize(text)
        
        # Suppression des stopwords
        tokenized_text_cleaned = [
            w for w in tokenized_text if not w in self.stop_words
        ]

        # POS-tagging
        pos_tagged = pos_tag(tokenized_text_cleaned)

        # Lemmatisation avec POS
        lemmatized = []
        pos_info = []
        for word, tag in pos_tagged:
            wordnet_pos = self.get_wordnet_pos(tag)
            lemmatized_word = self.lemmatizer.lemmatize(word, pos=wordnet_pos)
            lemmatized.append(lemmatized_word)
            pos_info.append((lemmatized_word, tag, wordnet_pos))

        cleaned_text = ' '.join(word for word in lemmatized)
        
        return cleaned_text, pos_info

    def normalize(self, text):
        """
        Normalise un texte.
        
        Args:
            text (str): Le texte à normaliser
            
        Returns:
            str: Le texte normalisé
        """
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        # Suppression des espaces en début et fin
        text = text.strip()
        
        return text

    def transform(self, texts):
        """
        Transforme une liste de textes.
        
        Args:
            texts: Les textes à transformer
            
        Returns:
            Series: Les textes transformés
        """
        # Nettoyage avec POS-tagging
        cleaned_texts = texts.apply(self.clean)
        
        # Normalisation
        normalized_texts = cleaned_texts.apply(self.normalize)
        
        return normalized_texts

    def transform_with_pos_info(self, texts):
        """
        Transforme une liste de textes avec informations POS.
        
        Args:
            texts: Les textes à transformer
            
        Returns:
            tuple: (textes_transformés, informations_pos_par_texte)
        """
        cleaned_texts = []
        pos_infos = []
        
        for text in texts:
            cleaned_text, pos_info = self.clean_with_pos_info(text)
            cleaned_texts.append(cleaned_text)
            pos_infos.append(pos_info)
        
        # Normalisation
        normalized_texts = [self.normalize(text) for text in cleaned_texts]
        
        return normalized_texts, pos_infos

    def get_pos_statistics(self, texts, sample_size=1000):
        """
        Calcule les statistiques des parties du discours dans un échantillon de textes.
        
        Args:
            texts: Les textes à analyser
            sample_size (int): Taille de l'échantillon
            
        Returns:
            dict: Statistiques des POS
        """
        if len(texts) > sample_size:
            sample_texts = texts.sample(n=sample_size, random_state=42)
        else:
            sample_texts = texts
        
        pos_counts = {}
        total_words = 0
        
        for text in sample_texts:
            _, pos_info = self.clean_with_pos_info(text)
            for word, tag, wordnet_pos in pos_info:
                pos_counts[tag] = pos_counts.get(tag, 0) + 1
                total_words += 1
        
        # Calcul des pourcentages
        pos_stats = {}
        for tag, count in pos_counts.items():
            pos_stats[tag] = {
                'count': count,
                'percentage': (count / total_words) * 100
            }
        
        return pos_stats

    def extract_pos_words(self, text, pos_types=None):
        """
        Extrait les mots selon leur partie du discours (POS).
        
        Args:
            text (str): Le texte à analyser
            pos_types (list): Liste des types POS à extraire. Par défaut: ['NN', 'VB', 'JJ']
                            - 'NN': Noms (nouns)
                            - 'VB': Verbes (verbs) 
                            - 'JJ': Adjectifs (adjectives)
                            - 'RB': Adverbes (adverbs)
                            - 'PRP': Pronoms (pronouns)
                            - etc.
        
        Returns:
            dict: Dictionnaire avec les mots groupés par type POS
        """
        if pos_types is None:
            pos_types = ['NN', 'VB', 'JJ']  # Noms, Verbes, Adjectifs par défaut
        
        # Nettoyage et POS-tagging
        _, pos_info = self.clean_with_pos_info(text)
        
        # Extraction des mots par type POS
        pos_words = {pos_type: [] for pos_type in pos_types}
        
        for word, tag, wordnet_pos in pos_info:
            # Vérifier si le tag commence par le type POS recherché
            for pos_type in pos_types:
                if tag.startswith(pos_type):
                    pos_words[pos_type].append(word)
                    break
        
        return pos_words

    def extract_verbs_adjectives_nouns(self, text):
        """
        Extrait spécifiquement les verbes, adjectifs et noms d'un texte.
        
        Args:
            text (str): Le texte à analyser
            
        Returns:
            dict: Dictionnaire avec 'verbs', 'adjectives', 'nouns'
        """
        pos_words = self.extract_pos_words(text, ['NN', 'VB', 'JJ'])
        
        return {
            'verbs': pos_words.get('VB', []),
            'adjectives': pos_words.get('JJ', []),
            'nouns': pos_words.get('NN', [])
        }

    def get_pos_summary(self, text):
        """
        Génère un résumé des parties du discours dans un texte.
        
        Args:
            text (str): Le texte à analyser
            
        Returns:
            dict: Résumé avec statistiques et mots par type
        """
        # Extraction des mots par POS
        pos_words = self.extract_pos_words(text)
        
        # Statistiques
        total_words = sum(len(words) for words in pos_words.values())
        
        summary = {
            'total_words_analyzed': total_words,
            'pos_distribution': {},
            'words_by_pos': pos_words
        }
        
        # Calcul de la distribution
        for pos_type, words in pos_words.items():
            if total_words > 0:
                percentage = (len(words) / total_words) * 100
            else:
                percentage = 0
                
            summary['pos_distribution'][pos_type] = {
                'count': len(words),
                'percentage': percentage,
                'words': words
            }
        
        return summary

    def clean_for_autoencoder(self, text):
        """
        Prétraitement minimal pour l'autoencodeur.
        Préserve les phrases complètes et la ponctuation pour la tokenisation.
        
        Args:
            text (str): Le texte à nettoyer
            
        Returns:
            str: Le texte nettoyé pour l'autoencodeur
        """
        # Nettoyage de base très léger
        text = text.strip()
        
        # Suppression des URLs et emails (peuvent perturber la tokenisation)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Suppression des caractères de contrôle
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Suppression des chiffres sauf ceux à exactement 4 chiffres
        # Utilise une regex pour préserver les nombres à 4 chiffres
        text = re.sub(r'\b(?!\d{4}\b)\d+\b', '', text)
        
        # Normalisation des espaces (mais pas suppression)
        text = re.sub(r'\s+', ' ', text)
        
        # Suppression des lignes vides multiples
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()

    def transform_for_autoencoder(self, texts):
        """
        Transforme une liste de textes pour l'autoencodeur.
        Utilise un prétraitement minimal pour préserver les phrases.
        
        Args:
            texts: Les textes à transformer
            
        Returns:
            list: Les textes transformés pour l'autoencodeur
        """
        cleaned_texts = []
        
        for text in texts:
            cleaned_text = self.clean_for_autoencoder(text)
            if cleaned_text:  # Ne garder que les textes non vides
                cleaned_texts.append(cleaned_text)
        
        return cleaned_texts

def encode_labels(labels):
    """
    Encode les labels en utilisant LabelEncoder.
    
    Args:
        labels: Les labels à encoder
        
    Returns:
        tuple: (encoded_labels, encoder)
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels, encoder

def normalize_text(text):
    """
    Normalise un texte unique.
    
    Args:
        text (str): Le texte à normaliser
        
    Returns:
        str: Le texte normalisé
    """
    # Suppression des espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    # Suppression des espaces en début et fin
    text = text.strip()
    
    return text

def search_wikipedia_smart(query):
    """
    Recherche intelligente sur Wikipedia avec gestion de l'ambiguïté.
    Retourne un dictionnaire avec le statut et les suggestions/pages trouvées.
    """
    wikipedia.set_lang("en")
    try:
        # Recherche de pages correspondant à la requête
        results = wikipedia.search(query, results=5)
        if not results:
            return {
                'status': 'error',
                'message': "Aucune page Wikipedia trouvée pour cette requête, kero ! 🐸"
            }
        if len(results) == 1:
            # Succès direct
            page = results[0]
            summary = wikipedia.summary(page, sentences=3)
            return {
                'status': 'success',
                'page': page,
                'summary': summary
            }
        else:
            # Ambiguïté : plusieurs pages possibles
            # On regroupe sous un seul mot-clé pour l'interface
            suggestions = {query: results}
            return {
                'status': 'ambiguous',
                'suggestions': suggestions
            }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Erreur lors de la recherche Wikipedia : {str(e)} kero 🐸"
        }