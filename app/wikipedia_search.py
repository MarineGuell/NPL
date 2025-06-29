"""
Module de recherche Wikipedia intelligente pour le chatbot Kaeru.

Utilise les modèles ML/DL entraînés pour extraire les mots-clés les plus pertinents
d'une phrase utilisateur et propose les pages Wikipedia correspondantes.
"""

import re
import numpy as np
import pandas as pd
import wikipedia
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import sys

# Ajout du chemin pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import TextPreprocessor
from models import MLModel, DLModel, AutoencoderSummarizer

class WikipediaIntelligentSearch:
    """
    Recherche Wikipedia intelligente utilisant les modèles entraînés.
    """
    
    def __init__(self):
        """
        Initialise le système de recherche Wikipedia.
        """
        self.preprocessor = TextPreprocessor()
        self.ml_model = MLModel()
        self.dl_model = DLModel()
        self.autoencoder = AutoencoderSummarizer()
        
        # Configuration Wikipedia
        wikipedia.set_lang("en")
        
        # Seuils de confiance
        self.min_confidence = 0.3
        self.max_suggestions = 8
        
        # Cache pour les recherches
        self.search_cache = {}
        
    def extract_keywords_tfidf(self, text, top_k=5):
        """
        Extrait les mots-clés en utilisant TF-IDF.
        
        Args:
            text (str): Texte utilisateur
            top_k (int): Nombre de mots-clés à extraire
            
        Returns:
            list: Liste des mots-clés avec leurs scores
        """
        # Prétraitement du texte
        cleaned_text = self.preprocessor.clean(text)
        
        # Si le modèle ML est disponible, utilise son vectorizer
        if self.ml_model.vectorizer is not None:
            vectorizer = self.ml_model.vectorizer
        else:
            # Vectorizer de secours
            vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            # Entraînement sur un corpus minimal
            dummy_corpus = [cleaned_text, "machine learning", "artificial intelligence"]
            vectorizer.fit(dummy_corpus)
        
        # Vectorisation
        tfidf_matrix = vectorizer.transform([cleaned_text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Extraction des scores TF-IDF
        scores = tfidf_matrix.toarray()[0]
        
        # Tri par score décroissant
        keyword_scores = [(feature_names[i], scores[i]) for i in range(len(feature_names)) if scores[i] > 0]
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return keyword_scores[:top_k]
    
    def extract_keywords_ml_confidence(self, text, top_k=5):
        """
        Extrait les mots-clés en utilisant la confiance du modèle ML.
        
        Args:
            text (str): Texte utilisateur
            top_k (int): Nombre de mots-clés à extraire
            
        Returns:
            list: Liste des mots-clés avec leurs scores de confiance
        """
        if self.ml_model.model is None:
            return self.extract_keywords_tfidf(text, top_k)
        
        # Découpage en phrases
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        keyword_scores = []
        
        for sentence in sentences:
            if len(sentence) < 10:  # Ignorer les phrases trop courtes
                continue
                
            # Prédiction avec le modèle ML
            try:
                prediction = self.ml_model.predict([sentence])[0]
                confidence = np.max(self.ml_model.predict_proba([sentence]))
                
                # Extraction des mots-clés de la phrase
                keywords = self.extract_keywords_tfidf(sentence, 3)
                
                for keyword, tfidf_score in keywords:
                    # Score combiné : TF-IDF * confiance du modèle
                    combined_score = tfidf_score * confidence
                    keyword_scores.append((keyword, combined_score))
                    
            except Exception as e:
                print(f"Erreur lors de la prédiction ML: {e}")
                continue
        
        # Tri et sélection des meilleurs
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        return keyword_scores[:top_k]
    
    def extract_named_entities(self, text):
        """
        Extrait les entités nommées (noms propres) du texte.
        
        Args:
            text (str): Texte utilisateur
            
        Returns:
            list: Liste des entités nommées
        """
        # Prétraitement avec POS-tagging
        try:
            cleaned_text, pos_info = self.preprocessor.clean_with_pos_info(text)
            
            # Extraction des noms propres (NNP, NNPS)
            named_entities = []
            for word, nltk_tag, wordnet_tag in pos_info:
                if nltk_tag.startswith('NNP'):  # Noms propres
                    named_entities.append(word)
            
            return named_entities
        except Exception as e:
            print(f"Erreur lors de l'extraction des entités nommées: {e}")
            return []
    
    def extract_keywords_advanced(self, text, top_k=8):
        """
        Extraction avancée des mots-clés combinant plusieurs méthodes.
        
        Args:
            text (str): Texte utilisateur
            top_k (int): Nombre de mots-clés à extraire
            
        Returns:
            list: Liste des mots-clés avec leurs scores
        """
        all_keywords = []
        
        # 1. Extraction TF-IDF
        tfidf_keywords = self.extract_keywords_tfidf(text, top_k=top_k//2)
        all_keywords.extend(tfidf_keywords)
        
        # 2. Extraction avec confiance ML
        ml_keywords = self.extract_keywords_ml_confidence(text, top_k=top_k//2)
        all_keywords.extend(ml_keywords)
        
        # 3. Extraction des entités nommées (priorité haute)
        named_entities = self.extract_named_entities(text)
        for entity in named_entities:
            all_keywords.append((entity, 1.0))  # Score maximum pour les entités nommées
        
        # 4. Agrégation et déduplication
        keyword_dict = {}
        for keyword, score in all_keywords:
            if keyword.lower() not in keyword_dict:
                keyword_dict[keyword.lower()] = score
            else:
                # Prendre le score maximum
                keyword_dict[keyword.lower()] = max(keyword_dict[keyword.lower()], score)
        
        # 5. Tri par score et sélection
        final_keywords = [(k, v) for k, v in keyword_dict.items()]
        final_keywords.sort(key=lambda x: x[1], reverse=True)
        
        return final_keywords[:top_k]
    
    def search_wikipedia_candidates(self, keywords, max_results=10):
        """
        Recherche les candidats Wikipedia pour les mots-clés donnés.
        
        Args:
            keywords (list): Liste des mots-clés avec scores
            max_results (int): Nombre maximum de résultats
            
        Returns:
            list: Liste des candidats avec scores de pertinence
        """
        candidates = []
        
        for keyword, score in keywords:
            if score < self.min_confidence:
                continue
                
            try:
                # Recherche Wikipedia
                search_results = wikipedia.search(keyword, results=5)
                
                for result in search_results:
                    # Calcul du score de pertinence
                    relevance_score = self.calculate_relevance_score(keyword, result, score)
                    
                    candidates.append({
                        'title': result,
                        'keyword': keyword,
                        'relevance_score': relevance_score,
                        'original_score': score
                    })
                    
            except Exception as e:
                print(f"Erreur lors de la recherche pour '{keyword}': {e}")
                continue
        
        # Tri par score de pertinence
        candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Déduplication par titre
        seen_titles = set()
        unique_candidates = []
        
        for candidate in candidates:
            if candidate['title'] not in seen_titles:
                seen_titles.add(candidate['title'])
                unique_candidates.append(candidate)
                
                if len(unique_candidates) >= max_results:
                    break
        
        return unique_candidates
    
    def calculate_relevance_score(self, keyword, page_title, original_score):
        """
        Calcule le score de pertinence d'une page Wikipedia.
        
        Args:
            keyword (str): Mot-clé de recherche
            page_title (str): Titre de la page Wikipedia
            original_score (float): Score original du mot-clé
            
        Returns:
            float: Score de pertinence
        """
        # Score de base
        relevance_score = original_score
        
        # Bonus pour correspondance exacte
        if keyword.lower() in page_title.lower():
            relevance_score += 0.5
        
        # Bonus pour correspondance au début du titre
        if page_title.lower().startswith(keyword.lower()):
            relevance_score += 0.3
        
        # Bonus pour longueur du titre (préfère les titres courts)
        title_length = len(page_title.split())
        if title_length <= 3:
            relevance_score += 0.2
        elif title_length <= 5:
            relevance_score += 0.1
        
        return relevance_score
    
    def get_page_summary(self, page_title, sentences=3):
        """
        Récupère le résumé d'une page Wikipedia.
        
        Args:
            page_title (str): Titre de la page
            sentences (int): Nombre de phrases dans le résumé
            
        Returns:
            dict: Résumé avec statut et contenu
        """
        try:
            # Récupération du résumé
            summary = wikipedia.summary(page_title, sentences=sentences)
            
            # Tentative de résumé avec l'autoencodeur si disponible
            if self.autoencoder.model is not None:
                try:
                    autoencoder_summary = self.autoencoder.summarize(summary, num_sentences=2)
                    return {
                        'status': 'success',
                        'title': page_title,
                        'summary': summary,
                        'autoencoder_summary': autoencoder_summary
                    }
                except Exception as e:
                    print(f"Erreur autoencodeur: {e}")
            
            return {
                'status': 'success',
                'title': page_title,
                'summary': summary
            }
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Page ambiguë
            return {
                'status': 'ambiguous',
                'title': page_title,
                'options': e.options[:5]  # Limiter à 5 options
            }
            
        except wikipedia.exceptions.PageError:
            return {
                'status': 'not_found',
                'title': page_title,
                'message': f"Page '{page_title}' non trouvée, kero! 🐸"
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'title': page_title,
                'message': f"Erreur lors de la récupération: {str(e)}"
            }
    
    def intelligent_search(self, user_input, max_suggestions=8):
        """
        Recherche Wikipedia intelligente complète.
        
        Args:
            user_input (str): Entrée utilisateur
            max_suggestions (int): Nombre maximum de suggestions
            
        Returns:
            dict: Résultats de la recherche
        """
        print(f"🔍 Recherche intelligente pour: '{user_input}'")
        
        # 1. Extraction des mots-clés
        keywords = self.extract_keywords_advanced(user_input, top_k=max_suggestions)
        
        if not keywords:
            return {
                'status': 'error',
                'message': "Impossible d'extraire des mots-clés pertinents, kero! 🐸"
            }
        
        print(f"📝 Mots-clés extraits: {[k[0] for k in keywords]}")
        
        # 2. Recherche des candidats Wikipedia
        candidates = self.search_wikipedia_candidates(keywords, max_results=max_suggestions*2)
        
        if not candidates:
            return {
                'status': 'error',
                'message': "Aucune page Wikipedia trouvée pour ces mots-clés, kero! 🐸"
            }
        
        # 3. Préparation des suggestions
        suggestions = []
        for i, candidate in enumerate(candidates[:max_suggestions]):
            suggestions.append({
                'id': i,
                'title': candidate['title'],
                'keyword': candidate['keyword'],
                'confidence': f"{candidate['relevance_score']:.2f}"
            })
        
        return {
            'status': 'success',
            'user_input': user_input,
            'keywords': keywords,
            'suggestions': suggestions,
            'message': f"J'ai trouvé {len(suggestions)} pages pertinentes, kero! 🐸"
        }
    
    def get_page_content(self, page_title):
        """
        Récupère le contenu complet d'une page.
        
        Args:
            page_title (str): Titre de la page
            
        Returns:
            dict: Contenu de la page
        """
        return self.get_page_summary(page_title, sentences=5)

# Instance globale
wikipedia_search = WikipediaIntelligentSearch()

def search_wikipedia_smart(query):
    """
    Fonction de recherche Wikipedia intelligente (interface compatible).
    
    Args:
        query (str): Requête utilisateur
        
    Returns:
        dict: Résultats de la recherche
    """
    return wikipedia_search.intelligent_search(query)

def get_wikipedia_summary(page_title):
    """
    Récupère le résumé d'une page Wikipedia.
    
    Args:
        page_title (str): Titre de la page
        
    Returns:
        dict: Résumé de la page
    """
    return wikipedia_search.get_page_summary(page_title) 