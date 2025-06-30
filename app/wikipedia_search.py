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
            
            # Vérification que le contenu correspond au titre demandé
            # Si le titre récupéré est différent du titre demandé, c'est suspect
            try:
                page = wikipedia.page(page_title)
                actual_title = page.title
                
                # Si le titre réel est très différent, c'est probablement une redirection incorrecte
                if actual_title.lower() != page_title.lower():
                    print(f"⚠️  Attention: Titre demandé '{page_title}' -> Titre réel '{actual_title}'")
                    
                    # Vérifier si le contenu semble correct en cherchant des mots-clés
                    summary_lower = summary.lower()
                    page_title_lower = page_title.lower()
                    
                    # Si le titre demandé n'apparaît pas dans le contenu, c'est suspect
                    if page_title_lower not in summary_lower:
                        print(f"❌ Contenu ne correspond pas au titre demandé '{page_title}'")
                        
                        # Essayer de trouver une page plus appropriée
                        alternative_search = wikipedia.search(page_title, results=5)
                        for alt_result in alternative_search:
                            if alt_result.lower() != actual_title.lower():
                                try:
                                    alt_page = wikipedia.page(alt_result)
                                    alt_summary = wikipedia.summary(alt_result, sentences=sentences)
                                    
                                    # Vérifier si cette page semble plus appropriée
                                    if page_title_lower in alt_summary.lower():
                                        print(f"✅ Page alternative trouvée: '{alt_result}'")
                                        summary = alt_summary
                                        actual_title = alt_result
                                        break
                                except:
                                    continue
            except:
                pass  # Si on ne peut pas vérifier, on continue avec le résumé original
            
            # Tentative de résumé avec l'autoencodeur si disponible
            if self.autoencoder.model is not None:
                try:
                    autoencoder_summary = self.autoencoder.summarize(summary, num_sentences=2)
                    return {
                        'status': 'success',
                        'title': actual_title if 'actual_title' in locals() else page_title,
                        'summary': summary,
                        'autoencoder_summary': autoencoder_summary
                    }
                except Exception as e:
                    print(f"Erreur autoencodeur: {e}")
            
            return {
                'status': 'success',
                'title': actual_title if 'actual_title' in locals() else page_title,
                'summary': summary
            }
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Page ambiguë - filtrer les options valides
            valid_options = []
            for option in e.options[:10]:  # Tester plus d'options
                try:
                    # Vérifier si l'option est accessible
                    test_summary = wikipedia.summary(option, sentences=1)
                    if test_summary and len(test_summary) > 10:  # Vérifier que le résumé n'est pas vide
                        valid_options.append(option)
                        if len(valid_options) >= 5:  # Limiter à 5 options valides
                            break
                except:
                    continue  # Ignorer les options qui ne fonctionnent pas
            
            if valid_options:
                return {
                    'status': 'ambiguous',
                    'title': page_title,
                    'options': valid_options
                }
            else:
                # Si aucune option valide, essayer la recherche directe
                try:
                    search_results = wikipedia.search(page_title, results=3)
                    if search_results:
                        best_match = search_results[0]
                        summary = wikipedia.summary(best_match, sentences=sentences)
                        return {
                            'status': 'success',
                            'title': best_match,
                            'summary': f"Found similar page '{best_match}': {summary}",
                            'note': f"Original search: {page_title}"
                        }
                except:
                    pass
                
                return {
                    'status': 'not_found',
                    'title': page_title,
                    'message': f"Page '{page_title}' non trouvée et aucune alternative valide, kero! 🐸"
                }
            
        except wikipedia.exceptions.PageError:
            # Page non trouvée - essayer des alternatives
            try:
                # Rechercher des pages similaires
                search_results = wikipedia.search(page_title, results=3)
                if search_results:
                    # Essayer la première suggestion
                    alternative_title = search_results[0]
                    try:
                        alt_summary = wikipedia.summary(alternative_title, sentences=sentences)
                        return {
                            'status': 'success',
                            'title': alternative_title,
                            'summary': f"Page '{page_title}' not found, but here's information about '{alternative_title}': {alt_summary}",
                            'note': f"Original search: {page_title}"
                        }
                    except:
                        pass
            except:
                pass
            
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
            'message': f"I'm not sure about the concept you are looking for. I recognized {len(suggestions)} of them, kero! 🐸"
        }
    
    def smart_search_by_combinations(self, user_input, max_suggestions=8):
        """
        Recherche Wikipedia intelligente par combinaisons de mots.
        
        Logique :
        1. Si input ≤ 5 mots : cherche page avec ce nom exact
        2. Si pas trouvé : teste combinaisons de 4 mots, puis 3, etc.
        3. Si input > 5 mots : retire stopwords et retente
        4. Si toujours > 5 mots : garde seulement noms et adjectifs
        5. Si rien trouvé : utilise la méthode actuelle
        
        Args:
            user_input (str): Entrée utilisateur
            max_suggestions (int): Nombre maximum de suggestions
            
        Returns:
            dict: Résultats de la recherche
        """
        print(f"🔍 Recherche par combinaisons pour: '{user_input}'")
        
        # Nettoyage de base
        cleaned_input = user_input.strip()
        words = cleaned_input.split()
        print(f"📝 Mots détectés: {words} ({len(words)} mots)")
        
        # ÉTAPE 1: Si ≤ 5 mots, chercher page avec ce nom exact
        if len(words) <= 5:
            print("🐸 Étape 1: Recherche exacte (≤ 5 mots)")
            exact_search = self._search_exact_page(cleaned_input)
            if exact_search['status'] == 'success':
                return exact_search
        
        # ÉTAPE 2: Si ≤ 5 mots, tester combinaisons décroissantes
        if len(words) <= 5:
            print("🐸 Étape 2: Test des combinaisons décroissantes")
            for combo_size in range(len(words)-1, 0, -1):
                print(f"   Test des combinaisons de {combo_size} mots")
                combinations = self._generate_combinations(words, combo_size)
                
                for combo in combinations:
                    combo_text = ' '.join(combo)
                    print(f"   Test: '{combo_text}'")
                    result = self._search_exact_page(combo_text)
                    if result['status'] == 'success':
                        return result
        
        # ÉTAPE 3: Si > 5 mots, retirer les stopwords
        if len(words) > 5:
            print("🐸 Étape 3: Retrait des stopwords")
            words_without_stopwords = self._remove_stopwords(words)
            print(f"   Mots après retrait stopwords: {words_without_stopwords}")
            
            if len(words_without_stopwords) <= 5:
                # Retester avec les combinaisons
                for combo_size in range(len(words_without_stopwords), 0, -1):
                    combinations = self._generate_combinations(words_without_stopwords, combo_size)
                    for combo in combinations:
                        combo_text = ' '.join(combo)
                        result = self._search_exact_page(combo_text)
                        if result['status'] == 'success':
                            return result
        
        # ÉTAPE 4: Si toujours > 5 mots, garder seulement noms et adjectifs
        if len(words) > 5:
            print("🐸 Étape 4: Extraction noms et adjectifs")
            pos_words = self.preprocessor.extract_verbs_adjectives_nouns(cleaned_input)
            nouns_and_adjectives = pos_words['nouns'] + pos_words['adjectives']
            print(f"   Noms et adjectifs: {nouns_and_adjectives}")
            
            if len(nouns_and_adjectives) <= 5:
                # Retester avec les combinaisons
                for combo_size in range(len(nouns_and_adjectives), 0, -1):
                    combinations = self._generate_combinations(nouns_and_adjectives, combo_size)
                    for combo in combinations:
                        combo_text = ' '.join(combo)
                        result = self._search_exact_page(combo_text)
                        if result['status'] == 'success':
                            return result
        
        # ÉTAPE 5: Si rien trouvé, utiliser la méthode actuelle
        print("🐸 Étape 5: Utilisation de la méthode actuelle")
        return self.intelligent_search(user_input, max_suggestions)

    def _search_exact_page(self, search_term):
        """
        Recherche une page Wikipedia avec un terme exact.
        
        Args:
            search_term (str): Terme de recherche
            
        Returns:
            dict: Résultat de la recherche
        """
        try:
            # Recherche directe
            search_results = wikipedia.search(search_term, results=5)
            
            if not search_results:
                return {'status': 'not_found'}
            
            # Vérifier si le premier résultat correspond exactement
            best_match = search_results[0]
            
            # Correspondance exacte (insensible à la casse)
            if search_term.lower() == best_match.lower():
                print(f"   ✅ Correspondance exacte trouvée: '{best_match}'")                
                return {
                    'status': 'success',
                    'user_input': search_term,
                    'keywords': [(search_term, 1.0)],
                    'suggestions': [{
                        'id': 0,
                        'title': best_match,
                        'keyword': search_term,
                        'confidence': '1.00'
                    }],
                    'message': f"Found exact match for '{search_term}', kero! 🐸"
                }
            
            # Correspondance partielle (le terme est dans le titre)
            if search_term.lower() in best_match.lower():
                print(f"   ✅ Correspondance partielle trouvée: '{best_match}'")
                return {
                    'status': 'success',
                    'user_input': search_term,
                    'keywords': [(search_term, 0.8)],
                    'suggestions': [{
                        'id': 0,
                        'title': best_match,
                        'keyword': search_term,
                        'confidence': '0.80'
                    }],
                    'message': f"Found partial match for '{search_term}', kero! 🐸"
                }
            
            return {'status': 'not_found'}
            
        except Exception as e:
            print(f"   ❌ Erreur lors de la recherche: {e}")
            return {'status': 'error', 'message': str(e)}

    def _generate_combinations(self, words, size):
        """
        Génère toutes les combinaisons possibles de mots de taille donnée.
        
        Args:
            words (list): Liste des mots
            size (int): Taille des combinaisons
            
        Returns:
            list: Liste des combinaisons
        """
        from itertools import combinations
        
        if size > len(words):
            return []
        
        # Générer toutes les combinaisons
        word_combinations = list(combinations(words, size))
        
        # Convertir en listes de chaînes
        result = []
        for combo in word_combinations:
            result.append(list(combo))
        
        return result

    def _remove_stopwords(self, words):
        """
        Retire les stopwords d'une liste de mots.
        
        Args:
            words (list): Liste des mots
            
        Returns:
            list: Liste sans stopwords
        """
        try:
            # Utiliser les stopwords de NLTK
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            
            # Filtrer les stopwords
            filtered_words = [word for word in words if word.lower() not in stop_words]
            
            return filtered_words
        except Exception as e:
            print(f"Erreur lors du retrait des stopwords: {e}")
            return words
    
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