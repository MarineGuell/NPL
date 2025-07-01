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

# # Ajout du chemin pour les imports
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import TextPreprocessor
from model_autoencodeur import AutoencoderSummarizer
from model_lstm import DLModel
from model_tfidf import MLModel
from mon_tokenizer import Tokenizer

class WikipediaIntelligentSearch:
    def __init__(self):
        wikipedia.set_lang("en")

    def get_page_summary(self, page_title):
        """
        Récupère le résumé d'une page Wikipedia.
        """
        try:
            summary = wikipedia.summary(page_title)
            page = wikipedia.page(page_title)
            return {
                'status': 'success',
                'title': page.title,
                'summary': summary
            }
        except wikipedia.exceptions.DisambiguationError:
            return {
                'status': 'not_found',
                'title': page_title,
                'message': f"Page '{page_title}' is ambiguous or not found, kero! 🐸"
            }
        except wikipedia.exceptions.PageError:
            return {
                'status': 'not_found',
                'title': page_title,
                'message': f"Page '{page_title}' not found, kero! 🐸"
            }
        except Exception as e:
            return {
                'status': 'error',
                'title': page_title,
                'message': f"Error: {str(e)}"
            }
