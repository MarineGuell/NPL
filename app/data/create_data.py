"""
Script de création de dataset enrichi pour le chatbot Kaeru
Sources : Wikipedia, ArXiv, The Conversation RSS
Format : Paragraphes entiers (plusieurs phrases)
Objectif : 3000 entrées par catégorie regroupée
Note : Pas de nettoyage/normalisation - c'est pour l'entraînement des modèles
"""

import wikipedia
import random
import csv
import nltk
import arxiv
import feedparser
import requests
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import sent_tokenize
import time
from bs4 import BeautifulSoup

# Téléchargement des données NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📥 Téléchargement automatique de punkt pour la création de données...")
    try:
        nltk.download('punkt', quiet=True)
        print("✅ punkt téléchargé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement de punkt: {e}")
        print("Le script continuera mais pourrait avoir des problèmes de tokenisation")

# ============================================================================
# CONFIGURATION DES CATÉGORIES REGROUPÉES
# ============================================================================

CATEGORIES_REGOUPEES = {
    'Life Sciences': {
        'sources': ['biology', 'microbiology', 'ecology'],
        'wiki_labels': ['Biology', 'Microbiology', 'Ecology'],
        'arxiv_codes': ['q-bio', 'q-bio.MN', 'q-bio.PE'],
        'rss_keywords': ['biology', 'microbiology', 'ecology', 'genetics', 'evolution']
    },
    'Physical Sciences': {
        'sources': ['astronomy', 'physics', 'climatology'],
        'wiki_labels': ['Astronomy', 'Physics', 'Climatology'],
        'arxiv_codes': ['astro-ph', 'physics', 'physics.ao-ph'],
        'rss_keywords': ['astronomy', 'physics', 'climate', 'space', 'energy']
    },
    'Historical Studies': {
        'sources': ['archaeology', 'history'],
        'wiki_labels': ['Archaeology', 'History'],
        'arxiv_codes': [],  # Pas de codes ArXiv pour l'histoire
        'rss_keywords': ['archaeology', 'history', 'ancient', 'heritage']
    },
    'Literature': {
        'sources': ['literature'],
        'wiki_labels': ['Literature'],
        'arxiv_codes': [],
        'rss_keywords': ['literature', 'books', 'poetry', 'writing']
    },
    'Social Sciences': {
        'sources': ['social science'],
        'wiki_labels': ['Social science'],
        'arxiv_codes': ['physics.soc-ph'],
        'rss_keywords': ['society', 'psychology', 'sociology', 'politics']
    },
    'Technology & Innovation': {
        'sources': ['new technology'],
        'wiki_labels': ['Technology'],
        'arxiv_codes': ['cs', 'cs.AI', 'cs.LG'],
        'rss_keywords': ['technology', 'innovation', 'artificial intelligence', 'robotics']
    },
    'Art and Culture': {
        'sources': ['art and culture'],
        'wiki_labels': ['Art', 'Culture'],
        'arxiv_codes': [],
        'rss_keywords': ['art', 'culture', 'music', 'theater', 'cinema']
    },
    'Economy': {
        'sources': ['economy'],
        'wiki_labels': ['Economics'],
        'arxiv_codes': ['q-fin'],
        'rss_keywords': ['economy', 'economics', 'finance', 'business']
    }
}

# Configuration globale
TARGET_PER_CATEGORY = 3000
TOTAL_TARGET = TARGET_PER_CATEGORY * len(CATEGORIES_REGOUPEES)
TEXTS_SET = set()
ROWS = []
CATEGORY_COUNTS = Counter()

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def extract_paragraphs_from_text(text, min_sentences=5, max_sentences=15):
    """Extrait des paragraphes cohérents d'un texte avec plus de phrases."""
    sentences = sent_tokenize(text)
    paragraphs = []
    
    if len(sentences) < min_sentences:
        return [text] if len(text) > 300 else []
    
    # Création de paragraphes avec plusieurs phrases
    current_paragraph = []
    for sentence in sentences:
        current_paragraph.append(sentence)
        
        # Créer un paragraphe quand on a assez de phrases
        if len(current_paragraph) >= min_sentences:
            paragraph_text = ' '.join(current_paragraph)
            if len(paragraph_text) > 300:  # Paragraphe plus long
                paragraphs.append(paragraph_text)
            current_paragraph = []
        
        # Limiter la taille du paragraphe
        if len(current_paragraph) >= max_sentences:
            paragraph_text = ' '.join(current_paragraph)
            if len(paragraph_text) > 300:
                paragraphs.append(paragraph_text)
            current_paragraph = []
    
    # Ajouter le dernier paragraphe s'il est assez long
    if current_paragraph:
        paragraph_text = ' '.join(current_paragraph)
        if len(paragraph_text) > 300:
            paragraphs.append(paragraph_text)
    
    return paragraphs

def extract_long_paragraphs_from_text(text, target_sentences=12, min_sentences=8):
    """Extrait des paragraphes longs avec beaucoup de phrases pour l'autoencodeur."""
    sentences = sent_tokenize(text)
    paragraphs = []
    
    if len(sentences) < min_sentences:
        return []
    
    # Création de paragraphes longs
    current_paragraph = []
    for sentence in sentences:
        current_paragraph.append(sentence)
        
        # Créer un paragraphe quand on a assez de phrases
        if len(current_paragraph) >= target_sentences:
            paragraph_text = ' '.join(current_paragraph)
            if len(paragraph_text) > 500:  # Paragraphe très long
                paragraphs.append(paragraph_text)
            current_paragraph = []
        
        # Limiter la taille du paragraphe (max 20 phrases)
        if len(current_paragraph) >= 20:
            paragraph_text = ' '.join(current_paragraph)
            if len(paragraph_text) > 500:
                paragraphs.append(paragraph_text)
            current_paragraph = []
    
    # Ajouter le dernier paragraphe s'il est assez long
    if current_paragraph and len(current_paragraph) >= min_sentences:
        paragraph_text = ' '.join(current_paragraph)
        if len(paragraph_text) > 500:
            paragraphs.append(paragraph_text)
    
    return paragraphs

def ensure_long_paragraphs_ratio(paragraphs, target_ratio=0.5, min_sentences=10):
    """
    S'assure qu'au moins target_ratio des paragraphes ont plus de min_sentences phrases.
    Si ce n'est pas le cas, essaie de créer des paragraphes plus longs.
    """
    if not paragraphs:
        return paragraphs
    
    # Compter les paragraphes longs
    long_paragraphs = 0
    for paragraph in paragraphs:
        sentences = sent_tokenize(paragraph)
        if len(sentences) >= min_sentences:
            long_paragraphs += 1
    
    current_ratio = long_paragraphs / len(paragraphs)
    
    if current_ratio >= target_ratio:
        return paragraphs
    
    print(f"⚠️ Ratio de paragraphes longs: {current_ratio:.2f} (objectif: {target_ratio})")
    print(f"   Paragraphes longs: {long_paragraphs}/{len(paragraphs)}")
    
    # Essayer de créer des paragraphes plus longs
    return paragraphs  # Pour l'instant, on garde les paragraphes existants

def add_text_if_valid(text, category):
    """Ajoute un texte s'il est valide et pas en doublon."""
    if (len(text) > 200 and 
        text not in TEXTS_SET and 
        CATEGORY_COUNTS[category] < TARGET_PER_CATEGORY):
        
        TEXTS_SET.add(text)
        ROWS.append((text, category))
        CATEGORY_COUNTS[category] += 1
        return True
    return False

# ============================================================================
# 1. COLLECTE WIKIPEDIA
# ============================================================================

def fetch_wikipedia_paragraphs(category_name, wiki_labels):
    """Collecte des paragraphes depuis Wikipedia."""
    print(f"📚 Collecting Wikipedia paragraphs for {category_name}...")
    
    for wiki_label in wiki_labels:
        if CATEGORY_COUNTS[category_name] >= TARGET_PER_CATEGORY:
            break
            
        try:
            # Recherche de pages
            results = wikipedia.search(wiki_label, results=500)
            random.shuffle(results)
            
            for title in tqdm(results[:100], desc=f"Processing {wiki_label}"):
                if CATEGORY_COUNTS[category_name] >= TARGET_PER_CATEGORY:
                    break
                    
                try:
                    # Récupération du contenu complet
                    page = wikipedia.page(title)
                    content = page.content
                    
                    # Extraction de paragraphes (essayer d'abord les longs)
                    paragraphs = extract_long_paragraphs_from_text(content)
                    
                    # Si pas assez de paragraphes longs, utiliser la méthode normale
                    if len(paragraphs) < 3:
                        paragraphs = extract_paragraphs_from_text(content)
                    
                    # S'assurer du ratio de paragraphes longs
                    paragraphs = ensure_long_paragraphs_ratio(paragraphs)
                    
                    for paragraph in paragraphs:
                        if add_text_if_valid(paragraph, category_name):
                            pass  # Paragraphe ajouté avec succès
                        else:
                            break  # Catégorie pleine
                            
                except Exception as e:
                    continue
                    
                time.sleep(0.1)  # Respecter les limites de l'API
                
        except Exception as e:
            print(f"⚠️ Erreur pour {wiki_label}: {e}")
            continue

# ============================================================================
# 2. COLLECTE ARXIV
# ============================================================================

def fetch_arxiv_paragraphs(category_name, arxiv_codes):
    """Collecte des abstracts depuis ArXiv et les transforme en paragraphes."""
    if not arxiv_codes:
        return
        
    print(f"🧪 Collecting ArXiv paragraphs for {category_name}...")
    
    client = arxiv.Client()

    for code in arxiv_codes:
        if CATEGORY_COUNTS[category_name] >= TARGET_PER_CATEGORY:
            break
            
        try:
            search = arxiv.Search(
                query=f"cat:{code}",
                max_results=1000,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            for result in tqdm(client.results(search), desc=f"Processing {code}"):
                if CATEGORY_COUNTS[category_name] >= TARGET_PER_CATEGORY:
                    break
                    
                abstract = result.summary
                
                # Créer un paragraphe à partir de l'abstract
                if len(abstract) > 200:
                    # Ajouter le titre pour enrichir le contexte
                    enriched_text = f"{result.title}. {abstract}"
                    
                    # Extraction de paragraphes (essayer d'abord les longs)
                    paragraphs = extract_long_paragraphs_from_text(enriched_text)
                    
                    # Si pas assez de paragraphes longs, utiliser la méthode normale
                    if len(paragraphs) < 2:
                        paragraphs = extract_paragraphs_from_text(enriched_text)
                    
                    # S'assurer du ratio de paragraphes longs
                    paragraphs = ensure_long_paragraphs_ratio(paragraphs)
                    
                    for paragraph in paragraphs:
                        if add_text_if_valid(paragraph, category_name):
                            pass
                        else:
                            break
                            
        except Exception as e:
            print(f"⚠️ Erreur ArXiv pour {code}: {e}")
            continue

# ============================================================================
# 3. COLLECTE THE CONVERSATION RSS
# ============================================================================

def fetch_conversation_rss(category_name, keywords):
    """Collecte des articles depuis The Conversation RSS."""
    if not keywords:
        return
        
    print(f"📰 Collecting The Conversation RSS for {category_name}...")
    
    # URL RSS de The Conversation
    rss_url = "https://theconversation.com/global/articles.rss"
    
    try:
        feed = feedparser.parse(rss_url)
        
        for entry in tqdm(feed.entries[:500], desc="Processing RSS entries"):
            if CATEGORY_COUNTS[category_name] >= TARGET_PER_CATEGORY:
                break
                
            # Vérifier si l'article correspond aux mots-clés
            title = entry.title.lower()
            summary = entry.summary.lower()
            
            matches_keyword = any(keyword.lower() in title or keyword.lower() in summary 
                                for keyword in keywords)
            
            if matches_keyword:
                try:
                    # Récupérer le contenu complet de l'article
                    response = requests.get(entry.link, timeout=10)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extraire le contenu principal
                    content_div = soup.find('div', class_='content-body')
                    if content_div:
                        content = content_div.get_text()
                        
                        # Extraction de paragraphes (essayer d'abord les longs)
                        paragraphs = extract_long_paragraphs_from_text(content)
                        
                        # Si pas assez de paragraphes longs, utiliser la méthode normale
                        if len(paragraphs) < 2:
                            paragraphs = extract_paragraphs_from_text(content)
                        
                        # S'assurer du ratio de paragraphes longs
                        paragraphs = ensure_long_paragraphs_ratio(paragraphs)
                        
                        for paragraph in paragraphs:
                            if add_text_if_valid(paragraph, category_name):
                                pass
                            else:
                                break
                                
                except Exception as e:
                    continue
                    
                time.sleep(0.5)  # Respecter les limites du serveur
                
    except Exception as e:
        print(f"⚠️ Erreur RSS pour {category_name}: {e}")

# ============================================================================
# EXÉCUTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale de collecte de données."""
    print("🚀 Début de la collecte de données enrichie...")
    print(f"📊 Objectif : {TARGET_PER_CATEGORY} entrées par catégorie")
    print(f"🎯 Total cible : {TOTAL_TARGET} entrées")
    print("📝 Note : Pas de nettoyage/normalisation - pour l'entraînement des modèles")
    
    # Configuration Wikipedia
    wikipedia.set_lang("en")
    
    # Collecte pour chaque catégorie regroupée
    for category_name, config in CATEGORIES_REGOUPEES.items():
        print(f"\n{'='*60}")
        print(f"🎯 TRAITEMENT DE LA CATÉGORIE : {category_name}")
        print(f"{'='*60}")
        
        # 1. Collecte Wikipedia
        fetch_wikipedia_paragraphs(category_name, config['wiki_labels'])
        
        # 2. Collecte ArXiv
        fetch_arxiv_paragraphs(category_name, config['arxiv_codes'])
        
        # 3. Collecte The Conversation RSS
        fetch_conversation_rss(category_name, config['rss_keywords'])
        
        print(f"✅ {category_name}: {CATEGORY_COUNTS[category_name]}/{TARGET_PER_CATEGORY} entrées collectées")
    
    # ============================================================================
    # SAUVEGARDE FINALE
    # ============================================================================
    
    print(f"\n{'='*60}")
    print("💾 SAUVEGARDE DU DATASET")
    print(f"{'='*60}")
    
    # Mélange final
    random.shuffle(ROWS)
    
    # Sauvegarde
    output_file = 'enriched_dataset_paragraphs.csv'
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'category'])
        writer.writerows(ROWS)
    
    print(f"✅ Dataset sauvegardé : {len(ROWS)} entrées dans {output_file}")
    print("\n📊 Distribution finale par catégorie :")
    for category, count in CATEGORY_COUNTS.items():
        print(f"  {category}: {count}/{TARGET_PER_CATEGORY} ({count/TARGET_PER_CATEGORY*100:.1f}%)")
    
    # Statistiques sur la longueur des textes
    text_lengths = [len(row[0]) for row in ROWS]
    print(f"\n📏 Statistiques de longueur :")
    print(f"  Moyenne : {sum(text_lengths)/len(text_lengths):.0f} caractères")
    print(f"  Min : {min(text_lengths)} caractères")
    print(f"  Max : {max(text_lengths)} caractères")

if __name__ == "__main__":
    main() 