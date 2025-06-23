import wikipedia, random, csv, nltk
import arxiv
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import sent_tokenize

# Téléchargement des données NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Cibles : catégories et libellés wiki
categories = {
    'astronomy': 'Astronomy',
    'biology': 'Biology',
    'microbiology': 'Microbiology',
    'literature': 'Literature',
    'history': 'History',
    'archaeology': 'Archaeology',
    'social science': 'Social science',
    'new technology': 'Technology',
    'physics': 'Physics',
    'ecology': 'Ecology',
    'climatology': 'Climatology'
}

arxiv_codes = {
    'astronomy': 'astro-ph',
    'biology': 'q-bio',
    'microbiology': 'q-bio.MN',
    'physics': 'physics',
    'ecology': 'q-bio.PE',
    'social science': 'physics.soc-ph',
}

texts = set()
rows = []
TARGET = 50000
per_cat = TARGET // len(categories)
category_counts = Counter()

# ---------- 1. Wikipedia ----------
def fetch_wikipedia(cat_key, wiki_label):
    results = wikipedia.search(wiki_label, results=1000)
    random.shuffle(results)
    for title in results:
        if category_counts[cat_key] >= per_cat:
            break
        try:
            extract = wikipedia.summary(title, sentences=5) # Un peu plus de phrases
            line = (extract.replace('\\n', ' ').strip(), cat_key)
            if line not in texts and len(line[0]) > 100:
                texts.add(line)
                rows.append(line)
                category_counts[cat_key] += 1
        except Exception:
            continue

print("📚 Collecting from Wikipedia...")
for cat, wiki_label in tqdm(categories.items()):
    fetch_wikipedia(cat, wiki_label)

# ---------- 2. ArXiv abstracts (with sentence augmentation) ----------
client = arxiv.Client()

print("🧪 Fetching and augmenting from ArXiv...")
for cat, code in tqdm(arxiv_codes.items()):
    if category_counts[cat] >= per_cat:
        continue # Catégorie déjà pleine
        
    search = arxiv.Search(
        query=f"cat:{code}",
        max_results=per_cat,  # Pas besoin de sur-échantillonner, on augmente
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    try:
        for result in client.results(search):
            if category_counts[cat] >= per_cat:
                break
            
            abstract = result.summary.replace('\\n', ' ').strip()
            # Découper l'abstract en phrases
            sentences = sent_tokenize(abstract)
            
            for sentence in sentences:
                if category_counts[cat] >= per_cat:
                    break
                
                # Ajouter la phrase si elle est assez longue
                if len(sentence) > 50:
                    line = (sentence, cat)
                    if line not in texts:
                        texts.add(line)
                        rows.append(line)
                        category_counts[cat] += 1
                        
    except arxiv.UnexpectedEmptyPageError:
        print(f"⚠️ Moins de résultats que prévu pour la catégorie {cat}, mais augmentation en cours.")
        continue

# ---------- Final shuffle & export ----------
rows = rows[:TARGET]
random.shuffle(rows)

with open('balanced_dataset_50k.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['text', 'category'])
    writer.writerows(rows)

print(f"✅ Dataset generated: {len(rows)} rows saved to balanced_dataset_50k.csv.")
print("Distribution par catégorie :")
print(category_counts)
