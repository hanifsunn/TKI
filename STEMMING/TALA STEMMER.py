import os
import re
import string
import chardet
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Stemmer
ps = PorterStemmer()

# Kamus kata dasar (bisa diperluas)
root_words = set([
    "politik", "ekonomi", "olahraga", "kesehatan", "pendidikan", 
    "sakit", "ajar", "main", "sehat", "kerja", "didik", "guna", "pakai",
    "tahu", "percaya", "ubah", "kaji", "nilai", "teknologi", "finance",
    "government", "education", "sport", "health", "play", "study", "run"
])

def is_root_word(word):
    return word in root_words

def tala_stemmer(word):
    word = re.sub(r'(lah|kah|tah|pun)$', '', word)
    if is_root_word(word): return word

    word = re.sub(r'(ku|mu|nya)$', '', word)
    if is_root_word(word): return word

    word = re.sub(r'^(me|di|ke|se|ber|ter|per)', '', word)
    if is_root_word(word): return word

    word = re.sub(r'^(pe|be|te)', '', word)
    if is_root_word(word): return word

    word = re.sub(r'(kan|an|i)$', '', word)
    return word

def read_all_files(folder_path):
    texts = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
            try:
                texts.append(raw_data.decode(encoding))
            except (UnicodeDecodeError, LookupError):
                texts.append(raw_data.decode('ISO-8859-1', errors='replace'))
    return texts

def preprocess(text):
    text = text.lower()
    text = re.sub(rf"[{string.punctuation}]", "", text)
    return text.split()

def stem_text(text, lang='id'):
    words = preprocess(text)
    if lang == 'id':
        return ' '.join([tala_stemmer(w) for w in words])
    elif lang == 'en':
        return ' '.join([ps.stem(w) for w in words])
    return text

def build_stem_map(texts, lang='id'):
    mapping = {}
    for text in texts:
        words = preprocess(text)
        for word in words:
            root = tala_stemmer(word) if lang == 'id' else ps.stem(word)
            if root not in mapping:
                mapping[root] = set()
            mapping[root].add(word)
    return mapping

def evaluate_stemming_map(stem_map):
    total_conflated_sets = len(stem_map)
    over_stemmed = sum(1 for words in stem_map.values() if len(words) > 5)
    under_stemmed = sum(1 for words in stem_map.values() if len(words) == 1)
    avg_conflation = sum(len(words) for words in stem_map.values()) / total_conflated_sets
    UI = under_stemmed / total_conflated_sets
    OI = over_stemmed / total_conflated_sets
    MWC = avg_conflation
    return UI, OI, MWC

def search(texts, query, lang='id'):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    query_stemmed = stem_text(query, lang=lang)
    query_vec = vectorizer.transform([query_stemmed])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    return scores

# ===== MAIN PROCESS =====

path_indo = "C:\\Users\\Lenovo\\Downloads\\BIND"
path_eng = "C:\\Users\\Lenovo\\Downloads\\BING"

indo_texts = read_all_files(path_indo)
eng_texts = read_all_files(path_eng)

# Lakukan stemming pada semua teks
indo_stemmed = [stem_text(t, lang='id') for t in indo_texts]
eng_stemmed = [stem_text(t, lang='en') for t in eng_texts]

# Evaluasi UI, OI, MWC
indo_map = build_stem_map(indo_texts, lang='id')
eng_map = build_stem_map(eng_texts, lang='en')

ui_id, oi_id, mwc_id = evaluate_stemming_map(indo_map)
ui_en, oi_en, mwc_en = evaluate_stemming_map(eng_map)

print(f"\nIndonesia - UI: {ui_id:.4f}, OI: {oi_id:.4f}, MWC: {mwc_id:.2f}")
print(f"English   - UI: {ui_en:.4f}, OI: {oi_en:.4f}, MWC: {mwc_en:.2f}")

