import os
import re
import string
import chardet
from stemming.lovins import stem as lovins_stem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Membaca semua file dalam folder
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

# Preprocessing dasar (lowercase + hapus tanda baca)
def preprocess(text):
    text = text.lower()
    text = re.sub(rf"[{string.punctuation}]", "", text)
    return text.split()

def stem_text(text):
    words = preprocess(text)
    stemmed_words = []
    for w in words:
        if len(w) < 3:
            stemmed_words.append(w)
            continue
        try:
            stemmed_words.append(lovins_stem(w))
        except Exception:
            stemmed_words.append(w)
    return ' '.join(stemmed_words)

def build_stem_map(texts):
    mapping = {}
    for text in texts:
        words = preprocess(text)
        for word in words:
            if len(word) < 3:
                continue
            try:
                root = lovins_stem(word)
            except Exception:
                root = word
            if root not in mapping:
                mapping[root] = set()
            mapping[root].add(word)
    return mapping

# Evaluasi UI, OI, MWC
def evaluate_stemming_map(stem_map):
    total_conflated_sets = len(stem_map)
    over_stemmed = sum(1 for words in stem_map.values() if len(words) > 5)
    under_stemmed = sum(1 for words in stem_map.values() if len(words) == 1)
    avg_conflation = sum(len(words) for words in stem_map.values()) / total_conflated_sets
    UI = under_stemmed / total_conflated_sets
    OI = over_stemmed / total_conflated_sets
    MWC = avg_conflation
    return UI, OI, MWC

# Fungsi pencarian menggunakan TF-IDF
def search(texts, query):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    query_stemmed = stem_text(query)
    query_vec = vectorizer.transform([query_stemmed])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    return scores

# Path dokumen
path_indo = "C:\\Users\\Lenovo\\Downloads\\BIND"
path_eng = "C:\\Users\\Lenovo\\Downloads\\BING"

# Baca file
indo_texts = read_all_files(path_indo)
eng_texts = read_all_files(path_eng)

# Lakukan stemming (menggunakan Lovins di kedua bahasa)
indo_stemmed = [stem_text(t) for t in indo_texts]
eng_stemmed = [stem_text(t) for t in eng_texts]

# Evaluasi
indo_map = build_stem_map(indo_texts)
eng_map = build_stem_map(eng_texts)

ui_id, oi_id, mwc_id = evaluate_stemming_map(indo_map)
ui_en, oi_en, mwc_en = evaluate_stemming_map(eng_map)

print(f"\nIndonesia (with Lovins Stemmer) - UI: {ui_id:.4f}, OI: {oi_id:.4f}, MWC: {mwc_id:.2f}")
print(f"English   (with Lovins Stemmer) - UI: {ui_en:.4f}, OI: {oi_en:.4f}, MWC: {mwc_en:.2f}")

